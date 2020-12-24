import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, use_aux1=True, use_aux2=True, init_weights=True):
        super(GoogleNet, self).__init__()
        self.use_aux1 = use_aux1
        self.use_aux2 = use_aux2

        self.until_aux1 = nn.Sequential(OrderedDict([
            ('conv1', BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)),
            ('maxPool1', nn.MaxPool2d(3, stride=2, ceil_mode=True)),
            # ceil_mode=True 当按照给定的kernel_size, stride不能恰好对原图进行maxpool时自动padding
            # ceil_mode=False 舍弃最后几列/行
            ('conv2', BasicConv2d(64, 192, kernel_size=1)),
            ('conv3', BasicConv2d(192, 192, kernel_size=3, padding=1)),
            ('maxPool2', nn.MaxPool2d(3, stride=2, ceil_mode=True)),
            ('inception3a', Inception(192, 64, 96, 128, 16, 32, 32)),
            ('inception3b', Inception(256, 128, 128, 192, 32, 96, 64)),
            ('maxPool3', nn.MaxPool2d(3, stride=2, ceil_mode=True)),
            ('inception4a', Inception(480, 192, 96, 208, 16, 48, 64))]))

        self.aux1_to_aux2 = nn.Sequential(OrderedDict([
            ('inception4b', Inception(512, 160, 112, 224, 24, 64, 64)),
            ('inception4c', Inception(512, 128, 128, 256, 24, 64, 64)),
            ('inception4d', Inception(512, 112, 144, 288, 32, 64, 64))]))

        self.aux2_to_end = nn.Sequential(OrderedDict([
            ('inception4e', Inception(528, 256, 160, 320, 32, 128, 128)),
            ('maxPool4', nn.MaxPool2d(3, stride=2, ceil_mode=True)),
            ('inception5a', Inception(832, 256, 160, 320, 32, 128, 128)),
            ('inception5b', Inception(832, 384, 192, 384, 48, 128, 128)),
            ('avgPool', nn.AdaptiveAvgPool2d((1, 1))),  # target size:(1, 1) 通过pooling得到的是1x1的feature map
            ('dropout', nn.Dropout(p=0.4)),
            ('flatten', torch.nn.Flatten(start_dim=1, end_dim=-1)),
            ('fc', nn.Linear(1024, num_classes))]))

        if self.use_aux1:
            self.aux1 = InceptionAux(512, num_classes)
        if self.use_aux2:
            self.aux2 = InceptionAux(528, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.until_aux1(x)
        if self.use_aux1 and self.training:
            aux1 = self.aux1(x)
        x = self.aux1_to_aux2(x)
        if self.use_aux2 and self.training:
            aux2 = self.aux2(x)
        x = self.aux2_to_end(x)

        if self.aux1 and self.aux2 and self.training:
            return x, aux1, aux2
        if self.aux1 and self.training:
            return x, aux1
        if self.aux2 and self.training:
            return x, aux2
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, ch_pool):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, ch_pool, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]  # list
        return torch.cat(outputs, 1)  # [N,C,H,W] 沿维度Channel拼接


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.AveragePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.AveragePool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)  # 延channel维度展开，即得到由每个batch中所有图片所得的全部feature map拼接成的平面
        x = F.dropout(x, p=0.5, training=self.training)  # 等价于nn.Dropout(p=0.5)(x)
        x = self.fc1(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x



