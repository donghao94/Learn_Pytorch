import torch
import torch.nn as nn


class VGGNet(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGGNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 多套模型的参数
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(2, stride=2)]
        else:
            con2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            in_channels = v
            layers += [con2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)  # 返回一个函数sequential

def vgg(model_name='vgg13', **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model name {} is not in cfgs dict!".format(model_name))
        exit(-1)

    features = make_features(cfg)
    model = VGGNet(features, **kwargs)
    return model
