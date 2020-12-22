import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt

# def main():
transform = transforms.Compose([
    transforms.ToTensor(),  # (H x W x C)-->(C x H x W) [0,255]-->[0.0,1.0]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize((mean),(std)) : 每个通道的均值和方差
                                                       # (input[channel] - mean[channel])/std
])  # 将多种高transforms的操作结合成一个操作
# 50000张训练图片
# 第一次使用download = True下载数据集
train_set = torchvision.datasets.CIFAR10(root='./data_set/CIFAR10_data', train=True,
                                         download=False, transform=transform)
# 每次随机（shuffle=True）从train_set中load 36张图片 num_works:载入数据的线程数（windows下只能设为0）
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                           shuffle=False, num_workers=2)

# 10000张验证图片
val_set = torchvision.datasets.CIFAR10(root='./data_set/CIFAR10_data', train=False,
                                       download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                         shuffle=False, num_workers=0)

# 将val_loader中的内容转换成一个迭代器，通过next()这个方法每次调用一个Batch，此处只有一个Batch，即一次性load 10000张图片
val_data_iter = iter(val_loader)
val_image, val_label = val_data_iter.next()  # val_image是10000张图片，val_label：[batch,1]

# classes 是一个元组。与列表的区别在于元组的元素不能改变，元组使用()，列表使用[]
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
loss_function = nn.CrossEntropyLoss()  # CrossEntropyLoss() 内置了softmax()
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 用Adam优化器对LeNet的参数进行优化，learning rate是0。001

# 训练过程

for epoch in range(5):  # 将全部训练集迭代5次
    running_loss = 0.0
    # 以下循环将训练集中的全部50000张图片训练一次
    for step, data in enumerate(train_loader, start=0):  # train_loader 包含了很多个Batch,每个Batch包含36张图片的data，
        # 每循环一次,load一个Batch，也就是36张图片，step加一
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，
        # 一般用在 for 循环当中
        inputs, labels = data  # data is a list of [inputs, labels]

        # zero the parameter gradients 每一次训练一个Batch开始前都需要先将历史损失梯度归零
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()  # 将loss反向传播
        optimizer.step()  # 更新参数（每训练完一个batch更新一次参数）

        # 用测试集测试上面的出的参数的准确率并print statistics
        # 虽然这不是训练过程的一部分，但是通过这个我们可以掌控在训练过程中的准确度是否是逐渐提升的
        running_loss += loss.item()  # 将每张图片的loss累加，最后计算每训练500张的loss的平均值
        if step % 500 == 499:  # 每训练500张图片测试一次
            with torch.no_grad():  # 接下来的操作不计算误差梯度
                outputs = net(val_image)  # 将全部10000张图片放到网络里进行计算
                # 每一张图片的输出都为一个1x10的行向量，总共有batch张图片，outputs：[batch,10]
                predict_y = torch.max(outputs, dim=1)[1]  # 在列维度查找，即查找每一行的最大值，[1]即得出最大值的索引，索引即分类
                # predict_y:[batch,1]
                accuracy = (predict_y == val_label).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  valid_accuracy: %.3f' % (epoch+1, step+1, running_loss/500, accuracy))
                running_loss = 0.0

print('Finished Training')

# 保存训练所得参数
save_path = './LeNet.pth'
torch.save(net.state_dict(), save_path)


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# imshow(torchvision.utils.make_grid(val_image))
# print(' '.join('%5s' % classes[val_label[j]] for j in range(4)))
