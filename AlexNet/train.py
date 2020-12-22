import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪成224x224，
                                 # 这一步不只可以将图片resize而且可以增加训练样本量
                                 transforms.RandomHorizontalFlip(),  # 随机翻转，增大训练样本量
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]),
    "valid": transforms.Compose([transforms.Resize((224, 224)),  # 不能是224 必须是(224,224)
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

image_path = "../data_set/flower_data"
assert os.path.exists(image_path), "{} path does not exit.".format(image_path)
# train_set
train_set = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
train_num = len(train_set)
# train_loader
batch_size = 32
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
print('Using {} dataloader workers every process.'.format(nw))
# num_workers一般是cpu的核心数。作用是，在for循环训练的时候加快寻batch的速度，因为下一轮迭代的batch很可能在之前已经加载到RAM里了，所以比较占运行内存。
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=nw)
# val_set
val_set = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["valid"])
val_num = len(val_set)
# val_loader
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=nw)
print('Using {} images for training, {} images for validation.'.format(train_num, val_num))

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_set.class_to_idx
# {0:'daisy', 1:'dandelion', 2:'roses', 3:'sunflower', 4:'tulips'}
class_dict = dict((val, cla) for cla, val in flower_list.items())
# write dict into json file
json_str = json.dumps(class_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# 实例化net, loss_function, optimizer
net = AlexNet(num_classes=5, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './AlexNet.pth'
best_acc = 0.0

for epoch in range(10):
    # train
    net.train()
    # 在model里用到了dropout函数，在训练时为了减少过拟合要使用dropout，
    # 但是在测试时不需要用dropout，所以在测试时使用net.eval(),这样可以去掉dropout
    running_loss = 0.0
    t1 = time.perf_counter()  # 此刻时间,为了方便后面计算训练所需时间
    # 循环batch
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1) / len(train_loader)
        # 进度条
        a = "*" * int(rate * 50)
        b = "." * int((1-rate) * 50)
        print("\r [{}{}]{:^3.0f}% train loss:{:.3f}".format(a, b, int(rate * 100), loss), end="")
    print()
    # 训练一个batch所需时间
    print(time.perf_counter()-t1)

    ############ 全部训练数据的一次训练结束 #########

    # validation
    net.eval()  # 使dropout失效
    acc = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = val_data
            outputs = net(val_inputs.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        acc = acc / val_num
        # 保存准确率最高的训练对应的参数
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)
        print('[epoch:%d] train_loss: %.3f test_accuracy: %.3f' % (epoch + 1, running_loss / batch_size, acc))

print('Finished Training')
