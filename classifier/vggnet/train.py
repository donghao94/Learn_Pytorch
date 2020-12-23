import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from model import vgg
import os
import json
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device is {}'.format(device))

data_transforms = {
    "train": transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ),
    "validate": transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
}

data_path = '../data_set/flower_data'
assert os.path.exists(data_path), "{} path does not exit.".format(data_path)
# train_set
train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=data_transforms["train"])
train_num = len(train_set)
# train_loader
batch_size = 32
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
print('Using {} dataloader workers every process.'.format(nw))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
# val_set
val_set = datasets.ImageFolder(os.path.join(data_path,'val'), transform=data_transforms["validate"])
val_num = len(val_set)
print("Using {} images for training and {} images for validation.".format(train_num, val_num))
# val_loader
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=nw)
# 创建并保存分类对应表
flower_list = train_set.class_to_idx
class_dict = dict((num, name) for name, num in flower_list.items())
class_json = json.dumps(class_dict, indent=4)
with open('num_classes.json', 'w') as json_file:
    json_file.write(class_json)
# 实例化net, loss_function, optimizer
model_name = 'vgg13'
net = vgg(model_name=model_name, num_classes=5, init_weights=True)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
# 设置当前训练模型所得参数的保存路径
save_path = './VGGNet_{}.pth'.format(model_name)
best_acc = 0.0

for epoch in range(10):
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        # t2 = time.perf_counter()
        outputs = net(images.to(device))  # images 是batch: [N, C, H, W]
        # print(time.perf_counter()-t2)
        loss = loss_function(outputs, labels.to(device))
        # print(time.perf_counter()-t2)
        loss.backward()  # 用时最长
        # print(time.perf_counter() - t2)
        optimizer.step()
        # print(time.perf_counter() - t2)  # 训练一个batch所需时间
        running_loss += loss
        # 进度条
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\r [{}{}]{:^3.0f}% train loss:{:.3f}".format(a, b, int(rate * 100), loss), end="")
    print()
    # 训练一遍全部训练数据所需时间
    print(time.perf_counter() - t1)

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data
            val_outputs = net(val_images.to(device))
            predict = torch.max(val_outputs, dim=1)[1].numpy()
            acc += (predict == val_labels).sum().item()
        acc = acc / val_num
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)
    print("Epoch {%d}: accuracy = {%.3f}, loss = {%.3f}".format(epoch+1, acc, running_loss / batch_size))

print('FINISHED TRAINING')


