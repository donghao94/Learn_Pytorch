import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from model import GoogleNet
import json
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

# dataset path
root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
data_path = os.path.join(root_path, "data_set", "flower_data")
assert os.path.exists(data_path), "{} path does not exist.".format(data_path)

# train data
train_set = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=data_transform["train"])
train_num = len(train_set)
batch_size = 32
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
print("Using {} dataloader workers every process.".format(nw))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
# validation data
val_set = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform["val"])
val_num = len(val_set)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=nw)
# 'num': class
flower_list = train_set.class_to_idx
class_dict = dict((val, name) for name, val in flower_list.items())
json_str = json.dumps(class_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# net,loss_function,optimizer
net = GoogleNet(num_classes=5, use_aux1=True, use_aux2=True, init_weights=True)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)

best_acc = 0.0
save_path = './GoogleNet.pth'

for epoch in range(30):
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs, aux1, aux2 = net(images.to(device))
        loss0 = loss_function(outputs, labels.to(device))
        loss1 = loss_function(aux1, labels.to(device))
        loss2 = loss_function(aux2, labels.to(device))
        loss = loss0 + 0.3 * loss1 + 0.3 * loss2  # 将主分类器于辅助分类器的loss按照1:0.3:0.3的比例加起来最终的loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # 验证过程中不使用辅助分类器
            predict = torch.max(outputs, dim=1)[1]
            acc += (predict == val_labels.to(device)).sum().item()
        acc = acc / val_num
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / (step+1), acc))

print('Finished Training')
