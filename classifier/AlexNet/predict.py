import torch
import torchvision.transforms as transforms
from PIL import Image
from model import AlexNet
import json

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

try:
    json_file = open('./class_indices.json', 'r')
    classes = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

net = AlexNet(num_classes=5, init_weights=True)
net.load_state_dict(torch.load('./AlexNet.pth'))
net.eval()

im = Image.open('Download.jpeg')  # [H,W,C]
im = transform(im)  # [C,H,W]
im = torch.unsqueeze(im, dim=0)  # [N,C,H,W] 给tensor添加一个维度：batchsize

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()  # [4]
print(classes[str(predict[0])])
