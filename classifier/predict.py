import torchvision.transforms as transforms
import torch
from PIL import Image
import json
from model import vgg

data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

try:
    json_file = open('./num_classes.json', 'r')
    classes = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

im = Image.open('./test.jpeg')  # [H, W, C]
im = data_transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # transform to batch:[batch_size, C, H, W]

model_name = 'vgg13'
net = vgg(mode_name=model_name, num_classes=5, init_weights=True)
net.load_state_dict(torch.load('./VGGNet_{}.pth'.format(model_name)))
net.eval()

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].numpy()
print(classes[str(predict[0])])  # [4] -> 4 -> '4'
