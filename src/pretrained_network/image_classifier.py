import torchvision

from torchvision import models

# torchvision 自带的所有模型
# print(models.list_models())

# Uppercase names correspond to classes that 
# implement poular architectures for computer vision.
alexnet = models.AlexNet()


# Lowercase names are functions that instantiate models
# with predefined numbers of layers and units and optionally
# download and load pretrained weights into them.

# 预加载权重
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# print(vit)

# transforms，allow us to quickly define pipelines of 
# basic preprocessing functions
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

from PIL import Image
img = Image.open("./data/5922.jpg")
# img.show()

img_t = preprocess(img)

import torch
batch_t = torch.unsqueeze(img_t, 0)

# inference in the eval model
vit.eval()
out = vit(batch_t)

with open('./data/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)

# normalize outputs to the range [0, 1] and divide by the sum
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# get the actual numerical value using .item()
print(labels[index.item()], percentage[index.item()].item())