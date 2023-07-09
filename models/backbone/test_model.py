import torch
from torchvision import datasets,transforms,models


model = models.resnet50(pretrained=False)


x = torch.zeros(2, 3, 224, 224)

y = model(x)
for u in y:
    print(u.shape)
print(model.out_channels)