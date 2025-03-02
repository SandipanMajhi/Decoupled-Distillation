import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNet18_cifar10(nn.Module):
    def __init__(self, is_pretrained = True):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained = is_pretrained)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.fc = nn.Linear(512, 10)
        self.resnet18.maxpool = nn.Identity()

        for param in self.resnet18.parameters():
            param.requires_grad = True
        
    
    def forward(self, x):
        return self.resnet18(x)
    

class ResNet32_cifar10(nn.Module):
    def __init__(self, is_pretrained = True):
        super().__init__()
        self.resnet32 = torchvision.models.resnet34(pretrained = is_pretrained)
        self.resnet32.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet32.fc = nn.Linear(512, 10)
        self.resnet32.maxpool = nn.Identity()

        for param in self.resnet32.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        return self.resnet32(x)