import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride = stride,
                                padding=1,
                                bias = False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride = 1,
                                padding=1,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)   
        return out


class ResNet8(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()

        self.in_channels = 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = BasicBlock(in_channels=self.in_channels, out_channels=16, stride = 1)
        self.layer2 = BasicBlock(in_channels=16, out_channels=32, stride = 2)
        self.layer3 = BasicBlock(in_channels=32, out_channels=64, stride = 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc_layer(out)
        return out