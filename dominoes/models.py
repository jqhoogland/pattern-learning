import sys
import logging
import random
from dataclasses import dataclass
import itertools

import torch as t
import numpy as np
import torchvision
import einops
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from torchtyping import TensorType
from scipy.optimize import linprog

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class ExtendedModule(nn.Module):
    def predict(self, x):
        with t.no_grad():
            x = self.forward(x)
            return F.log_softmax(x, dim=1)
    
    @property
    def num_params(self):
        return sum((p.numel() for p in self.parameters()))

    @property
    def hyperparams(self):
        return {
            "model_name": self.__class__.__name__,
            "num_params": self.num_params
        }


class DominoDetector(ExtendedModule):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.conv3 = nn.Conv2d(64, 128, 3, 1)
        
        # self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc1 = nn.Linear(32 * 15 * 31, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.conv3(x)
        # x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = t.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
        

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

class ResNet8(ExtendedModule):
    def __init__(self, num_classes=10):
        super(ResNet8, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(
            ResNetBlock(16, 16),
            ResNetBlock(16, 16)
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(16, 32, stride=2),
            ResNetBlock(32, 32)
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(32, 64, stride=2),
            ResNetBlock(64, 64)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

