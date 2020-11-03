#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as f


# Residual block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_pool):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=max_pool)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = f.dropout2d(self.maxpool(out), 0.5)
        return out


class FCN(nn.Module):
    def __init__(self, block, input_shape=1, output_shape=80,
                  filters_num = [32, 32, 32, 32, 32],
                  filters_size = [(3,3), (3,3), (3,3), (3,3), (3,3)],
                  max_pool = [(2,2), (2,2), (2,2), (2,2), (2,2)]):
        super(FCN, self).__init__()
        self.layer1 = block(1, filters_num[0], filters_size[0], max_pool[0])
        self.layer2 = block(filters_num[0], filters_num[1], filters_size[1], max_pool[1])
        self.layer3 = block(filters_num[1], filters_num[2], filters_size[2], max_pool[2])
        self.layer4 = block(filters_num[2], filters_num[3], filters_size[3], max_pool[3])
        self.layer5 = block(filters_num[3], filters_num[4], filters_size[4], max_pool[4])

        self.fc = nn.Linear(filters_num[4], output_shape)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = f.sigmoid(self.fc(out))
        return out

class FCN(nn.Module):
    def __init__(self, block, input_shape=1, output_shape=80,
                  filters_num = [32, 32, 32, 32, 32],
                  filters_size = [(3,3), (3,3), (3,3), (3,3), (3,3)],
                  max_pool = [(2,2), (2,2), (2,2), (2,2), (2,2)]):
        super(FCN, self).__init__()
        self.layer1 = block(1, filters_num[0], filters_size[0], max_pool[0])
        self.layer2 = block(filters_num[0], filters_num[1], filters_size[1], max_pool[1])
        self.layer3 = block(filters_num[1], filters_num[2], filters_size[2], max_pool[2])
        self.layer4 = block(filters_num[2], filters_num[3], filters_size[3], max_pool[3])
        self.layer5 = block(filters_num[3], filters_num[4], filters_size[4], max_pool[4])

        self.fc = nn.Linear(filters_num[4], output_shape)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = f.sigmoid(self.fc(out))
        return out


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, input_shape, output_shape,
                  filters_num = [32, 32, 32, 32, 32],
                  filters_size = [(3,3), (3,3), (3,3), (3,3), (3,3)],
                  max_pool = [(2,2), (2,2), (2,2), (2,2), (2,2)]):
        super(ConvNet, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv2d(1, filters_num[0], kernel_size=filters_size[0], padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pool[0]))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))


        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
