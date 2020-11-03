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
        out = f.dropout2d(self.maxpool(out), 0.5, training=self.training)
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

        #self.fc0 = nn.Linear(filters_num[4], 512)
        #self.fc1 = nn.Linear(512, output_shape)
        self.fc = nn.Linear(filters_num[4], output_shape)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        #out = self.fc(out)

        #out = self.fc0(out)
        #out = torch.sigmoid(self.fc1(out))
        #out = torch.sigmoid(self.fc(out))

        out = self.fc(out)

        #out = out.log()

        return out

    def loss(self):
        #return nn.NLLLoss()
        #return nn.BCELoss()   # BE expects to outputs with sigmoid
        return nn.BCEWithLogitsLoss()   # This ones applies the sigmoid in the loss, so the output layer should be logits

    def forward_with_evaluation(self, x):
        #return torch.exp(self.forward(x))
        return torch.sigmoid(self.forward(x))  # if the output layers is the scores (logits), we need sigmoid for evsaluateion


class ExampleCNN(nn.Module):
    def __init__(self):
        super(ExampleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        x = f.relu(f.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = self.fc2(x)
        #return f.log_softmax(x, dim=1) # if using NLLloss
        return x  # if using cross entropy loss, do note that softmax needs to happen outside to view returns
    
    def loss(self):
        return nn.CrossEntropyLoss()  # cross entropy cobmines log softmax and NLL, so no softmax in forward pass
        #return nn.NLLLoss()
        #return f.nll_loss

    def forward_with_evaluation(self, x):
        return f.softmax(self.forward(x), dim=1)


class CNN(nn.Module):
    def __init__(self, num_class=15):
        super(CNN, self).__init__()

        # init bn
        self.bn_init = nn.BatchNorm2d(1)

        # layer 1
        self.conv_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.mp_1 = nn.MaxPool2d((2, 4))

        # layer 2
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.mp_2 = nn.MaxPool2d((2, 4))

        # layer 3
        self.conv_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.mp_3 = nn.MaxPool2d((2, 4))

        # layer 4
        self.conv_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.mp_4 = nn.MaxPool2d((3, 5))

        # layer 5
        self.conv_5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)
        self.mp_5 = nn.MaxPool2d((4, 4))

        # classifier
        self.dense = nn.Linear(64, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #x = x.unsqueeze(1) # disable if using transorms, because the batch size is already [batch, channel, width, ehight]

        # init bn
        x = self.bn_init(x)

        # layer 1
        x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))

        # layer 2
        x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))

        # layer 3
        x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))

        # layer 4
        x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))

        # layer 5
        x = self.mp_5(nn.ELU()(self.bn_5(self.conv_5(x))))

        # classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logit = nn.Sigmoid()(self.dense(x))

        return logit
