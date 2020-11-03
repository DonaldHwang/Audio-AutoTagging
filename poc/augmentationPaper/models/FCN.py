#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as torchdata
import torch.optim as optim
import numpy as np

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


class FCN_legacy(nn.Module):
    def __init__(self, block=ConvBlock, input_channels=1, output_shape=80,
                 filters_num = [32, 32, 32, 32, 32],
                 filters_size = [(3,3), (3,3), (3,3), (3,3), (3,3)],
                 max_pool = [(2,2), (2,2), (2,2), (4,4), (4,4)]):
        super(FCN_legacy, self).__init__()

        assert len(filters_num) == len(filters_size), "Wrong configuration for FCN_legacy model"
        assert len(filters_size) == len(max_pool), "Wrong configuration for FCN_legacy model"

        self.layer1 = block(input_channels, filters_num[0], filters_size[0], max_pool[0])
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

        # This ones applies the sigmoid in the loss, so the output layer should be logits
        return nn.BCEWithLogitsLoss()

    def forward_with_evaluation(self, x):
        # if the output layers is the scores (logits), we need sigmoid for evaluation
        return torch.sigmoid(self.forward(x))


def init_weights(m):
    if type(m) == nn.Conv1d:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)



def generate_data(shape, n, n_classes=50, multi=False):
    data, labels = [], []
    labs = np.arange(0, n_classes)
    for _ in range(n):
        data.append(np.random.normal(size=(shape[0], shape[1], shape[2])))
        lab = np.zeros(n_classes)
        # Set only
        if multi == True:
            # Pick amount of labels:
            nlab = np.random.randint(1, n_classes)
            set_true = np.random.choice(labs, size=nlab, replace=False)
            for l in set_true:
                lab[l] = 1
        else:
            lab[np.random.randint(0, n_classes-1)] = 1
        labels.append(lab)
    data = np.array(data)
    labels = np.array(labels)
    data = torch.from_numpy(data)
    data = data.float()

    combined = []
    for i in range(len(data)):
        combined.append([data[i], labels[i]])

    return combined



def main():
    '''
    Unit test for the FCN_legacy model.
    This test generaters a fake dataset of 1 data point, computes the loss before training, then trains.
    If the model is correct, and given enough capacity, the model should overfit quite well.

    :return:
    '''

    # Set up learning rate and directory where to save the weights
    import math

    learning_rate = 0.01
    datapints = 1
    freq_bins = 128
    frames = 128
    batch = 1
    n_classes = 100
    epochs = 3000 if n_classes < 20 else n_classes * 200
    multi_label = True  # True use sigmoid, False use softmax
    use_logits = True  # True for BCEWithLogitsLoss(), so no sigmoid in last layer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up model. Either flex or simple
    # model = simple_sampleCNN()
    model = FCN_legacy(input_channels=1,
                       output_shape=n_classes).to(device)

    loss_f = model.loss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print(model)


    # Generate random data to test the model
    # Here input size is (1,59049), batch size is batch and there are datapints data points
    data = generate_data((1, freq_bins, frames), datapints * batch, n_classes, multi=multi_label)
    data = torchdata.DataLoader(data, batch_size=batch)

    # Initial lose:
    x, target = data.dataset[0]
    x, target = x.unsqueeze(0).to(device), torch.from_numpy(target.astype(np.float32)).unsqueeze(0).to(device)
    model.eval()
    out = model(x)
    loss = loss_f(out, target)
    #print('Initial loss = {:.6f}'.format(loss.item()))

    # For BCELoss, log(0.5) = 0.69, assumign equal probablity for 2 clases
    if loss_f.__repr__() == nn.BCELoss().__repr__() or loss_f.__repr__() == nn.BCEWithLogitsLoss().__repr__():
        expecte_loss = -math.log(1/n_classes)
        if not multi_label:
            prob = 1 / n_classes
        else:
            prob = 0.5
        expecte_loss = (-math.log(prob) - (n_classes - 1) * math.log(1 - (prob))) / n_classes
        assert np.allclose(loss.item(), expecte_loss, atol=1.e-1), \
            'Loss = {} , expected = {}'.format(loss.item(), expecte_loss)
    model.train()
    ctr = 0
    for epoch in range(epochs):
        ctr += 1
        for _, (x, target) in enumerate(data):
            x, target = x.to(device), target.to(device)
            model.zero_grad()
            out = model(x)

            target = target.float()
            loss = loss_f(out, target)
            loss.backward()
            optimizer.step()
            if ctr % 50 == 0:
                print('loss {:.6f}'.format(loss.item()))
                print('outputs : {}'.format(out.detach().cpu().numpy()))

    model.eval()
    out = model.forward_with_evaluation(x)
    print('outputs : {}'.format(out.detach().cpu().numpy()))
    a = out.detach().cpu().numpy()
    b = target.detach().cpu().numpy()
    assert np.allclose(a, b, atol=1.e-1), 'Wrong outputs'

    print("Unit test completed.")


if __name__ == "__main__":
    main()