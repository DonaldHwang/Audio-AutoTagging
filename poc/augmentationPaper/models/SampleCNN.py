import torch
import torch.nn as nn
import torch.utils.data as torchdata
import torch.optim as optim
import numpy as np


class flex_sampleCNN(nn.Module):
    # More flexible model with parameters m and n
    # Inputs:
    # m is filter size (pooling size) of convolution layers
    # n number of modules

    def __init__(self, m=None,
                 in_channels=1, debug_mode=False, dropout=True,
                 num_classes=50, multi_label= True, use_logits=True,
                 n_filters=[128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512]):
        super(flex_sampleCNN, self).__init__()
        # Inputs: m is kernel_size, n is amount of conv layers, n_filters is initial amount of channels
        # eval is bool whether model is testing or training, dropout is bool whether to apply dropout
        # filter nums is a list of filter sizes

        # First and last layers have are slightly different that middle layers

        # Make sure that the filter size list is in relation with n

        # Defaul is kernel size 3 for all layers except the last layer, with 1
        if m is None:
            m = [3] * len(n_filters)
            m[-1] = 1

        n = len(m) - 2
        assert len(n_filters) == n + 2
        assert len(m) == n + 2
        self.use_logits = use_logits
        # self.n_filters = n_filters
        self.input_size = m[0] * m[0] ** n
        # self.eval = eval
        self.in_channels = in_channels
        self.n_filters = n_filters
        # First layer
        self.mods = nn.ModuleList()

        first = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=n_filters[0],
                      kernel_size=m[0], stride=m[0], padding=0),
            nn.BatchNorm1d(n_filters[0]),
            nn.ReLU()
        )
        self.mods.append(first)

        for i in range(1, n + 1):
            # ic, oc = self.get_channels(i)
            # print(f'i: {i}, ic: {ic}, oc: {oc}')
            layer = nn.Sequential(
                nn.Conv1d(in_channels=n_filters[i-1], out_channels=n_filters[i],
                          kernel_size=m[i], stride=1, padding=1, bias=False),


                nn.ReLU(),
                nn.BatchNorm1d(n_filters[i]),  # BN after RELu, should be better, but I have to confirm this

                nn.MaxPool1d(kernel_size=m[i], stride=m[i]),
            )
            self.mods.append(layer)

        last = nn.Sequential(
            nn.Conv1d(in_channels=n_filters[-2], out_channels=n_filters[-1],
                      kernel_size=m[-1], stride=1, padding=0),
            )
        # if batch_size = 1, Batchnorm will fail in the last layer.
        # https://github.com/pytorch/pytorch/issues/7716
        if not debug_mode:
            last.add_module('batchnorm', nn.BatchNorm1d(n_filters[-1]))
        last.add_module('relu', nn.ReLU())

        if dropout:
            last.add_module("dropout", nn.Dropout(p=0.5))

        self.mods.append(last)

        self.full = nn.Linear(n_filters[-1], num_classes)
        self.activation = nn.Sigmoid() if multi_label else nn.Softmax()

    def forward(self, x):
        # input shape is batch_size x input_channels x width
        inp = x
        x = x.view(x.shape[0], self.in_channels, -1)

        for layer in self.mods:
            x = layer(x)

        x = x.view(inp.shape[0], self.n_filters[-1])
        x = self.full(x)
        if not self.use_logits:
            x = self.activation(x)

        return x

    def forward_with_evaluation(self, x):
        if self.use_logits:
            # Same as forward bu added sigmoid activation
            x = self.activation(self.forward(x))
        else:
            x = self.forward(x)
        return x

    def loss(self):
        if self.use_logits:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.BCELoss()

    # Old function, probably not nescessary anymore
    # Can be modified to generate parameter lists?
    def get_channels(self, i):
        # Double the initial filter size at range indices
        # By default 3th and 8th layer, ranges = [7,2]
        ranges = [7, 2]
        for index, r in enumerate(ranges):
            if i >= r:
                if i == r:
                    return 2 ** (len(ranges) - index - 1) * self.n_filters, 2 ** (len(ranges) - index) * self.n_filters
                else:
                    return 2 ** (len(ranges) - index) * self.n_filters, 2 ** (len(ranges) - index) * self.n_filters
        return self.n_filters, self.n_filters


class simple_sampleCNN(nn.Module):

    # Same model as in the paper
    def __init__(self):
        super(simple_sampleCNN, self).__init__()
        # Input size: 59049
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128,
                      kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.ReLU()
        )
        # Increase channels from 128 to 256
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.ReLU()
        )
        # Increase channels from 256 to 512
        self.conv9 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.full = nn.Linear(512, 50)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        # input shape is minbatch size x input_channels x width
        inp = inp.view(inp.shape[0], 1, -1)

        out = self.conv1(inp)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)

        out = out.view(inp.shape[0], out.size(1) * out.size(2))

        out = self.full(out)
        out = self.sigmoid(out)

        return out


def generate_data(shape, n, n_classes=50, multi=False):
    data, labels = [], []
    labs = np.arange(0, n_classes)
    for _ in range(n):
        data.append(np.random.normal(size=(shape[0], shape[1])))
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
    Unit test for the SampleCNN model.
    This test generaters a fake dataset of 1 data point, computes the loss before training, then trains.
    If the model is correct, and given enough capacity, the model should overfit quite well.

    :return:
    '''

    # Set up learning rate and directory where to save the weights
    import math

    learning_rate = 0.001
    datapints = 1
    batch = 1
    n_classes = 50
    epochs = 1000 if n_classes < 20 else n_classes * 200
    multi_label = True  # True use sigmoid, False use softmax
    use_logits = True  # True for BCEWithLogitsLoss(), so no sigmoid in last layer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up model. Either flex or simple
    # model = simple_sampleCNN()
    model = flex_sampleCNN(debug_mode=True,
                           num_classes=n_classes,
                           multi_label=multi_label,
                           use_logits=use_logits).to(device)
    loss_f = model.loss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print(model)


    # Generate random data to test the model
    # Here input size is (1,59049), batch size is batch and there are datapints data points
    data = generate_data((1, 59049), datapints * batch, n_classes, multi=multi_label)
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
    for epoch in range(epochs):
        for ctr, (x, target) in enumerate(data):
            x, target = x.to(device), target.to(device)
            model.zero_grad()
            out = model(x)

            target = target.float()
            loss = loss_f(out, target)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print('Epoch: {} / {} , loss {:.6f}'.format(epoch, epochs, loss.item()))
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