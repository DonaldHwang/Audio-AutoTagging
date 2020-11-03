#!/usr/bin/env python3'
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

from constants import Constants
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import numpy as np
import warnings

def check_outputs(outputs, multilabel=False):
    '''Validates that the outpus has valid values for all data points.
    In multiclass=False mode , the all columns should sum to 1 (softmax).
    In multiclass=True mode, all columns shouuld be in the range of 0 to 1 (sigmoid)'''

    if torch.is_tensor(outputs):
        outputs = outputs.cpu().numpy()

    if np.any(np.isnan(outputs)):
        return False

    if not multilabel:
        # Check that the sum of all colums == 1, or close to it
        tmp = outputs.sum(axis=1)
        tmp2 = np.isclose(tmp, np.ones_like(tmp))
        return np.all(tmp2)
    else:
        # check that no column is negative or larger than 1, for all data samples
        tmp = np.invert(np.logical_or(outputs < 0, outputs > 1))
        return np.all(tmp)


def view_image(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def computeMeanAndSTD(dataset):
    '''Computes the mean and std of the dataset'''
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset.transform = transform

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, (data, labels) in enumerate(dataloader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data.numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    return pop_mean, pop_std0, pop_std1

## Good tutorial
## https://nextjournal.com/gkoehler/pytorch-mnist

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 128
num_epochs = 20
batch_size = 256

constants = Constants(3)



# Image processing
train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
    transforms.Normalize(mean=[0.5,0.5,0.5],  # 3 for RGB channels
                         std=[0.5,0.5,0.5])])

test_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
    transforms.Normalize(mean=[0.5,0.5,0.5],  # 3 for RGB channels
                         std=[0.5,0.5,0.5])
    #torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

# MNIST dataset
train_mnist = torchvision.datasets.MNIST(root=constants.MNIST,
                                         train=True,
                                         transform=train_transform,
                                         download=True)

test_mnist = torchvision.datasets.MNIST(root=constants.MNIST,
                                         train=False,
                                         transform=test_transform,
                                         download=True)

# Data loader
train_data_loader = torch.utils.data.DataLoader(dataset=train_mnist,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=3)

test_data_loader = torch.utils.data.DataLoader(dataset=test_mnist,
                                                batch_size=1000,
                                                shuffle=False,
                                               num_workers=3)


allMeans, allStd00, allStd01 = computeMeanAndSTD(train_mnist)

train_mnist.transform = train_transform

## ############################################################
## Look at some data
## ############################################################

examples = enumerate(test_data_loader)
batch_ids, (example_data, example_labels) = next(examples)

tmp = example_data[0:32,:,:,:]
tmp = torchvision.utils.make_grid(tmp.reshape(tmp.size(0), tmp.size(1), image_size, image_size),nrow=4)
plt.figure()
plt.imshow(view_image(tmp, nrow=4))
plt.savefig('data.png')



## ############################################################
## Model stuff
## ############################################################


from cnn import FCN
from cnn import ConvBlock
from cnn import ExampleCNN
import torch.nn as nn
import torch.nn.functional as f
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# input shape (batch, 1, 128, 1366)
#model = FCN(ConvBlock, output_shape=80,
#            max_pool=[(2,4), (2,4), (2,4), (3,5), (4,4)]).to(device)
#from torchsummary import summary
#summary(model, input_size=(1, 128, 1366))

# input shape (batch, 1, 256, 256)
model = FCN(ConvBlock, output_shape=len(train_mnist.classes),
            max_pool=[(2,2), (2,2), (2,2), (4,4), (8,8)],
            filters_num=[32, 32, 32, 32, 32]).to(device)

#model = ExampleCNN()

summary(model, input_size=(3, image_size, image_size))


learning_rate = 0.001
num_epochs = 5
print_every = 10

# Loss and optimizer

criterion = model.loss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## ############################################################
## Training
## ############################################################

# Train the model
t_sum = 0
t_epoch = 0
last_time = time.time()
last_epoch = time.time()
all_time = []

total_step = len(train_data_loader)
curr_lr = learning_rate
loss_history_train = []

train_loss_step = 0.0
valid_loss_step = 0.0
train_loss_epoch = 0.0
valid_loss_epoch = 0.0
train_loss_history = []
valid_loss_history = []

train_steps = 0
valid_steps = 0
for epoch in range(num_epochs):
    train_loss_epoch = 0.0
    valid_loss_epoch = 0.0
    train_loss_step = 0.0
    valid_loss_step = 0.0

    # Training loop

    pbar = tqdm(enumerate(train_data_loader))
    for i, (images, labels) in pbar:
    #for i, sample in enumerate(trainloader):
        model.train()

        t = time.time()
        t_diff = t - last_time
        t_sum += t_diff
        last_time = t
        all_time.append(t_diff)

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, torch.zeros(len(labels), len(train_mnist.classes)).to(device).scatter_(1, labels.unsqueeze(1), 1.))
        # outputs.cpu().detach().numpy()  # evaluate outputs like this

        #train_loss_epoch += loss.item() * images.size(0) / len(train_dataset)  # avg loss of batch, times batch size
        train_loss_epoch += loss.item()
        train_loss_step = loss.item()

        train_steps += 1
        # Append losses for plotting
        train_loss_history.append(train_loss_step)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        myString = "Epoch [{}/{}], Step [{}/{}],   T_diff {} , T_total {} , train_step_Loss: {:.4f}" \
                   .format(epoch + 1, num_epochs,
                           i + 1, total_step,
                           t_diff, t_sum,
                           loss.item())

        #pbar.set_description(myString)
        #print(myString)

        if train_steps % print_every == 0:
            model.eval()

            with torch.no_grad():
                sample = next(iter(test_data_loader))
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, torch.zeros(len(labels), len(train_mnist.classes)).to(device).scatter_(1, labels.unsqueeze(1), 1.))

                # train_loss_epoch += loss.item() * images.size(0) / len(train_dataset)  # avg loss of batch, times batch size
                valid_loss_step = loss.item()

                valid_steps += 1
                # Append losses for plotting
                valid_loss_history.append(valid_loss_step)

            pbar.set_description("Epoch [{}/{}], Step [{}/{}],   T_diff {} , T_total {} , train_Loss: {:.4f} , valid_Loss {:.4f}"
                                 .format(epoch + 1, num_epochs, i + 1, total_step,
                                         t_diff, t_sum, train_loss_step, valid_loss_step))

    t = time.time()
    t_epoch = t - last_epoch
    last_epoch = t
    print("Epoch [{}/{}] , time per epoch {} , train_loss {} , valid_loss {}"
          .format(epoch + 1, num_epochs, t_epoch,
                  train_loss_epoch, valid_loss_epoch))

    # # Decay learning rate
    # if (epoch + 1) % 20 == 0:
    #     curr_lr /= 3
    #     update_lr(optimizer, curr_lr)



## Plot time for each step
t = np.linspace(0, num_epochs, num=len(all_time), endpoint=False)

plt.figure()
plt.plot(t, all_time, 'r-+', label='Time per step')
plt.legend()
plt.savefig('Output_time_per_step.png')
plt.xlabel('Epochs')
plt.ylabel('Time (s)')
plt.title("Train data = %d" %(len(train_mnist)))
plt.show()


## Plot loss
t = np.linspace(0, num_epochs, num=train_steps, endpoint=False)
t_valid = np.linspace(0, num_epochs, num=valid_steps, endpoint=False)

plt.figure()
plt.plot(t, train_loss_history, 'r-+', label='Train')
plt.plot(t_valid, valid_loss_history, 'b--o', label='Valid')
plt.legend()
plt.yscale('log')
plt.ylabel('Loss')
plt.title("Train data = %d" %(len(train_mnist)))
plt.savefig('Output_loss.png')
plt.show()



# Test the model
from sklearn.metrics import roc_auc_score, auc, average_precision_score, \
    mean_squared_error, label_ranking_average_precision_score, accuracy_score
from sklearn.preprocessing import binarize

model.eval()
loss_history_valid = []
with torch.no_grad():
    correct = 0
    total = 0
    numBatches = 0

    score_lwlrap = 0
    score_roc_auc = 0
    score_pr_auc = 0
    score_mse = 0

    allPreds = np.empty((0, len(train_mnist.classes)), float)
    allLabels = np.empty((0, len(train_mnist.classes)), int)

    for images, labels in test_data_loader:
        images = images.to(device)
        labels = labels.cpu().numpy()
        labels = indices_to_one_hot(labels, len(train_mnist.classes))
        outputs = model.forward_with_evaluation(images).cpu().numpy()

        if not check_outputs(outputs, False):
            warnings.warn("Warning, the ouputs appear to have wrong values!!!!!")

        allPreds = np.append(allPreds, outputs, axis=0)
        allLabels = np.append(allLabels, labels, axis=0)

        total += labels.shape[0]
        numBatches += 1

    try:
        score_accuracy = accuracy_score(allLabels, indices_to_one_hot(allPreds.argmax(axis=1), len(train_mnist.classes)))
        score_lwlrap = label_ranking_average_precision_score(allLabels, allPreds)
        score_mse = math.sqrt(mean_squared_error(allLabels, allPreds))
        score_pr_auc = average_precision_score(allLabels, allPreds)
        score_roc_auc = roc_auc_score(allLabels, allPreds)
    except ValueError as e:
        print("Soemthing wrong with evaluation")
        print(e)

    print("Accuracy =  %f" % (score_accuracy))
    print("Label ranking average precision for train =  %f" % (score_lwlrap))
    print("AUC_ROC score for train = %f" % (score_roc_auc))
    print("AUC_PR score for train = %f" % (score_pr_auc))
    print("MSE score for train = %f" % (score_mse ))
