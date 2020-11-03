#!/usr/bin/env python3'
import torch
torch.multiprocessing.set_start_method("fork", force=True)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from constants import  Constants
from ATPP import AudioTaggingPreProcessing, Util
from dataset import PreComputedMelDataset_MultiFiles, ToTensor, PreComputedMelDataset_Vanilla
from PIL import Image
import pandas as pd
import numpy as np
#import sounddevice as sd
from easydict import EasyDict as edict
import torch
import math
import os
import matplotlib
#matplotlib.use('agg')
#####matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.interactive(False)
import seaborn as sns
sns.set()
from PIL import Image
import warnings

import time

## based on
## https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789534092/1/ch01lvl1sec13/loading-data
##


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


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def computeMeanAndSTD(dataset, n_mels):
    '''Computes the mean and std of the dataset.

    Based on:
    https://forums.fast.ai/t/image-normalization-in-pytorch/7534/12
    '''
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((n_mels, 256), pad_if_needed=True, padding_mode='reflect'),
        transforms.ToTensor()
    ])

    dataset.transform = transform

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, sample in enumerate(dataloader, 0):
        # shape (batch_size, 3, height, width)
        data = sample['spectrogram'].cpu()
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

def main():
    from tqdm import tqdm
    constants = Constants(3)

    config = edict()
    config.resampling_rate = 22050
    config.hop_length_ms = 10
    config.hop_length = math.ceil(config.hop_length_ms / 1000 * config.resampling_rate)
    config.fmin = 20  # min freq
    config.fmax = config.resampling_rate // 2  # max freq for the mels
    config.n_mels = 128
    config.n_fft = 2048
    config.useMel = True

    train_transforms = transforms.Compose([
                                   transforms.ToPILImage(),
                                   transforms.RandomCrop((config.n_mels, 256), pad_if_needed=True, padding_mode='reflect'),
                                   transforms.ToTensor(),
                                   #transforms.Normalize([0.5], [0.5])
                                   transforms.Normalize([-48.78], [19.78])
                                  ])

    valid_transforms = transforms.Compose([
                                   transforms.ToPILImage(),
                                   transforms.RandomCrop((config.n_mels, 256), pad_if_needed=True, padding_mode='constant'),
                                   transforms.ToTensor(),
                                   transforms.Normalize([-50.87], [20.30])
                                  ])

    train_dataset = PreComputedMelDataset_MultiFiles(constants.WORK,
                                                     constants.WORK_SPECS,
                                                     constants.WORK_LABELS,
                                                     constants.target_labels,
                                                     config,
                                                     transforms=train_transforms
                                                     )

    valid_dataset = PreComputedMelDataset_MultiFiles(constants.WORK,
                                                     constants.WORK_SPECS,
                                                     constants.WORK_LABELS,
                                                     constants.target_labels,
                                                     config,
                                                     transforms=valid_transforms,
                                                     train=False
                                                     )

    train_dataset = PreComputedMelDataset_Vanilla(constants.WORK,
                                                  constants.WORK_SPECS,
                                                  constants.WORK_LABELS,
                                                  constants.target_labels,
                                                  config,
                                                  transforms=train_transforms
                                                  )

    valid_dataset = PreComputedMelDataset_Vanilla(constants.WORK,
                                                  constants.WORK_SPECS,
                                                  constants.WORK_LABELS,
                                                  constants.target_labels,
                                                  config,
                                                  transforms=valid_transforms,
                                                  train=False
                                                  )







    print(train_dataset)
    print(len(train_dataset))


    ## ############################################################
    ## Compute means for dataset
    ## ############################################################

    #train_means, train_std0, train_std1 = computeMeanAndSTD(train_dataset, config.n_mels)
    #valid_means, valid_std0, valid_std1 = computeMeanAndSTD(valid_dataset, config.n_mels)

    #train_dataset.transform = train_transforms
    #valid_dataset.transform = valid_transforms

    #print(train_means)
    #print(train_std0)


    ## ############################################################
    ## Load data and look at a few examples
    ## ############################################################

    ## Get ramdpm element
    idx = np.random.randint(0, len(train_dataset))
    dataFile = train_dataset[0]
    # tensors are (channels, mel_bins, frames)
    AudioTaggingPreProcessing.show_spectrogram(dataFile['spectrogram'], config, saveFig=True, filepath=os.path.join(os.getcwd(), ''))
    print(dataFile['spectrogram'].shape)

    ## Get a batch of elements
    subset_indices_train = np.random.choice(range(len(train_dataset)), 6279, replace=False)
    subset_indices_valid = np.random.choice(range(len(valid_dataset)), 2691, replace=False)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=64,
                                              #shuffle=True,
                                              num_workers=1,
                                              pin_memory=False,
                                              sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices_train))

    validLoader = torch.utils.data.DataLoader(valid_dataset,
                                              batch_size=128,
                                              #shuffle=False,
                                              num_workers=1,
                                              pin_memory=False,
                                              sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices_valid))

    dataIter = iter(trainloader)
    batch = dataIter.next()  # batch is a dictionary of tensors
    print(batch['labels'][0, 0:])  # print labels mfor the first sample in the batch

    AudioTaggingPreProcessing.show_spectrogram(batch['spectrogram'][0, :, :, :], config, saveFig=True, filepath=os.path.join(os.getcwd(), ''))
    AudioTaggingPreProcessing.show_spectrogram_batch(batch, config, saveFig=True)

    ## ############################################################
    ## Model stuff
    ## ############################################################




    from cnn import FCN
    from cnn import ConvBlock
    import torch.nn as nn
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # input shape (batch, 1, 128, 1366) ss
    #model = FCN(ConvBlock, output_shape=80,
    #            max_pool=[(2,4), (2,4), (2,4), (3,5), (4,4)]).to(device)
    #from torchsummary import summary
    #summary(model, input_size=(1, 128, 1366))

    # input shape (batch, 1, 128, 256)
    model = FCN(ConvBlock, output_shape=80,
                max_pool=[(2,2), (2,2), (2,4), (3,3), (4,4)],
                filters_num=[64, 128, 256, 512, 1024]).to(device)


    summary(model, input_size=(1, 128, 256))


    learning_rate = 0.001
    num_epochs = 30
    print_every = 99 # 99 with 64 batch size, 25 wtih 256

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    criterion = model.loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # For updating learning rate
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Another hacky way to get the iterator for the validation set
    def loopy(dl):
        while True:
            for x in iter(dl): yield x


    # Train the model
    t_sum = 0
    t_epoch = 0
    last_time = time.time()
    last_epoch = time.time()
    all_time = []
    all_time_epoch = []

    total_step = len(trainloader)
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

    # Note: The idea is to generate an iterator for the validation dataloader, so that we can get a single batch and
    # compute the validation loss only for that batch, instead of for all the validation set
    # The problem is that when the iterator is exhausted, an StopIteration exception is raised.
    # So a hacky way to solve this is to catch the exception and regenerate the iterator.
    # More info:
    # https://github.com/pytorch/pytorch/issues/1917

    myIter = loopy(validLoader)

    # for i in range(5):
    #     tmp = next(myIter)
    #     img = tmp['spectrogram'].numpy()
    #     print("Iteration {:4d}    Mean = {:2.4f}".format(i, img.mean()))
    #
    #     plt.figure()
    #     plt.imshow(img[0, :, :, :].squeeze())
    #     plt.show()
    #
    # del myIter
    #
    # for h in range(2):
    #     print('==========================')
    #     counter = 0
    #     for j in validLoader:
    #         counter += 1
    #         img = j['spectrogram'].numpy()
    #         print("yolo {:2d}     Mean = {:2.4f}".format(counter, img.mean()))
    #
    # return

    for epoch in range(num_epochs):
        train_loss_epoch = 0.0
        valid_loss_epoch = 0.0
        train_loss_step = 0.0
        valid_loss_step = 0.0

        # Training loop
        pbar = tqdm(enumerate(trainloader))
        for i, sample in pbar:
        #for i, sample in enumerate(trainloader):

            t = time.time()
            t_diff = t - last_time
            t_sum += t_diff
            last_time = t
            all_time.append(t_diff)

            images = sample['spectrogram'].to(device)
            labels = sample['labels'].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
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
                    # Hacky way to get a single validation batch, see above for comments.
                    try:
                        sample = next(myIter)
                    except StopIteration:
                        myIter = loopy(validLoader)
                        sample = next(myIter)

                    images = sample['spectrogram'].to(device)
                    labels = sample['labels'].to(device)

                    outputs = model(images)
                    #loss = criterion(outputs, torch.zeros(len(labels), 80).to(device).scatter_(1, labels.unsqueeze(1), 1.))
                    loss = criterion(outputs, labels)

                    # train_loss_epoch += loss.item() * images.size(0) / len(train_dataset)  # avg loss of batch, times batch size
                    valid_loss_step = loss.item()
                    valid_loss_epoch += valid_loss_step

                    valid_steps += 1
                    # Append losses for plotting
                    valid_loss_history.append(valid_loss_step)

            pbar.set_description("Epoch [{}/{}], Step [{}/{}],   T_diff {} , T_total {} , train_Loss: {:.4f} , valid_Loss {:.4f}"
                                 .format(epoch + 1, num_epochs, i + 1, total_step,
                                         t_diff, t_sum, train_loss_step, valid_loss_step))

        t = time.time()
        t_epoch = t - last_epoch
        last_epoch = t
        all_time_epoch.append(t_epoch)
        print("--------- Epoch [{}/{}] Summary , time per epoch {} , train_loss {} , valid_loss {}"
              .format(epoch + 1, num_epochs, t_epoch,
                      train_loss_epoch / len(trainloader), valid_loss_epoch / max(valid_steps, 1)))
        #divide train loss epoch by how many batches
        #divide valid loss peoch by how many steps were actually computed

        # # Decay learning rate
        # if (epoch + 1) % 20 == 0:
        #     curr_lr /= 3
        #     update_lr(optimizer, curr_lr)



    ## Plot time for each step
    t = np.linspace(0, num_epochs, num=len(all_time), endpoint=False)

    plt.figure(figsize=(16,8))
    plt.plot(t, all_time, 'r-+', label='Time per step')
    plt.legend()
    plt.savefig('Output_time_per_step.png')
    plt.xlabel('Epochs')
    plt.ylabel('Time (s)')
    plt.title("Train data = %d" %(len(train_dataset)))
    plt.show()

    ## Plot time per epoch
    t = np.linspace(0, num_epochs, num=len(all_time_epoch), endpoint=False)

    plt.figure(figsize=(16,8))
    plt.plot(t, all_time_epoch, 'r-+', label='Time per step')
    plt.legend()
    plt.savefig('Output_time_per_epoch.png')
    plt.xlabel('Epochs')
    plt.ylabel('Time (s)')
    plt.title("Train data = %d" %(len(train_dataset)))
    plt.show()


    ## Plot loss
    t = np.linspace(0, num_epochs, num=train_steps, endpoint=False)
    t_valid = np.linspace(0, num_epochs, num=valid_steps, endpoint=False)

    plt.figure(figsize=(20,8))
    plt.plot(t, train_loss_history, 'r-+', label='Train')
    plt.plot(t_valid, valid_loss_history, 'b--o', label='Valid')
    plt.legend()
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.title("Train data = %d" %(len(train_dataset)))
    plt.savefig('Output_loss.png')
    plt.show()



    # Test the model
    from sklearn import metrics

    def get_auc(y_true, y_preds, labels_list):
        from sklearn import metrics
        score_accuracy = 0
        score_lwlrap = 0
        score_roc_auc = 0
        score_pr_auc = 0
        score_mse = 0
        score_roc_auc_all = np.zeros((len(labels_list), 1))
        score_pr_auc_all = np.zeros((len(labels_list), 1))
        try:
            # for accuracy, lets pretend this is a single label problem, so assign only one label to every prediction
            score_accuracy = metrics.accuracy_score(y_true, indices_to_one_hot(y_preds.argmax(axis=1), 80))
            score_lwlrap = metrics.label_ranking_average_precision_score(y_true, y_preds)
            score_mse = math.sqrt(metrics.mean_squared_error(y_true, y_preds))
            # Average precision is a single number used to approximate the integral of the PR curve
            score_pr_auc = metrics.average_precision_score(y_true, y_preds, average="macro")
            score_roc_auc = metrics.roc_auc_score(y_true, y_preds, average="macro")
        except ValueError as e:
            print("Soemthing wrong with evaluation")
            print(e)
        print("Accuracy =  %f" % (score_accuracy))
        print("Label ranking average precision =  %f" % (score_lwlrap))
        print("ROC_AUC score  = %f" % (score_roc_auc))
        print("PR_AUC score = %f" % (score_pr_auc))
        print("MSE score for train = %f" % (score_mse))
        # These are per tag
        try:
            score_roc_auc_all = metrics.roc_auc_score(y_true, y_preds, average=None)
            score_pr_auc_all = metrics.average_precision_score(y_true, y_preds, average=None)
        except ValueError as e:
            print("Soemthing wrong with evaluation")
            print(e)

        print("")
        print("Per tag, roc_auc, pr_auc")
        for i in range(len(labels_list)):
            print('%s \t\t\t\t\t\t %.4f \t%.4f' % (labels_list[i], score_roc_auc_all[i], score_pr_auc_all[i]))


        tmp = {
            'accuracy': score_accuracy,
            'lwlrap': score_lwlrap,
            'mse': score_mse,
            'roc_auc': score_roc_auc,
            'pr_auc': score_pr_auc,
            'roc_auc_all': score_roc_auc_all,
            'pr_auc_all': score_pr_auc_all
        }
        scores = edict(tmp)
        plt.figure(figsize=(20, 8))
        plt.bar(np.arange(len(labels_list)), scores.roc_auc_all[:, 0])
        plt.ylabel('ROC_AUC')
        plt.title("Train data = %d" % (len(train_dataset)))
        plt.savefig('Output_scores_per_label.png')
        plt.show()

        return scores

    del myIter  # Hacky way to reuse the dataloader for validaiton
    model.eval()
    with torch.no_grad():
        total = 0
        numBatches = 0


        allPreds = np.empty((0, 80), float)
        allLabels = np.empty((0, 80), int)

        for batch in validLoader:
            images = batch['spectrogram'].to(device)
            labels = batch['labels'].cpu().numpy()

            outputs = model.forward_with_evaluation(images).cpu().numpy()

            if not check_outputs(outputs, True):
                warnings.warn("Warning, the ouputs appear to have wrong values!!!!!")

            allPreds = np.append(allPreds, outputs, axis=0)
            allLabels = np.append(allLabels, labels, axis=0)

            total += labels.shape[0]
            numBatches += 1

        print("Evaluated on {} validation batches".format(numBatches))

        get_auc(allLabels, allPreds, constants.target_labels)




if __name__ == "__main__":
    main()