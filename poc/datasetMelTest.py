#!/usr/bin/env python3'
from cnn import ConvBlock
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import os
import librosa
import librosa.display
from PIL import Image
import pandas as pd
import numpy as np
#import sounddevice as sd
from easydict import EasyDict as edict
import torch
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import warnings
from tqdm import tqdm
import time

## based on
## https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789534092/1/ch01lvl1sec13/loading-data
##


class MelDataset(Dataset):

    def get_Mels(self):
        '''
        returns a mel fitler bank, used to compute the mel spectrograms
        '''

        if self.__melBank__ is None:
            self.__melBank__ = librosa.filters.mel(sr=self.config.resampling_rate,
                                                   n_fft=self.config.n_fft,
                                                   n_mels=self.config.n_mels,
                                                   fmin=self.config.fmin,
                                                   fmax=self.config.fmax,
                                                   htk=False,
                                                   norm=1)

        return self.__melBank__

    def audio_to_melspectrogram(self, audio):
        spec = librosa.core.stft(audio,
                                 hop_length=self.config.hop_length,
                                 win_length=self.config.hop_length,
                                 n_fft=self.config.n_fft,
                                 center=False
                                 )

        mel = self.get_Mels()

        # Manually compute the mel spectrogram,
        melSpec = librosa.power_to_db(mel.dot(np.abs(spec) ** 2), ref=np.max)
        melSpec = melSpec.astype(np.float32)

        return melSpec

    def audio_to_stft(self, audio, config):
        spectrogram = librosa.core.stft(audio,
                                        hop_length=config.hop_length,
                                        win_length=config.hop_length,
                                        n_fft=config.n_fft,
                                        center=False
                                        )

        # spectrgoram returns complex values (magnitude + phase)
        # abs to get ampltiude
        # then we go to power and show in db
        # where the ref is the maximum value in each file
        spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
        spectrogram = spectrogram.astype(np.float32)

        return spectrogram

    def __init__(self, datapath, labelsFile, targetLabels, config, transforms=None):
        self.dataPath = datapath
        self.targetLabels = targetLabels
        self.transform = transforms
        self.config = config
        self.__melBank__ = None

        tmp = pd.read_csv(os.path.join(self.dataPath, labelsFile))
        self.fnames = tmp['fname'].tolist()
        self.labelsString = tmp['labels'].tolist()

        # check that all images files exist
        for i in range(len(self.labelsString)):
            try:
                assert os.path.isfile(datapath + '/' + self.fnames[i])
                assert len(self.fnames) == len(self.labelsString)
            except AssertionError as error:
                print('Error %d       %s ' % (i, self.fnames[i]))
                raise

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        '''

        :param idx:
        :return: Each item has:
        an np.array of the audio file, with shaoe (mel_bins, frames)
        a one hot encoded vector for the labels (based on  the tarfgetLabels provided), with shape (labels)
        the string of the labels for this item
        the sample rate
        the diration in seconds
        '''
        audioName = self.fnames[idx]
        labelString = self.labelsString[idx]
        labels = self.getLabelVector(labelString, self.targetLabels)
        labels = labels.astype(np.float32)
        audioPath = os.path.join(self.dataPath, audioName)

        audio, sr = librosa.load(path=audioPath,
                                 mono=True)

        duration = len(audio) / sr  # Keep duration in seconds
        duration = float(duration)

        # Resampling
        if sr != self.config.resampling_rate:
            audio = librosa.resample(audio, sr, self.config.resampling_rate)

        # Normalize
        audio = librosa.util.normalize(audio)

        if config.useMel:
            spec = self.audio_to_melspectrogram(audio)
        else:
            spec = self.audio_to_stft(audio, config)

        spec = Image.fromarray(spec)

        # Transforms the image if required
        if self.transform:
            spec = self.transform(spec)

        sample = {'spectrogram': spec,
                  'labels': labels,
                  'labelString': labelString,
                  'sr': sr,
                  'duration': duration,
                  'fname': audioName}
        #return ((spec, labels, labelString, sr, duration))
        return sample

    def getLabelVector(self, labels, targetLabels):
        '''
        Inputs:
        labels = string with labels of sound file e.g. "door,clap,stuff_happening"
        thisTargetLabels = string array of ALL availabel target labels

        Returns:
        target_arr = one hot encoding the target labels for this sound file
        '''
        lbs = labels.split(",")
        target_arr = np.zeros(len(targetLabels))
        for lb in lbs:
            if (lb in targetLabels):
                i = targetLabels.index(lb)
                target_arr[i] = 1
                break
        return target_arr


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """
    def to_tensor(self, spec):

        if isinstance(spec, np.ndarray):
            # handle numpy array
            if spec.ndim == 2:
                spec = spec[:, :, None]

            img = torch.from_numpy(spec.transpose((2, 0, 1)))
            # backward compatibility
            if isinstance(img, torch.ByteTensor):
                return img.float().div(255)
            else:
                return img

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return self.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


## ########################################################################################################
## Main code
## ########################################################################################################


#Running from local laptop
csvFile = '/Volumes/scratch/work/falconr1/audioTaggingPKG/data/train_curated.csv'
datapath = '/Volumes/scratch/work/falconr1/audioTaggingPKG/data/train_curated'

#Running from cs-181
csvFile = '/m/cs/work/falconr1/audioTaggingPKG/data/train_curated.csv'
datapath = '/m/cs//work/falconr1/audioTaggingPKG/data/train_curated'

targetLabels = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark',
                     'Bass_drum', 'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell',
                     'Burping_and_eructation', 'Bus', 'Buzz', 'Car_passing_by', 'Cheering', 'Chewing_and_mastication',
                     'Child_speech_and_kid_speaking', 'Chink_and_clink', 'Chirp_and_tweet', 'Church_bell', 'Clapping',
                     'Computer_keyboard', 'Crackle', 'Cricket', 'Crowd', 'Cupboard_open_or_close',
                     'Cutlery_and_silverware', 'Dishes_and_pots_and_pans', 'Drawer_open_or_close', 'Drip',
                     'Electric_guitar', 'Fart', 'Female_singing', 'Female_speech_and_woman_speaking',
                     'Fill_(with_liquid)', 'Finger_snapping', 'Frying_(food)', 'Gasp', 'Glockenspiel', 'Gong',
                     'Gurgling', 'Harmonica', 'Hi-hat', 'Hiss', 'Keys_jangling', 'Knock', 'Male_singing',
                     'Male_speech_and_man_speaking', 'Marimba_and_xylophone', 'Mechanical_fan', 'Meow',
                     'Microwave_oven', 'Motorcycle', 'Printer', 'Purr', 'Race_car_and_auto_racing', 'Raindrop', 'Run',
                     'Scissors', 'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard', 'Slam',
                     'Sneeze', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush',
                     'Traffic_noise_and_roadway_noise', 'Trickle_and_dribble', 'Walk_and_footsteps',
                     'Water_tap_and_faucet', 'Waves_and_surf', 'Whispering', 'Writing', 'Yell', 'Zipper_(clothing)']

def viewSpec(spec, config):
    if isinstance(spec, torch.Tensor):
        spec = spec.numpy().squeeze() #should be a (mel_bins, frames) after the squeeze

    plt.figure()
    if config.useMel:
        librosa.display.specshow(spec, y_axis='mel', x_axis='time', sr=config.resampling_rate,
                                 hop_length=config.hop_length, fmax=config.fmax, fmin=config.fmin)
        plt.title('NormalizedMel spectrogram')
    else:
        librosa.display.specshow(spec, y_axis='log', x_axis='time', sr=config.resampling_rate,
                                 hop_length=config.hop_length)
        plt.title('Normalized Linear-frequency power STFT')
    plt.colorbar(format='%+2.0f dB')
    #plt.savefig('Tester_spectrgoram.png')
    plt.show()


# Helper function to show a batch
def viewSpecBatch(thisBatch, config, dataPath, gridSize=4):
    """Provides 3 visualizations of the batch:
            Spectrograms
            Wave files
            Time domeain plots
            Labels
    """

    specs, labels, fnames = thisBatch['spectrogram'], thisBatch['labels'], thisBatch['fname']
    batchSize = specs.shape[0]
    idx = np.random.choice(range(batchSize), gridSize, replace=False)

    imageSize = specs.size(2)
    assert math.sqrt(gridSize).is_integer(), "gridSize should be a squared number"
    plt.figure()
    for j, i in enumerate(idx):
        plt.subplot(math.sqrt(gridSize), math.sqrt(gridSize), j + 1)
        tmpSpec = specs[i, :, :, :].numpy().squeeze()
        if config.useMel:
            librosa.display.specshow(tmpSpec, y_axis='mel', x_axis='time', sr=config.resampling_rate,
                                     hop_length=config.hop_length, fmax=config.fmax, fmin=config.fmin)
            plt.title('NormalizedMel spectrogram')
        else:
            librosa.display.specshow(tmpSpec, y_axis='linear', x_axis='time', sr=config.resampling_rate,
                                     hop_length=config.hop_length)
            plt.title('Normalized Linear-frequency power STFT')
        plt.colorbar(format='%+2.0f dB')

    plt.savefig('Output_spectrogram.png')
    plt.show()

    ## Lets look at one random waveforms
    # note, we can listen to them too
    plt.figure()
    for j, i in enumerate(idx):
        plt.subplot(math.sqrt(gridSize), math.sqrt(gridSize), j + 1)
        fname = fnames[i]

        warnings.warn("WARNING  -- This code should be in a function or somewhere else")
        audio, sr = librosa.load(path=os.path.join(dataPath, fname),
                                 mono=True)

        # Resampling
        if sr != config.resampling_rate:
            audio = librosa.resample(audio, sr, config.resampling_rate)

        # Normalize
        audio = librosa.util.normalize(audio)

        librosa.display.waveplot(audio, sr=config.resampling_rate)
        plt.title('%d %s' % (j, fname.split('.')[0].split('/')[-1]))
        # librosa.output.write_wav('help_%d' %(i) + fname, wave, conf.resampling_rate)
        librosa.output.write_wav('Output_%d' % (j) + '.wav', audio, config.resampling_rate)

    plt.savefig('Output_waves.png')
    plt.show()

# Helper function to show a batch
def viewPredictions(thisBatch, config, gridSize=4):
    """Visualizes the predictions and groundtruth of some files
    """
    specs, labels, fnames = thisBatch['spectrogram'], thisBatch['labels'], thisBatch['fname']
    batchSize = specs.shape[0]
    idx = np.random.choice(range(batchSize), gridSize, replace=False)

    raise NotImplementedError
    imageSize = specs.size(2)
    assert math.sqrt(gridSize).is_integer(), "gridSize should be a squared number"
    plt.figure()
    for j, i in enumerate(idx):
        plt.subplot(math.sqrt(gridSize), math.sqrt(gridSize), j + 1)
        tmpSpec = specs[i, :, :, :].numpy().squeeze()
        if config.useMel:
            librosa.display.specshow(tmpSpec, y_axis='mel', x_axis='time', sr=config.resampling_rate,
                                     hop_length=config.hop_length, fmax=config.fmax, fmin=config.fmin)
            plt.title('NormalizedMel spectrogram')
        else:
            librosa.display.specshow(tmpSpec, y_axis='linear', x_axis='time', sr=config.resampling_rate,
                                     hop_length=config.hop_length)
            plt.title('Normalized Linear-frequency power STFT')
        plt.colorbar(format='%+2.0f dB')

    #Ground truth
    plt.subplot(4, 4, i+1)
    tmp = y_test[tmpIdx, :].reshape(10, 8)
    plt.imshow(tmp, vmin=0, vmax=1)

    # predictions
    plt.subplot(4, 4, i+2)
    tmp = preds_test[tmpIdx, :].reshape(10, 8)
    plt.imshow(tmp, vmin=0, vmax=1)



    plt.savefig('Output_waves.png')
    plt.show()




config = edict()
config.resampling_rate = 22050
config.hop_length_ms = 10
config.hop_length = math.ceil(config.hop_length_ms / 1000 * config.resampling_rate)
config.fmin = 20  # min freq
config.fmax = config.resampling_rate // 2  # max freq for the mels
config.n_mels = 128
config.n_fft = 2048
config.useMel = True

myDataset = MelDataset(datapath, csvFile, targetLabels, config,
                       transforms=ToTensor())

myDataset = MelDataset(datapath, csvFile, targetLabels, config,
                       transforms=transforms.Compose([
                           transforms.RandomCrop((config.n_mels, 256), pad_if_needed=True, padding_mode='reflect'),
                           transforms.ToTensor()
                       ]))
print(myDataset)
print(len(myDataset))

## Get ramdpm element
idx = np.random.randint(0, len(myDataset))
dataFile = myDataset[0]
# tensors are (channels, mel_bins, frames)
viewSpec(dataFile['spectrogram'], config)
print(dataFile['spectrogram'].shape)


## Get a batch of elements
subset_indices = np.random.choice(range(len(myDataset)), 500, replace=False)
trainloader = torch.utils.data.DataLoader(myDataset,
                                          batch_size=38,
                                          #shuffle=True,
                                          num_workers=40,
                                          sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))
dataIter = iter(trainloader)
batch = dataIter.next()  # batch is a dictionary of tensors
print(batch['labels'][0, 0:])  # print labels mfor the first sample in the batch
viewSpec(batch['spectrogram'][0, :, :, :], config)   # view spec for the first sample in the batch
viewSpecBatch(batch, config, myDataset.dataPath, gridSize=4)






## ==============================================================
## NN model

from cnn import FCN
from cnn import ConvBlock
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input shape (batch, 1, 128, 1366)
#model = FCN(ConvBlock, output_shape=80,
#            max_pool=[(2,4), (2,4), (2,4), (3,5), (4,4)]).to(device)
#from torchsummary import summary
#summary(model, input_size=(1, 128, 1366))

# input shape (batch, 1, 128, 256)
model = FCN(ConvBlock, output_shape=80,
            max_pool=[(2,2), (2,2), (2,4), (3,3), (4,4)],
            filters_num=[64, 128, 256, 512, 1024]).to(device)

from torchsummary import summary
summary(model, input_size=(1, 128, 256))


learning_rate = 0.001
num_epochs = 50

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Train the model
t_sum = 0
t_epoch = 0
last_time = time.time()
last_epoch = time.time()

total_step = len(trainloader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    pbar = tqdm(enumerate(trainloader))
    for i, sample in pbar:
        t = time.time()
        t_diff = t - last_time
        t_sum += t_diff
        last_time = t

        images = sample['spectrogram'].to(device)
        labels = sample['labels'].to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("Epoch [{}/{}], Step [{}/{}] T_diff {} , T_total {} , Loss: {:.4f}"
                             .format(epoch + 1, num_epochs, i + 1, total_step, t_diff, t_sum, loss.item()))

        #if (i + 1) % 1 == 0:
        #    print("Epoch [{}/{}], Step [{}/{}] T_diff {} , T_total {} , Loss: {:.4f}"
        #          .format(epoch + 1, num_epochs, i + 1, total_step, t_diff, t_sum, loss.item()))

    t = time.time()
    t_epoch = t - last_epoch
    last_epoch = t
    print("Epoch [{}/{}] , time {}" .format(epoch + 1, num_epochs, t_epoch))

    # Decay learning rate
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
from sklearn.metrics import roc_auc_score, auc, average_precision_score, \
    mean_squared_error, label_ranking_average_precision_score
from sklearn.preprocessing import binarize

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    numBatches = 0

    score_lwlrap = 0
    score_roc_auc = 0
    score_pr_auc = 0
    score_mse = 0

    allPreds = np.empty((0, 80), float)
    allLabels = np.empty((0, 80), int)

    for batch in trainloader:
        images = batch['spectrogram'].to(device)
        labels = batch['labels'].cpu().numpy()
        outputs = model(images).cpu().numpy()

        allPreds = np.append(allPreds, outputs, axis=0)
        allLabels = np.append(allLabels, labels, axis=0)

        total += labels.shape[0]
        numBatches += 1

    try:
        score_lwlrap = label_ranking_average_precision_score(binarize(allLabels, threshold=0.5), allPreds)
        score_roc_auc = roc_auc_score(binarize(allLabels, threshold=0.5), allPreds)
        score_pr_auc = average_precision_score(binarize(allLabels, threshold=0.5), allPreds)
        score_mse = math.sqrt(average_precision_score(allLabels, allPreds))
    except ValueError as e:
        print("Soemthing wrong with evaluation")
        print(e)

    print("Label ranking average precision for train =  %f" % (score_lwlrap / numBatches))
    print("AUC_ROC score for train = %f" % (score_roc_auc / numBatches))
    print("AUC_PR score for train = %f" % (score_pr_auc / numBatches))
    print("MSE score for train = %f" % (score_mse / numBatches))



## ------------------------------------------------------------------------
# Lets look at some results that were wrong

batch = dataIter.next()  # batch is a dictionary of tensors
print(batch['labels'][0, 0:])  # print labels mfor the first sample in the batch
viewSpec(batch['spectrogram'][0, :, :, :], config)   # view spec for the first sample in the batch
viewSpecBatch(batch, config, myDataset.dataPath, gridSize=4)



##look at outoputs

##plot loss

