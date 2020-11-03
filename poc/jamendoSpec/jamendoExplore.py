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
import matplotlib.pyplot as plt
from PIL import Image
import warnings
from tqdm import tqdm
import time

## based on
## https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789534092/1/ch01lvl1sec13/loading-data
##


class MelDataset(Dataset):
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

from scripts import commons
root_path = '/Volumes/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master/data/splits/split-0'
input_file = os.path.join(root_path, 'autotagging_top50tags-train.tsv')

tracks, tags, extra = commons.read_file(input_file)





