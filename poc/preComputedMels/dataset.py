#!/usr/bin/env python3'
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import librosa
from PIL import Image
import pandas as pd
import numpy as np
from easydict import EasyDict as edict
import torch
import math
from PIL import Image
import warnings
from tqdm import tqdm
import time
import pickle
from ATPP import Util
import h5py

class PreComputedMelDataset_MultiFiles(Dataset):
    def __init__(self, datapath_meta, datapath_mels, datapath_labels, targetLabels, config, train=True, transforms=None):
        self.datapath_meta = datapath_meta
        self.datapath_mels = datapath_mels
        self.datapath_labels = datapath_labels
        self.targetLabels = targetLabels
        self.transform = transforms
        self.config = config
        self.isTrain = train

        if not self.isTrain:
            fn = '/m/cs/work/falconr1/audioTagging2019/work/data_train_bkp.hdf5'
        else:
            fn = '/m/cs/work/falconr1/audioTagging2019/work/data_train.hdf5'

        # if not self.isTrain:
        #     fn = '/Volumes/scratch/work/falconr1/audioTagging2019/work/data_train_bkp.hdf5'
        # else:
        #     fn = '/Volumes/scratch/work/falconr1/audioTagging2019/work/data_train.hdf5'

        self.f = h5py.File(fn, 'r', swmr=True)

        self.group_specs = self.f['specs']  # Get the group
        self.group_labels = self.f['labels']  # Get the group
        self.group_fnames = self.f['fnames']  # Get the group

        with open(os.path.join(self.datapath_meta, 'fnames.pkl'), 'rb') as pickle_file:
            self.raw_fnames = pickle.load(pickle_file)
        with open(os.path.join(self.datapath_meta, 'fnames_y.pkl'), 'rb') as pickle_file:
            self.raw_fnames_y = pickle.load(pickle_file)
        with open(os.path.join(self.datapath_meta, 'fnames_ids.pkl'), 'rb') as pickle_file:
            self.raw_fnames_ids = pickle.load(pickle_file)
        with open(os.path.join(self.datapath_meta, 'fnames_short.pkl'), 'rb') as pickle_file:
            self.raw_fnames_short = pickle.load(pickle_file)
        with open(os.path.join(self.datapath_meta, 'train_ids.pkl'), 'rb') as pickle_file:
            self.train_ids = pickle.load(pickle_file)
        with open(os.path.join(self.datapath_meta, 'valid_ids.pkl'), 'rb') as pickle_file:
            self.valid_ids = pickle.load(pickle_file)

        print("Loaded files ")
        print("fnames with shape %d" % (len(self.raw_fnames)))
        print("fnames_y with shape %d" % (len(self.raw_fnames_y)))
        print("fnames_ids with shape %d" % (len(self.raw_fnames_ids)))
        print("fnames_short with shape %d" % (len(self.raw_fnames_ids)))
        print("train_ids with shape %d" % (len(self.train_ids)))
        print("valid_ids with shape %d" % (len(self.valid_ids)))

        if self.isTrain:
            tmpSet = self.train_ids
        else:
            tmpSet = self.valid_ids

        self.fnames = []
        for row in tqdm(tmpSet):
            tmp = None
            tmp = self.raw_fnames[row]
            if tmp is None or tmp == '':
                pass
            self.fnames.append(tmp)

        ## Are ids unique?
        a = tmpSet
        b = set(a)
        print(len(a))
        print(len(b))

        c = set(self.fnames)
        print(len(c))

        print('Hey bro')
        print(len(set(self.fnames)))

        #Intersection in train and vlid
        print(len(set(self.train_ids) & set(self.valid_ids)))

        # check that all images files exist
        print("Cheking Files exist")
        if False:
            for id in tqdm(tmpSet):
                item = self.raw_fnames[self.raw_fnames_ids[id]]

                try:
                    f_spec, f_label = Util.getNpyFilename(item, self.isTrain)

                    tmpPath = os.path.join(self.datapath_meta, self.datapath_mels, f_spec)
                    assert os.path.isfile(tmpPath)

                    tmpPath = os.path.join(self.datapath_meta, self.datapath_labels, f_label)
                    assert os.path.isfile(tmpPath)
                except AssertionError as error:
                    print('Error       %s ' % (item))
                    raise

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        '''

        :param idx:
        :return: Each item has:
        np arrary of spectrgoram
        np array of one hot code labels
        fname -  full file path for the audio file

        an np.array of the audio file, with shaoe (mel_bins, frames)
        a one hot encoded vector for the labels (based on  the tarfgetLabels provided), with shape (labels)
        the string of the labels for this item
        '''

        audioName = self.fnames[idx]  # this is probably wrong, it does not match with fname down below


        #all_keys = [key for key in self.f.keys()]  # Keys for groups, e.g. 'specs'
        # all_keys = list(self.f.keys())

        #group_keys = [key for key in group_specs.keys()]  # These are keys to entries in the group

        #spec = group_specs[group_keys[0]].value  # Extract data

        # Lets look at the keys that we have in the group:
        tmp = np.array(list(map(int, self.group_fnames.keys())))

        #print('HEY idx %d' % (len(set(tmp) & set([idx]))))
        try:
            spec = self.group_specs[str(idx)][()]
            labels = self.group_labels[str(idx)][()].astype(np.float32)
            fname = self.group_fnames[str(idx)][()]

        except Exception as e:
            self.f.close()
            print("error")
            print(e)

        # f_spec, f_label = Util.getNpyFilename(audioName, self.isTrain)
        # tmpPath = os.path.join(self.datapath_meta, self.datapath_mels, f_spec)
        # spec = np.load(tmpPath)
        #
        # tmpPath = os.path.join(self.datapath_meta, self.datapath_labels, f_label)
        # labels = np.load(tmpPath)
        #
        # spec = list(self.f['specs'])
        #
        #
        # #spec = Image.fromarray(spec)
        # labels = labels.astype(np.float32)jjj

        # Transforms the image if required now
        if self.transform:
            spec = self.transform(spec)

        sample = {'spectrogram': spec,
                  'labels': labels,
                  'fname': fname}
        return sample


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


class PreComputedMelDataset_Vanilla(Dataset):
    '''
    Reads spectrograms saved as individual npy files
    '''
    def __init__(self, datapath_meta, datapath_mels, datapath_labels, targetLabels, config, train=True, transforms=None):
        self.datapath_meta = datapath_meta
        self.datapath_mels = datapath_mels
        self.datapath_labels = datapath_labels
        self.targetLabels = targetLabels
        self.transform = transforms
        self.isTrain = train

        with open(os.path.join(self.datapath_meta, 'fnames.pkl'), 'rb') as pickle_file:
            self.raw_fnames = pickle.load(pickle_file)
        with open(os.path.join(self.datapath_meta, 'fnames_y.pkl'), 'rb') as pickle_file:
            self.raw_fnames_y = pickle.load(pickle_file)
        with open(os.path.join(self.datapath_meta, 'fnames_ids.pkl'), 'rb') as pickle_file:
            self.raw_fnames_ids = pickle.load(pickle_file)
        with open(os.path.join(self.datapath_meta, 'fnames_short.pkl'), 'rb') as pickle_file:
            self.raw_fnames_short = pickle.load(pickle_file)
        with open(os.path.join(self.datapath_meta, 'train_ids.pkl'), 'rb') as pickle_file:
            self.train_ids = pickle.load(pickle_file)
        with open(os.path.join(self.datapath_meta, 'valid_ids.pkl'), 'rb') as pickle_file:
            self.valid_ids = pickle.load(pickle_file)

        print("Loaded files ")
        print("fnames with shape %d" % (len(self.raw_fnames)))
        print("fnames_y with shape %d" % (len(self.raw_fnames_y)))
        print("fnames_ids with shape %d" % (len(self.raw_fnames_ids)))
        print("fnames_short with shape %d" % (len(self.raw_fnames_ids)))
        print("train_ids with shape %d" % (len(self.train_ids)))
        print("valid_ids with shape %d" % (len(self.valid_ids)))

        if self.isTrain:
            tmpSet = self.train_ids
        else:
            tmpSet = self.valid_ids

        self.fnames = []
        for row in tqdm(tmpSet):
            tmp = None
            tmp = self.raw_fnames[row]
            if tmp is None or tmp == '':
                pass
            self.fnames.append(tmp)

        # Checking if files exist
        if False:
            for id in tqdm(tmpSet):
                item = self.raw_fnames[self.raw_fnames_ids[id]]

                try:
                    f_spec, f_label = Util.getNpyFilename(item, self.isTrain)

                    tmpPath = os.path.join(self.datapath_meta, self.datapath_mels, f_spec)
                    assert os.path.isfile(tmpPath)

                    tmpPath = os.path.join(self.datapath_meta, self.datapath_labels, f_label)
                    assert os.path.isfile(tmpPath)
                except AssertionError as error:
                    print('Error       %s ' % (item))
                    raise

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        '''

        :param idx:
        :return: Each item has:
        np arrary of spectrgoram
        np array of one hot code labels
        fname -  full file path for the audio file

        an np.array of the audio file, with shaoe (mel_bins, frames)
        a one hot encoded vector for the labels (based on  the tarfgetLabels provided), with shape (labels)
        the string of the labels for this item
        '''

        fname = self.fnames[idx]  # this is probably wrong, it does not match with fname down below

        f_spec, f_label = Util.getNpyFilename(fname, self.isTrain)
        tmpPath = os.path.join(self.datapath_meta, self.datapath_mels, f_spec)
        spec = np.load(tmpPath)

        tmpPath = os.path.join(self.datapath_meta, self.datapath_labels, f_label)
        labels = np.load(tmpPath)

        #spec = Image.fromarray(spec)
        labels = labels.astype(np.float32)

        # Transforms the image if required now
        if self.transform:
            spec = self.transform(spec)

        sample = {'spectrogram': spec,
                  'labels': labels,
                  'fname': fname}
        return sample