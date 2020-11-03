#!/usr/bin/env python3'
import os, math, warnings
import numpy as np
import pickle
import torch
import torch.utils.data.dataloader
from torch.utils.data import Dataset


class JamendoAudioDataset(Dataset):
    '''
    Creates a data set that reads individual numpy files, precomputed from audio files (mp3) of the Jamendo dataset.
    '''

    audio_path = 'data/processed/audio_npy'

    def __init__(self, root_directory='/m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master/',
                 subset='top50tags', split=0, mode='train', mono=True, trim_to_size=-1, fs=44100,
                 data_limit=100000,
                 transform=None,
                 target_transform=None):
        self.mode = mode
        self.root = root_directory
        self.subset = subset
        self.split = split
        self.mode = mode
        self.mono = mono
        self.trim_to_size = trim_to_size  # if -1, read full file, else, read only those samples (similar to random crop)
        self.fs = fs  # default fs for Jamendo is 44100
        self.data_limit=data_limit
        self.data_path = self._get_data_path()

        self.dictionary = self.get_dictionary(self.root, split, subset, mode)
        self.taglist = self.get_taglist(self.root, split, subset)

        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

    def _get_data_path(self):
        if self.fs == 44100:
            audio_path = 'data/processed/audio_npy'
        elif self.fs == 16000:
            audio_path = 'data/processed/audio_npy_16k'

        return os.path.join(self.root, self.audio_path)

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: The output tuple (audio, target, fname) where target
            is a one hot encoded tensor or the audio labels, fname is the fname of the file, e.g. '/14/1234.mp3'
        '''

        fname = self.dictionary[index]['path']
        fn = os.path.join(self.data_path, fname[:-3] + 'npy')
        audio = np.load(fn, mmap_mode='r')
        targets = self.dictionary[index]['tags']

        # Sanity check, the audio file should not be empty
        if audio.size < 2:
            warnings.warn('Hey, the data seems to be corrupt.')

        # Keep only relevant channels, shape should be [channels, time]
        channels = audio.shape[0] if not self.mono else 1

        # Crop if needed
        if self.trim_to_size > 0:
            output_size = self.trim_to_size
            input_size = audio.shape[-1]  # Assuming last axis is time

            if input_size < output_size:
                to_pad = math.floor((output_size - input_size) / 2)
                audio = np.pad(audio, [(0, 0), (to_pad, to_pad)], 'constant', constant_values=0)
            else:
                idx = np.random.randint(0, input_size - output_size)
                audio = audio[:, idx:idx + output_size]

        if self.mono:
            audio = np.mean(audio, 0, keepdims=True)

        audio = torch.from_numpy(audio)  # Cast memmap to Tensor. Maybe this is not the best way to do it.

        if self.transform:
            pass
        if self.target_transform:
            pass

        return audio, torch.from_numpy(targets.astype(np.float32)), fname

    def __len__(self):
        return len(self.dictionary)

    @property
    def num_classes(self):
        return len(self.taglist)

    @property
    def tags_list(self):
        return self.taglist

    def _check_exists(self):
        return os.path.exists(self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)

        return fmt_str

    def get_dictionary(self, root, split, subset, mode):
        fn = 'data/splits/split-%d/%s_%s_dict.pickle' % (split, subset, mode)
        fn = os.path.join(root, fn)
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)

        if self.data_limit < len(dictionary):
            while len(dictionary) > self.data_limit:
                dictionary.popitem()
        return dictionary

    def get_taglist(self, root, split, subset):
        if subset == 'top50tags':
            tmp = 'tag_list_50.npy'
        else:
            tmp = 'tag_list.npy'

        # fn_tags = 'data/splits/split-%d/%s' % (split, tmp)
        fn_tags = 'scripts/baseline/%s' % tmp
        fn_tags = os.path.join(root, fn_tags)
        tag_list = np.load(fn_tags)

        if subset == 'all':
            pass
        elif subset == 'genre':
            tag_list = tag_list[:87]
        elif subset == 'instrument':
            tag_list = tag_list[87:127]
        elif subset == 'moodtheme':
            tag_list = tag_list[127:]
        elif subset == 'top50tags':
            pass

        return tag_list


if __name__ == '__main__':
    '''
    Test the dataset.

    Try to test all combinations for prefecth_files, and read_raw_audio
    '''

    splits = [0,1,2,3,4]
    modes = ['train', 'valid', 'test']

    for opt1 in splits:
        for opt2 in modes:
            dataset = JamendoAudioDataset(root_directory='/m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master/',
                                          mode='train',
                                          subset='top50tags',
                                          split=0,
                                          mono=True,
                                          trim_to_size=30 * 44100,
                                          data_limit=100)

            audio, target, fname = dataset[0]

            print("")
            print(dataset)
            print(dataset.__repr__())
            print('Length of the dataset = {} '.format(len(dataset)))
            print('Shape of the first observation = {}'.format(audio.shape))
            print('Shape of the targets = {}'.format(target.shape))
            print('Name of the file = ')
            print(fname)



