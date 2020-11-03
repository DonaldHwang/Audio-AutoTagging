#!/usr/bin/env python3'
from torch.utils.data import Dataset
import os
import numpy as np
import torchaudio
import torch
from tqdm import tqdm, trange
import torch.utils.data.dataloader
import warnings
import math


class SpeechCommandsDataset(Dataset):
    """Google speech commands dataset. Only 'yes', 'no', 'up', 'down', 'left',
    'right', 'on', 'off', 'stop' and 'go' are treated as known classes.
    All other classes are used as 'unknown' samples.
    See for more information: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

    Code based on :
    https://github.com/tugstugi/pytorch-speech-commands/blob/master/datasets/speech_commands_dataset.py
    """

    audio = {}

    def __init__(self, root_directory='/m/cs/work/falconr1/datasets/speechCommands/',
                 raw_directory='raw',
                 processed_directory='processed',
                 transform=None,
                 classes=None,
                 normalize= True,
                 prefetch_files=True,
                 read_raw_audio=True,
                 trim_audio=-1,  # positive numbers to get only x samples
                 data_limit=20,  # limit how data samples to read
                 ):

        self.normalize = normalize
        self.trim_audio = trim_audio
        self.prefetch_files = prefetch_files
        self.read_raw_audio = read_raw_audio
        if self.read_raw_audio:
            self.data_directory = os.path.join(root_directory, raw_directory)
        else:
            self.data_directory = os.path.join(root_directory, processed_directory)

        if classes is None:
            classes = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go'.split(', ')
            classes = 'yes, no, up, down, left, right, on, off, stop, go, one, two, three, four, five, six, seven, eight, nine, zero'.split(', ')

        self.data_limit = data_limit / len(classes) if data_limit <= len(classes) else 100000
        self.data_limit = data_limit if data_limit > -1 else 100000

        all_classes = [d for d in os.listdir(self.data_directory) if os.path.isdir(os.path.join(self.data_directory, d)) and not d.startswith('_')]

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for c in all_classes:
            if c not in class_to_idx:
                class_to_idx[c] = 0

        data = []

        self.data = []
        self.targets = np.empty((0, len(classes)), dtype=np.long)
        self.fnames = []

        print("Loading data.")

        counter = 0
        #for i in trange(len(all_classes), desc='Directories'):  # Iterate directories (classes)
        for c in tqdm(all_classes):
            if counter >= self.data_limit:
                break
            if c not in classes: continue
            d = os.path.join(self.data_directory, c)
            target_id = class_to_idx[c]

            #dirs = os.listdir(d)
            #for j in trange(len(dirs), desc='Files in directory'):  # Iterate files in directories
            #    f = dirs[j]
            counter_sub = 0
            for f in (os.listdir(d)):
                counter_sub += 1
                if counter_sub > math.ceil(self.data_limit / len(classes)):
                    break
                counter += 1

                fname = os.path.join(c,f)
                tmp_path = os.path.join(d,f)
                target = np.zeros((1,len(classes)))
                target[0, target_id] = 1

                if self.prefetch_files:
                    audio = self.__read_audio(tmp_path)
                    data.append(audio)

                self.fnames.append(fname)
                self.targets = np.append(self.targets, target, axis=0)

        self.tags_list = classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    @property
    def num_classes(self):
        return len(self.tags_list)

    def __repr__(self):
        return 'SpeechCommands dataset with {} files and {} classes.'.format(self.__len__(), self.num_classes)

    def __getitem__(self, index):
        fname = self.fnames[index]
        target = self.targets[index]

        if self.prefetch_files:
            audio = self.data[index]
        else:
            audio = self.__read_audio(fname)

        if self.transform is not None:
            audio = self.transform(audio)

        return audio, \
               torch.from_numpy(target.astype(np.float32)), \
               fname

    def __read_audio(self, fname_with_path):
        if self.read_raw_audio:
            audio, fs = torchaudio.load(fname_with_path, normalization=self.normalize)

            # Sanity check, the audio file should not be empty
            if audio.shape[-1] < 2:
                warnings.warn('Hey, the data seems to be corrupt.')

            # Crop if needed
            if self.trim_audio > 0:
                output_size = self.trim_audio
                input_size = audio.shape[-1]  # Assuming last axis is time

                # Clip too short, add 0s
                if input_size < output_size:
                    to_pad = math.floor((output_size - input_size) / 2)
                    tmp = (output_size - input_size) / 2
                    if tmp % 2 == 0:
                        to_pad = int(tmp)
                    else:
                        to_pad = (math.floor(tmp), math.floor(tmp)+1)
                    padder = torch.nn.ConstantPad1d(to_pad, 0)
                    audio = padder(audio)
                else:
                    if input_size - output_size > 0:
                        idx = np.random.randint(0, input_size - output_size)
                    else:
                        idx = 0
                    audio = audio[:, idx:idx + output_size]
        else:
            audio = np.load(os.path.join(fname_with_path[-4] + '.npy'))  # no mmap, to avoid "too many open files error"#
            fs = 16000
            audio = torch.from_numpy(audio.astype(np.float))
            audio = audio / np.max(np.abs(audio))

        if audio.shape[-1] != self.trim_audio:  # quick hack to fix padding
            audio = audio[:, :-1]

        return audio


def _test():
    '''
    Test the dataset
    '''

    prefetch_options = [True]
    read_raw_options = [True]

    for opt1 in prefetch_options:
        for opt2 in read_raw_options:
            dataset = SpeechCommandsDataset(root_directory='/m/cs/work/falconr1/datasets/speechCommands',
                                 raw_directory='raw/',
                                 processed_directory='processed/',
                                 prefetch_files=opt1,
                                 read_raw_audio=opt2,
                                 data_limit=5)

            audio, target, fname = dataset[0]

            print("")
            print(dataset)
            print(dataset.__repr__())
            print('Length of the dataset = {} '.format(len(dataset)))
            print('Shape of the first observation = {}'.format(audio.shape))
            print('Shape of the targets = {}'.format(target.shape))
            print('Name of the file = ')
            print(fname)


if __name__ == '__main__':
    _test()
