#!/usr/bin/env python3'
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import h5py
import lmdb
import torchaudio
import torch
import msgpack
import msgpack_numpy as mg
from tqdm import tqdm
import audtorch
import audtorch.datasets.utils as ut
import torch.utils.data.dataloader
import warnings

def get_dictionary(root, split, subset, mode):
    fn = 'data/splits/split-%d/%s_%s_dict.pickle' % (split, subset, mode)
    fn = os.path.join(root, fn)
    with open(fn, 'rb') as pf:
        dictionary = pickle.load(pf)
    return dictionary


def get_taglist(root, split, subset):
    if subset == 'top50tags':
        tmp = 'tag_list_50.npy'
    else:
        tmp = 'tag_list.npy'

    #fn_tags = 'data/splits/split-%d/%s' % (split, tmp)
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


def _include_repr(name, obj):
    r"""Include __repr__ from other object as indented string.
    Args:
        name (str): Name of the object to be documented, e.g. "Transform".
        obj (object with `__repr__`): Object that provides `__repr__` output.
    Results:
        str: Format string of object to include into another `__repr__`.
    Example:
        >>> t = transforms.Pad(2)
        >>> datasets._include_repr('Transform', t)
        '    Transform: Pad(padding=2, value=0, axis=-1)\n'
    """
    part1 = '    {}: '.format(name)
    part2 = obj.__repr__().replace('\n', '\n' + ' ' * len(part1))
    return '{0}{1}\n'.format(part1, part2)


class Seq2Seq_short(audtorch.collate.Seq2Seq):
    '''Wrapper over audtoch.collate.Seq2Seq that returns only the data and labels. '''

    def __call__(self, batch):
        r"""Collate and pad sequences of mini-batch.
        The output tensor is augmented by the dimension of `batch_size`.
        Args:
            batch (list of tuples): contains all samples of a batch.
                Each sample is represented by a tuple (`features`, `targets`)
                which is returned by data set's __getitem__ method
        Returns:
            torch.tensors: `features`, `feature lengths`, `targets`
                and `target lengths` in data format according to
                :attr:`batch_first`.
        """

        data, data_length, labels, labels_length = super().__call__(batch)
        return data, labels


class JamendoAudioFolder_npy(Dataset):
    '''
    Creates a data set that reads individual numpy files, precomputed from audio files (mp3) of the Jamendo dataset.
    '''

    audio_path = 'data/processed/audio_npy'

    def __init__(self, root, subset, split, mode='train', mono=True, trim_to_size=-1, transform=None, target_transform=None, return_fname=False):
        self.mode = mode
        self.root = root
        self.subset = subset
        self.split = split
        self.mode = mode
        self.mono = mono
        self.data_path = os.path.join(self.root, self.audio_path)
        self.return_fname = return_fname
        self.trim_to_size = trim_to_size  # if -1, read full file, else, read only those samples (similar to random crop)

        self.dictionary = get_dictionary(self.root, split, subset, mode)
        self.taglist = get_taglist(self.root, split, subset)

        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

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
        tags = self.dictionary[index]['tags']

        #Sanity check, the audio file should not be empty
        if audio.size < 2:
            warnings.warn('Hey, the data seems to be corrupt.')

        # Keep only relevant channels, shape should be (channels, time)
        channels = audio.shape[0] if not self.mono else 1

        # Crop if needed
        if self.trim_to_size > 0:
            output_size = self.trim_to_size
            input_size = audio.shape[-1]  # Assuming last axis is time

            if input_size < output_size:  # TODO: fix padding
                import math
                to_pad = math.floor((output_size - input_size) / 2)
                audio = np.pad(audio, [(0, 0), (to_pad, to_pad)], 'constant', constant_values=0)
            else:
                idx = np.random.randint(0, input_size - output_size)
                #audio = audio[0:channels, idx:idx + output_size ]
                audio = audio[:, idx:idx + output_size]

        if self.mono:
            audio = np.mean(audio, 0, keepdims=True)

        raw_audio = torch.from_numpy(audio)  # Cast memmap to Tensor. Maybe this is not the best way to do it.

        if self.transform:
            audio = self.transform(raw_audio)
        if self.target_transform:
            tags = self.target_transform(tags)

        if self.return_fname:
            return audio, torch.from_numpy(tags.astype(np.float32)), fname, raw_audio
        else:
            return audio, torch.from_numpy(tags.astype(np.float32))

    def __len__(self):
        return len(self.dictionary)

    @property
    def num_classes(self):
        return len(self.taglist)

    def _check_exists(self):
        return os.path.exists(self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        # if self.sampling_rate == self.original_sampling_rate:
        #     fmt_str += '    Sampling Rate: {}Hz\n'.format(self.sampling_rate)
        # else:
        #     fmt_str += ('    Sampling Rate: {}Hz (original: {}Hz)\n'
        #                 .format(self.sampling_rate,
        #                         self.original_sampling_rate))

        if self.transform:
            fmt_str += _include_repr('Transform', self.transform)
        if self.target_transform:
            fmt_str += _include_repr('Target Transform', self.target_transform)
        return fmt_str


class JamendoAudioFolder_torchaudio(Dataset):
    '''
    Creates a data set that reads individual audio files (mp3) of the Jamendo dataset
    '''

    audio_path = 'data/raw_30s/'

    def __init__(self, root, subset, split, mode='train', mono=True, trim_to_size=-1,
                 transform=None, normalize=True, target_transform=None, return_fname=False):
        self.mode = mode
        self.root = root
        self.subset = subset
        self.split = split
        self.mode = mode
        self.mono = mono
        self.trim_to_size = trim_to_size
        self.data_path = os.path.join(self.root, self.audio_path)
        self.normalize = normalize
        self.return_fname = return_fname

        self.dictionary = get_dictionary(self.root, split, subset, mode)
        self.taglist = get_taglist(self.root, split, subset)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: The output tuple (audio, target, fname, raw_audio) where:
                audio - audio data as tensor
                target - one hot encoded tensor of the labels
                fname - fname of the file, e.g. '/14/1234.mp3'
                raw_audio - audio data as tensor before any transforms
        '''
        fname = self.dictionary[index]['path']
        fn = os.path.join(self.data_path, fname)

        if not self.normalize:
            audio, sample_rate = torchaudio.load(fn)  # this is a torch.tensor
        else:
            audio, sample_rate = torchaudio.load(fn, normalization=lambda x: torch.abs(x).max())
        tags = self.dictionary[index]['tags']

        if self.normalize:  # normalize again, the code above is not working
            audio = torch.div(audio, torch.abs(audio).max())

        # Keep only relevant channels, shape should be (channels, time)
        channels = audio.shape[0] if not self.mono else 1

        if self.mono:
            audio = torch.mean(audio, 0, keepdims=True)

        # Crop if needed
        if self.trim_to_size > 0:
            output_size = self.trim_to_size
            input_size = audio.shape[-1]  # Assuming last axis is time

            if input_size < output_size:  # TODO: add padding?
                audio = audio[0:channels, :]
            else:
                idx = np.random.randint(0, input_size - output_size)
                audio = audio[0:channels, idx:idx + output_size]

        raw_audio = audio

        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            tags = self.target_transform(tags)

        if self.return_fname:
            return audio, torch.from_numpy(tags), fname, raw_audio
        else:
            return audio, torch.from_numpy(tags)

    def __len__(self):
        return len(self.dictionary)

    @property
    def num_classes(self):
        return len(self.taglist)


class JamendoSpecFolder(Dataset):
    '''
    Creates a data set that reads individual precomputed spectrogram files (npy) fo the Jamendo dataset
    '''
    def __init__(self, root, subset, split, mode='train', spec_folder='data/raw_30s_specs/', transform=None, return_fname=False):
        self.mode = mode
        self.root = root
        self.spec_path = os.path.join(self.root, spec_folder)
        self.return_fname = return_fname

        self.dictionary = get_dictionary(self.root, split, subset, mode)

        self.taglist = get_taglist(self.root, split, subset)

        self.transform = transform

    def __getitem__(self, index):
        fname = self.dictionary[index]['path']
        fn = os.path.join(self.spec_path, fname[:-3]+'npy')
        spec = np.array(np.load(fn)).astype('float32')
        tags = self.dictionary[index]['tags']

        # Transforms the image if required
        if self.transform:
            spec = self.transform(spec)

        if self.return_fname:
            return spec, tags.astype('float32'), fname
        else:
            return spec, tags.astype('float32')

    def __len__(self):
        return len(self.dictionary)

    @property
    def num_classes(self):
        return len(self.taglist)

    def _check_exists(self):
        return os.path.exists(self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Data Location: {}\n'.format(self.spec_path)
        fmt_str += '    Classes {}\n'.format(self.num_classes)
        # if self.sampling_rate == self.original_sampling_rate:
        #     fmt_str += '    Sampling Rate: {}Hz\n'.format(self.sampling_rate)
        # else:
        #     fmt_str += ('    Sampling Rate: {}Hz (original: {}Hz)\n'
        #                 .format(self.sampling_rate,
        #                         self.original_sampling_rate))

        if self.transform:
            fmt_str += _include_repr('Transform', self.transform)
        if self.target_transform:
            fmt_str += _include_repr('Target Transform', self.target_transform)
        return fmt_str

