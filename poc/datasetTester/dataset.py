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

class JamendoAudioFolder_torch(Dataset):
    '''
    Creates a data set that reads individual numpy files, precomputed from audio files (mp3) of the Jamendo dataset.
    '''

    audio_path = 'data/processed/audio_pt'

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
        fn = os.path.join(self.data_path, fname[:-3] + 'pt')
        audio = torch.load(fn)
        tags = self.dictionary[index]['tags']

        # # Crop if needed
        # if self.trim_to_size > 0:
        #     output_size = self.trim_to_size
        #     input_size = audio.shape[-1]  # Assuming last axis is time
        #

        # Keep only relevant channels, shape should be (channels, time)
        channels = audio.shape[0] if not self.mono else 1
        audio = audio[0:channels, :].astype('float32')

        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            tags = self.target_transform(tags)

        if self.return_fname:
            return audio, torch.from_numpy(tags), fname
        else:
            return audio, torch.from_numpy(tags)

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

        # Keep only relevant channels, shape should be (channels, time)
        channels = audio.shape[0] if not self.mono else 1

        # Crop if needed
        if self.trim_to_size > 0:
            output_size = self.trim_to_size
            input_size = audio.shape[-1]  # Assuming last axis is time

            if input_size < output_size:  # TODO: add padding?
                audio = audio[0:channels, :]
            else:
                idx = np.random.randint(0, input_size - output_size)
                audio = audio[0:channels, idx:idx + output_size ]

        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            tags = self.target_transform(tags)

        if self.return_fname:
            return np.array(audio), torch.from_numpy(tags), fname
        else:
            return np.array(audio), torch.from_numpy(tags)

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


class JamendoAudioFolder_audtorch(Dataset):
    '''
    Creates a data set that reads individual audio files (mp3) of the Jamendo dataset, using the audtorch packagage as
    backend.
    '''

    audio_path = 'data/raw_30s/'

    def __init__(self, root, subset, split, mode='train', transform=None, target_transform=None, return_fname=False):
        self.mode = mode
        self.root = root
        self.subset = subset
        self.split = split
        self.mode = mode
        self.data_path = os.path.join(self.root, self.audio_path)
        self.return_fname = return_fname

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
        fn = os.path.join(self.data_path, fname)

        audio, sample_rate = ut.load(fn)  # this is numpy
        tags = self.dictionary[index]['tags']

        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            tags = self.target_transform(tags)

        if self.return_fname:
            return audio, torch.from_numpy(tags), fname
        else:
            return audio, torch.from_numpy(tags)

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

    def __init__(self, root, subset, split, mode='train', transform=None, normalize=True, target_transform=None, return_fname=False):
        self.mode = mode
        self.root = root
        self.subset = subset
        self.split = split
        self.mode = mode
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
            Tuple[torch.Tensor, torch.Tensor, str]: The output tuple (audio, target, fname) where target
            is a one hot encoded tensor or the audio labels, fname is the fname of the file, e.g. '/14/1234.mp3'
        '''
        fname = self.dictionary[index]['path']
        fn = os.path.join(self.data_path, fname)

        if not self.normalize:
            audio, sample_rate = torchaudio.load(fn)  # this is a torch.tensor
        else:
            audio, sample_rate = torchaudio.load(fn, normalization=lambda x: torch.abs(x).max())
        tags = self.dictionary[index]['tags']

        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            tags = self.target_transform(tags)

        if self.return_fname:
            return audio, torch.from_numpy(tags), fname
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


class JamendoSpecHDF5(Dataset):
    '''
    Creates a data set that reads precomputed spectrogram files (npy) from a single, huge HDF5 file of the Jamendo
    dataset
    '''
    def __init__(self, root, subset, split, mode='train', transform=None, hdf5_filename='data/processed/jamendo.hdf5', return_fname=False):
        self.mode = mode
        self.root = root
        self.spec_path = os.path.join(self.root, 'data/raw_30s_specs/')
        self.hdf5file = os.path.join(self.root, hdf5_filename)
        self.return_fname = return_fname

        # fn = 'data/splits/split-%d/%s_%s_dict.pickle' % (split, subset, self.mode)
        # fn = os.path.join(root, fn)
        # self.get_dictionary(fn)

        self.dictionary = get_dictionary(self.root, split, subset, mode)

        # fn_tags = 'data/splits/split-%d/tag_list_50.npy' % (split)
        # fn_tags = os.path.join(root, fn_tags)
        # self.taglist = np.load(fn_tags)

        self.taglist = get_taglist(self.root, split, subset)

        self.transform = transform

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

    def __getitem__(self, index):
        with h5py.File(self.hdf5file, 'r', swmr=True) as f:
            # e.g. '00', '1274153.mp3'
            [key_group, key_file] = self.dictionary[index]['path'].split('/')
            spec = f['specs'][key_group][key_file][()]

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


class JamendoSpecLMDB(Dataset):
    """
    Jamendo Precomputed Spectograms from LMDB dataset


    Params:
        root (string) : Path to the /mtg-jamendo-dataset-master directory.
        subset (string) : {genre, mood, instrument, all, top50tags}
        split (int) : [0,1,2,3,4]
        mode : {train, validation, test}
        transform : Dataset transfroms from torch.utils.data
        lmdb_path (string): Path to where the lmdb dataset is, relative to the jamendo root.
    """

    def __init__(self, root, subset, split, mode='train', transform=None, lmdb_path='data/processed/triton', return_fname=False):
        self.mode = mode
        self.root = root
        self.lmdbpath = os.path.join(self.root, lmdb_path)
        self.return_fname = return_fname
        #self.db = Reader(self.lmdbpath, lock=False)

        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py#L19-L20
        self.env = lmdb.open(self.lmdbpath, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        # Read keys, if we wan to have them somewhere
        with self.env.begin(write=False) as txn:
            self.keys = []
            counter = 0
            for key, _ in txn.cursor():
                counter += 1
                self.keys.append(key)
                if counter > 100:  # Folr debugging, dont read all keys
                    break

            #self.keys = [key for key, _ in txn.cursor()]  # This reads all keys, takes a bit too long

        # cache_file = '_cache_' + ''.join(c for c in root if c in string.ascii_letters)
        # if os.path.isfile(cache_file):
        #     self.keys = pickle.load(open(cache_file, "rb"))
        # else:
        #     with self.env.begin(write=False) as txn:
        #         self.keys = [key for key, _ in txn.cursor()]
        #     pickle.dump(self.keys, open(cache_file, "wb"))

        # fn = 'data/splits/split-%d/%s_%s_dict.pickle' % (split, subset, self.mode)
        # fn = os.path.join(root, fn)
        # self.get_dictionary(fn)

        self.dictionary = get_dictionary(self.root, split, subset, mode)

        # fn_tags = 'data/splits/split-%d/tag_list_50.npy' % (split)
        # fn_tags = os.path.join(root, fn_tags)
        # self.taglist = np.load(fn_tags)

        self.taglist = get_taglist(self.root, split, subset)

        self.transform = transform

    def __len__(self):
        # return self.length
        return len(self.dictionary)

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        return dictionary

    def __getitem__(self, index):
        try:
            key = self.dictionary[index]['path'].encode('ascii')
        except e:
            pass
            raise ValueError
        tags = self.dictionary[index]['tags']

        #key = self.keys[100]  # for debuggin, just get some key
        env = self.env
        with env.begin(write=False) as txn:
            buf = txn.get(key)

        spec = msgpack.unpackb(buf, object_hook=mg.decode)
        #buf = six.BytesIO()
        #buf.write(imgbuf)
        #buf.seek(0)
        #img = Image.open(buf).convert('RGB')

        if self.transform:
            spec = self.transform(spec)

        if self.return_fname:
            return spec, tags.astype('float32'), fname
        else:
            return spec, tags.astype('float32')


class JamendoSpecLMDBsubdirClass(Dataset):
    """
    Jamendo Precomputed Spectograms from LMDB dataset


    Params:
        root (string) : Path to the /mtg-jamendo-dataset-master directory.
        subdir: Subdir for the lmdb file, e.g. '00', or '14'
        transform : Dataset transfroms from torch.utils.data
        lmdb_path (string): Relative path to where the lmdb dataset is, relative to the jamendo root.

        Based on LSUN example from TorchVision:
        https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py#L19-L20
    """

    def __init__(self, root, subdir, lmdb_path='data/processed/subdirs/', return_fname=False):
        self.root = root
        self.subdir = subdir
        self.lmdbpath = os.path.join(self.root, lmdb_path, self.subdir)
        self.return_fname = return_fname

        self.env = lmdb.open(self.lmdbpath, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        # Read keys, if we want to have them somewhere
        with self.env.begin(write=False) as txn:
            self.keys = []
            counter = 0
            for key, _ in txn.cursor():
                counter += 1
                self.keys.append(key)
                if counter > 10:  # Folr debugging, dont read all keys
                    break

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Args:
            index (int): Index should be the full key, e.g. '14/1234.mp3'
        Returns:
            tuple: spectrgoram 2d nparray of the mel spectrogram
        """
        key = index
        env = self.env
        with env.begin(write=False) as txn:
            buf = txn.get(key)

        spec = msgpack.unpackb(buf, object_hook=mg.decode)

        return spec


class JamendoSpecLMDBsubdir(Dataset):
    """
    Jamendo Precomputed Spectograms from LMDB dataset, using LMDB divided in subdirectories

    Params:
        root (string) : Path to the /mtg-jamendo-dataset-master directory.
        subset (string) : {genre, mood, instrument, all, top50tags}
        split (int) : [0,1,2,3,4]
        mode : {train, validation, test}
        transform : Dataset transfroms from torch.utils.data
        lmdb_path (string): Relative path to where the lmdb dataset is, relative to the jamendo root.

    Based on:
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py#L19-L20
    """

    def __init__(self, root, subset, split, mode='train', transform=None, lmdb_path='data/processed/chunks', return_fname=False):
        self.mode = mode
        self.root = root
        self.lmdbpath = os.path.join(self.root, lmdb_path)
        self.subdirs = self.get_subdirs()
        self.return_fname = return_fname

        self.dbs = {}
        print('----Initisalizing LMDB subfiles -------')
        for sub in tqdm(self.subdirs):
            self.dbs[sub] = JamendoSpecLMDBsubdirClass(
                root=root,
                subdir=sub,
                lmdb_path=lmdb_path
            )

        self.dictionary = get_dictionary(self.root, split, subset, mode)
        self.taglist = get_taglist(self.root, split, subset)

        self.transform = transform

        # Validate Length of all sub dbs
        length = 0
        for db in self.dbs:
            length += len(db)

        if length != self.__len__():
            print("WARNING, the total length of the subdirs LMDB does not match the lenght of the dictionary")

    def get_subdirs(self):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        ## TODO make this actually read the data and get the subdirs

        raise NotImplementedError

        tmp = [x[1] for x in tracks.items()]  # get second element of tuple
        tracks_df = pd.DataFrame(tmp)

        # Get list of subdirectories
        tmp = tracks_df.loc[:, 'path']  # Get column path, as series
        tmp2 = tmp.apply(lambda x: x[0:1])  # extract first  character, whcih are subdirs
        # print(tmp2.loc[0:5])

        # how many rows for each sub?
        # tmp2.loc[:].value_counts()

        subdirs = tmp2.unique()  # Get unique values
        tracks_df['subdir'] = tmp2  # add column of subdirs

    def __len__(self):
        # return self.length
        return len(self.dictionary)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (spec, labels) 2d nparrays for the spectrogram and labels
        """
        try:
            tmp = self.dictionary[index]['path']
            key = tmp.encode('ascii')
            sub = tmp[0:1]  # subdir is the first chars in the key, e.g. in '14/1234.mp3', it is '1'
        except e:
            pass
            raise ValueError
        tags = self.dictionary[index]['tags']

        db = self.dbs[sub]
        spec = db[key]

        if self.transform:
            spec = self.transform(spec)

        if self.return_fname:
            return spec, tags.astype('float32'), fname
        else:
            return spec, tags.astype('float32')
