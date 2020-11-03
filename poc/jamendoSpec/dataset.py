#!/usr/bin/env python3'
import os
import numpy as np
import pickle
from torch.utils import data
from scripts import commons
from torchvision import transforms


class MyAudioFolder(data.Dataset):
    def __init__(self, root, subset, tr_val='train', split=0):
        self.trval = tr_val
        self.root = root
        fn = os.path.join(root, 'split', subset, 'split-'+str(split), tr_val+'dict.pickle')
        fn = os.path.join(root, 'splits', 'split-' + str(split), 'autotagging_' + str(subset) + '-' + tr_val + '.tsv')
        fn = 'data/splits/split-%d/%s_%s_dict.pickle' % (split, subset, tr_val)
        fn = os.path.join(root, fn)
        #tracks, tags, extra = commons.read_file(fn)
        #self.dictionary = tracks
        #self.tags = tags
        #self.extra = extra

        self.get_dictionary(fn)

        if subset == 'top50tags':
            tmp = 'tag_list_50.npy'
        else:
            tmp = 'tag_list.npy'

        fn_tags = 'scripts/baseline/%s' % tmp
        fn_tags = os.path.join(root, fn_tags)
        self.taglist = np.load(fn_tags)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((96, 1366)),
            transforms.ToTensor(),
        ])

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

    def __getitem__(self, index):
        fn = os.path.join(self.root, 'data/raw_30s_specs/', self.dictionary[index]['path'][:-3]+'npy')
        audio = np.array(np.load(fn)).astype('float32')
        tags = self.dictionary[index]['tags']

        # Transforms the image if required
        if self.transform:
            audio = self.transform(audio)

        return audio, tags.astype('float32')

    def __len__(self):
        return len(self.dictionary)

class AudioFolder(data.Dataset):
    def __init__(self, root, subset, tr_val='train', split=0):
        self.trval = tr_val
        self.root = root
        fn = os.path.join(root, 'split', subset, 'split-'+str(split), tr_val+'dict.pickle')
        self.get_dictionary(fn)

    def __getitem__(self, index):
        fn = os.path.join(self.root, 'npy', self.dictionary[index]['path'][:-3]+'npy')
        audio = np.array(np.load(fn))
        tags = self.dictionary[index]['tags']
        return audio.astype('float32'), tags.astype('float32')

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

    def __len__(self):
        return len(self.dictionary)



def get_audio_loader(root, subset, batch_size, tr_val='train', split=0, num_workers=0):
    data_loader = data.DataLoader(dataset=MyAudioFolder(root, subset, tr_val, split),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    tmp = MyAudioFolder(root, subset, tr_val, split)
    data_loader.tag_list = tmp.taglist
    return data_loader

