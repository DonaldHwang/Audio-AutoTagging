#!/usr/bin/env python3
import argparse
import os, math
from tqdm import tqdm
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import transforms
import pandas as pd
import numpy as np
import csv


'''
Load Dataset (divided into train/validate/test sets)
* audio data : saved as segments in npy file
* labels : 50-d labels in csv file

Losely based on:
https://github.com/kyungyunlee/sampleCNN-pytorch/blob/master/data_loader.py

https://github.com/amalad/Multi-Scale-Music-Tagger

https://github.com/keunwoochoi/magnatagatune-list#histogram-of-tags
'''


class MTTDataset(Dataset):
    synonyms = [['beat', 'beats'],
                ['chant', 'chanting'],
                ['choir', 'choral'],
                ['classical', 'clasical', 'classic'],
                ['drum', 'drums'],
                ['electro', 'electronic', 'electronica', 'electric'],
                ['fast', 'fast beat', 'quick'],
                ['female', 'female singer', 'female singing', 'female vocals', 'female voice', 'woman', 'woman singing',
                 'women'],
                ['flute', 'flutes'],
                ['guitar', 'guitars'],
                ['hard', 'hard rock'],
                ['harpsichord', 'harpsicord'],
                ['heavy', 'heavy metal', 'metal'],
                ['horn', 'horns'],
                ['india', 'indian'],
                ['jazz', 'jazzy'],
                ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
                ['no beat', 'no drums'],
                ['no singer', 'no singing', 'no vocal', 'no vocals', 'no voice', 'no voices', 'instrumental'],
                ['opera', 'operatic'],
                ['orchestra', 'orchestral'],
                ['quiet', 'silence'],
                ['singer', 'singing'],
                ['space', 'spacey'],
                ['string', 'strings'],
                ['synth', 'synthesizer'],
                ['violin', 'violins'],
                ['vocal', 'vocals', 'voice', 'voices'],
                ['strange', 'weird']]

    def __init__(self, directory='/m/cs/work/falconr1/datasets/MTT/',
                 raw_directory='raw/',
                 processed_directory='processed/',
                 prefetch_files=True,
                 read_raw_audio=False,
                 use_synonyms=True,
                 top_tags=135,
                 data_limit=-1,
                 trim_audio=-1):

        """
        :param directory:  Root directory to dataset.
        /Volumes/scratch/work/falconr1/datasets/MTT
        /m/cs/work/falconr1/datasets/MTT
        :param raw_directory: Relative path to raw audio.
        :param processed_directory: Relative path to processed (npy) files.
        :param prefetch_files: Read all files in the init.
        :param read_raw_audio: Flag to read audio.
        :param top_tags: How many tags to use, for example, top 50.
        :param data_limit: Limit the number of files to read, for debugging. -1 = no limit.
        :param trim_audio: How many samples to read, -1 to read the full file.
        """

        self.data = []  # fnames, load on the fly
        self.root = directory
        self.raw_directory = raw_directory
        self.processed_directory = processed_directory
        self.read_raw_audio = read_raw_audio
        self.prefetch_files = prefetch_files
        self.fs = 16000
        self.data_limit = data_limit if data_limit > 0 else 100000
        self.tags_list = []
        self.fnames = []
        self.data_directory = self.raw_directory if self.read_raw_audio else self.processed_directory
        self.use_synonyms=use_synonyms
        self.trim_audio = trim_audio

        targets = {}
        targets_np = np.empty((0, 188), dtype=np.long)
        counter = 0
        with open(os.path.join(directory, 'annotations_final_fixed.csv')) as f:
            csvreader = csv.reader(f, delimiter='\t')
            j = 0
            for i, row in enumerate(csvreader):
                if i == 0:
                    self.tags_list = row[1:-1]  # Remove first and last col (clip_id, mp3_path)
                    continue

                tags = [int(x) for x in row[1:-1]]
                if max(tags) == 0:
                    continue  # skip those with no labels (about 16% of data)

                counter += 1
                if counter > self.data_limit:
                    break

                # label = np.argmax(tags)  # tags are unique (max==1)
                label = np.array(tags)

                fname = row[-1].split('.mp3')[0]
                targets[fname] = label
                targets_np = np.append(targets_np, np.expand_dims(label, axis=0), axis=0)
                self.fnames.append(fname)

        self.tags_list = np.array(self.tags_list)

        # Check is the new tags are in the same order
        ctr = 0
        for k, v in targets.items():
            if not np.equal(v, np.squeeze(targets_np[ctr, :])).all():
                print('Tags are wrong.')
                print(k)
                raise ValueError
            ctr += 1

        # Merge synonyms of tags
        if self.use_synonyms:
            for syn in self.synonyms:
                ind = [np.where(self.tags_list == x)[0][0] for x in syn]
                ind.sort()
                comb = np.amax(targets_np[:, ind], axis=1)
                targets_np[:, ind[0]] = comb
                targets_np = np.delete(targets_np, ind[1:], axis=1)
                self.tags_list = np.delete(self.tags_list, ind[1:])

        # The dataset is heavily imbalanced, so we
        # keep only the X top tags, for easier classificatioon.
        if top_tags <= targets_np.shape[-1]:
            ind = np.argsort(np.sum(targets_np, axis=0))[-top_tags:]
            targets_np = targets_np[:, ind]
            self.tags_list = self.tags_list[ind]

        ctr = 0
        for k,v in targets.items():
            targets[k] = np.squeeze(targets_np[ctr,:])
            ctr += 1



        print("MTT Dataset, with {} files to read, and {} tags.".format(len(self.fnames), len(self.tags_list)))

        if self.prefetch_files:
            print("Pre-fetching files.")

            direc = os.path.join(self.root, self.data_directory)
            print("\t\t Directory {}".format(directory))

            for fname in tqdm(self.fnames):
                if fname not in targets:
                    continue

                label = targets[fname]

                if self.read_raw_audio:
                    audio, fs = torchaudio.load(os.path.join(direc, fname + '.mp3'), normalization=True)
                    audio = audio / audio.abs().max()
                else:
                    audio = np.load(os.path.join(direc, fname + '.npy'))  # no mmap, to avoid "too many open files error"#
                    fs = self.fs
                    audio = audio / np.max(np.abs(audio))

                self.data.append((fname, audio, label, fs))

            print("Pre-fetching done, {} files loaded.".format(len(self.data)))
        else:
            self.data = targets

    def __len__(self):
        return len(self.data)

    @property
    def num_classes(self):
        return len(self.tags_list)

    def __repr__(self):
        return 'MTTDataset'

    def __getitem__(self, index):
        if self.prefetch_files:  # prefetched files
            fname, audio, target, fs = self.data[index]
            if not self.read_raw_audio:  # prefetch_files + read_war_audio = loads torch tensors
                audio = torch.from_numpy(audio).float()
        else:
            fname = self.fnames[index]
            target = self.data[fname]

            if not self.read_raw_audio:
                audio = np.load(os.path.join(self.root, self.processed_directory, fname+'.npy'), mmap_mode='r')
                audio = torch.from_numpy(audio).float()
                fs = self.fs
            else:
                try:
                    audio, fs = torchaudio.load(os.path.join(self.root, self.raw_directory, fname+'.mp3'),
                                                normalization=True)
                except Exception as e:
                    print("Error reading file:")
                    print(e)
                    print(fname)
                    raise

            audio = audio / audio.abs().max()

        # Crop if needed
        if self.trim_audio > 0:
            output_size = self.trim_audio
            input_size = audio.shape[-1]  # Assuming last axis is time

            if input_size < output_size:
                to_pad = math.floor((output_size - input_size) / 2)
                audio = np.pad(audio, [(0, 0), (to_pad, to_pad)], 'constant', constant_values=0)
            else:
                idx = np.random.randint(0, input_size - output_size)
                audio = audio[:, idx:idx + output_size]


        #mean, std = np.mean(y), np.std(y)
        #data = (data - mean) / std
        #data = y

        return audio, \
               torch.from_numpy(target.astype(np.float32)), \
               fname


def fix_MTT_annotations(root_directory):
    '''
    For some reason the MTT dataset is corrupt, some of the mo3 files have size = 0.
    Here I am remvoing those files from the annotations.
    '''

    removed_fnames = []
    with open(os.path.join(root_directory, 'annotations_final_fixed.csv'), 'w') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        with open(os.path.join(root_directory, 'annotations_final.csv')) as f:
            csvreader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(csvreader):
                if i == 0:
                    writer.writerow(row)
                    continue

                fname = row[-1]
                try:
                    # Read a few frames of the file to make sure the file is correct
                    tmp, fs = torchaudio.load(os.path.join(root_directory, 'raw', fname), num_frames=10)
                    if tmp.shape[-1] > 0:
                        writer.writerow(row)
                    else:
                        raise ValueError
                except Exception as e:
                    print('Error on file:')
                    print(fname)
                    removed_fnames.append(fname)

    print('Removed {} files.'.format(len(removed_fnames)))


if __name__ == '__main__':
    '''
    Test the dataset.
    
    Try to test all combinations for prefecth_files, and read_raw_audio
    '''

    #fix_MTT_annotations('/m/cs/work/falconr1/datasets/MTT/')

    prefetch_options = [True, False]
    read_raw_options = [True, False]

    for opt1 in prefetch_options:
        for opt2 in read_raw_options:

            dataset = MTTDataset(directory='/m/cs/work/falconr1/datasets/MTT/',
                                 raw_directory='raw/',
                                 processed_directory='processed/',
                                 prefetch_files=False,
                                 read_raw_audio=False,
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
