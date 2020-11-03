#!/usr/bin/env python3
import argparse
import librosa
import numpy as np
import os
import math
from tqdm import tqdm
import random
from easydict import EasyDict
from constants import Constants
from ATPP import AudioTaggingPreProcessing, Util

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="Where are we running, 0=local, 1=kaggle, 2= triton",  type=int, default=0)
parser.add_argument("-i", "--startId", help="Start id",  type=int, default=0)
parser.add_argument("-s", "--set", help="Set (0 = training, 1 = valid)",  type=int, default=0)
args = parser.parse_args()

## #############################################################################################################
## Main Code
## #############################################################################################################

constants = Constants(args.mode)

## Parameters for the experiment

# For the spectrograms
config = EasyDict()
config.resampling_rate = 22050
config.hop_length_ms = 10
config.hop_length = math.ceil(config.hop_length_ms / 1000 * config.resampling_rate)
config.fmin = 20  # min freq
config.fmax = config.resampling_rate // 2  # max freq for the mels
config.n_mels = 128
config.n_fft = 2048
config.useMel = True
config.fixedLength = False  # If the spectrograms should have the same size
config.max_length = 0


## --------------------------------------------------------------------
# Read the data and generate data lists X and y

X = []  # list of features (e.g. mel spectrograms)
y = []  # list of one hot encoded vectors of labels
import pickle

with open(os.path.join(constants.WORK,'fnames.pkl'), 'rb') as pickle_file:
    fnames = pickle.load(pickle_file)
with open(os.path.join(constants.WORK,'fnames_y.pkl'), 'rb') as pickle_file:
    fnames_y = pickle.load(pickle_file)
with open(os.path.join(constants.WORK,'fnames_ids.pkl'), 'rb') as pickle_file:
    fnames_ids = pickle.load(pickle_file)
with open(os.path.join(constants.WORK,'fnames_short.pkl'), 'rb') as pickle_file:
    fnames_short = pickle.load(pickle_file)
with open(os.path.join(constants.WORK,'fnames_directory.pkl'), 'rb') as pickle_file:
    fnames_directory = pickle.load(pickle_file)
with open(os.path.join(constants.WORK,'train_ids.pkl'), 'rb') as pickle_file:
    train_ids = pickle.load(pickle_file)
with open(os.path.join(constants.WORK,'valid_ids.pkl'), 'rb') as pickle_file:
    valid_ids = pickle.load(pickle_file)
counter = 0

print("Working on %d set (0=trainig, 1= valid)"%(args.set))
print("Loaded files ")
print("fnames with shape %d" %(len(fnames)))
print("fnames_y with shape %d" %(len(fnames_y)))
print("fnames_ids with shape %d" %(len(fnames_ids)))
print("fnames_short with shape %d" %(len(fnames_short)))
print("fnames_directory with shape %d" %(len(fnames_directory)))
print("train_ids with shape %d" %(len(train_ids)))
print("valid_ids with shape %d" %(len(valid_ids)))


## batchsize
if args.mode == 0:
    tmp = 20
else:
    tmp = 100


startId = args.startId * tmp
stopId = startId + tmp


full_ids = [*train_ids, *valid_ids]
print("full_ids with shape %d" %(len(full_ids)))
full_ids = full_ids[startId:stopId]


#train_ids = train_ids[startId:stopId]
#valid_ids = valid_ids[startId:stopId]


def worker_procAudio(idx, path, conf, constants):
    # idx is the content of the fnames_ids that we want
    # path = where to save the computed spectrograms and labels
    # print(fnames[idx])
    tmpPath = os.path.join(constants.DATA, fnames_directory[idx], fnames_short[idx])
    a = fnames[idx]
    wave = AudioTaggingPreProcessing.read_audio(conf, tmpPath)
    wave_y = fnames_y[idx]

    spec = AudioTaggingPreProcessing.audio_to_spectrogram(conf, wave)

    # filenmaes like: /trn_curated_0019ef41.npy
    f_spec, f_label = Util.getNpyFilename(fnames[idx], args.set == 0)
    tmpPath = os.path.join(path, constants.WORK_SPECS, f_spec)
    np.save(tmpPath, spec)
    tmpPath = os.path.join(path, constants.WORK_LABELS, f_label)
    np.save(tmpPath, wave_y)

    return 1

print("========== WORKER %04d ---> %04d" % (startId, stopId))

#Select train or valdidation files
if args.set == 0:
    tmpSet = train_ids
else:
    tmpSet = valid_ids

tmpSet = full_ids

result = []
pbar = tqdm(enumerate(tmpSet))
for i, sample in pbar:
    result.append(worker_procAudio(sample, constants.WORK, config, constants))

    pbar.set_description("Processing ids [{} -- {}], Step [{}/{}]"
                         .format(startId, stopId, i, len(tmpSet)))

#for row in tqdm(tmpSet):
#    result.append(worker_procAudio(row, constants.WORK, conf))

print( "========== WORKER %04d ---> %04d         Processed  %d files" % (startId, stopId, len(result)))







