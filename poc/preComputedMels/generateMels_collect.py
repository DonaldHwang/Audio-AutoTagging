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
import pickle
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="Where are we running, 0=local, 1=kaggle, 2= triton",  type=int, default=0)
parser.add_argument("-l", "--limit", help="Limit",  type=int, default=10000)
parser.add_argument("-s", "--set", help="Set (0 = training, 1 = valid)",  type=int, default=0)
parser.add_argument("-i", "--startId", help="Start id",  type=int, default=0)
args = parser.parse_args()

## #############################################################################################################
## Main Code
## #############################################################################################################

constants = Constants(args.mode)


with open(os.path.join(constants.WORK, 'fnames.pkl'), 'rb') as pickle_file:
    fnames = pickle.load(pickle_file)
with open(os.path.join(constants.WORK, 'fnames_y.pkl'), 'rb') as pickle_file:
    fnames_y = pickle.load(pickle_file)
with open(os.path.join(constants.WORK, 'fnames_ids.pkl'), 'rb') as pickle_file:
    fnames_ids = pickle.load(pickle_file)
with open(os.path.join(constants.WORK, 'fnames_short.pkl'), 'rb') as pickle_file:
    fnames_short = pickle.load(pickle_file)
with open(os.path.join(constants.WORK, 'train_ids.pkl'), 'rb') as pickle_file:
    train_ids = pickle.load(pickle_file)
with open(os.path.join(constants.WORK, 'valid_ids.pkl'), 'rb') as pickle_file:
    valid_ids = pickle.load(pickle_file)
counter = 0

print("working on %d set (0=trainig, 1= valid)"%(args.set))
print("Loaded files ")
print("fnames with shape %d" %(len(fnames)))
print("fnames_y with shape %d" %(len(fnames_y)))
print("fnames_iods with shape %d" %(len(fnames_ids)))
print("train_ids with shape %d" %(len(train_ids)))
print("valid_ids with shape %d" %(len(valid_ids)))


## #####################################################################
## Collect spectrograms
conf = EasyDict()
conf.augmentationTimes = 1

## batchsize
if args.mode == 0:
    tmp = 10000
else:
    tmp = 10000


startId = args.startId * tmp
stopId = startId + tmp


train_ids = train_ids[startId:stopId]
valid_ids = valid_ids[startId:stopId]
full_ids = [*train_ids, *valid_ids]

#Select train or valdidation files
if args.set == 0:
    tmpSet = train_ids
    fn ='data_train.hdf5'
else:
    tmpSet = valid_ids
    fn = 'data_valid.hdf5'

tmpSet = full_ids

result=[]
X_train = []
y_train = []
X_valid = []
y_valid = []
counter = 0



with h5py.File(os.path.join(constants.WORK, fn), 'w') as f:
    grp_specs = f.create_group('specs')
    grp_labels = f.create_group('labels')
    grp_fname = f.create_group(('fnames'))

    for row in tqdm(tmpSet):
        spec = None
        y = None
        tmpName = fnames[row]

        counter += 1
        if counter > args.limit:
            break

        try:
            # filenmaes like: /trn_curated-0019ef41_spec.npy
            f_spec, f_label = Util.getNpyFilename(tmpName, args.set == 0)
            spec = np.load(os.path.join(constants.WORK, constants.WORK_SPECS, f_spec))
            y = np.load(os.path.join(constants.WORK, constants.WORK_LABELS, f_label))

            if spec is None or y is None:
                raise ValueError("Error, filename not found.")

            #X_train.append(spec)
            #y_train.append(y)

            grp_specs.create_dataset(str(row), data=spec)
            grp_labels.create_dataset(str(row), data=y)
            grp_fname.create_dataset(str(row), data=fnames_short[row])



        except Exception as e:
            print("Error in file:     %s " %(tmpName))
            f.close()
            pass

#
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# X_valid = np.array(X_valid)
# y_valid = np.array(y_valid)
#
# print("X_train shape is = %s" % (str(X_train.shape)))
# print("y_train shape is = %s" % (str(y_train.shape)))
# print("X_valid shape is = %s" % (str(X_valid.shape)))
# print("y_valid shape is = %s" % (str(y_valid.shape)))



print("X_train shape is = %d" % (len(X_train)))
print("y_train shape is = %d" % (len(y_train)))
print("X_valid shape is = %d" % (len(X_valid)))
print("y_valid shape is = %d" % (len(y_valid)))

print("Processed %d files" % counter)

if args.set == 0:
    pass


    #np.save('X_train_batch_%d' % (startId), X_train)
    #np.save('y_train_batch_%d' % (startId), y_train)
else:
    with h5py.File(os.path.join(constants.WORK, 'X_valid.hdf5'), 'w') as f:
        dset = f.create_dataset("default", data=X_valid)
    with h5py.File(os.path.join(constants.WORK, 'y_valid%d.hdf5'), 'w') as f:
        dset = f.create_dataset("default", data=y_valid)
    #np.save('X_valid_batch_%d' % (startId), X_valid)
    #np.save('y_valid_batch_%d' % (startId), y_valid)



