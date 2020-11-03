#!/usr/bin/env python3
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pickle
from easydict import EasyDict
from constants import Constants
from ATPP import AudioTaggingPreProcessing, Util
import sklearn
import sklearn.model_selection
import random


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="Where are we running, 0=local, 1=kaggle, 2= triton",  type=int, default=0)
args = parser.parse_args()

## #########################################################################
## Main Code
## ########################################################################

mode = args.mode
constants = Constants(mode)

fnames = []
fnames_y = []
fnames_ids = []
fnames_short =[ ]
fnames_directory = []
counter = 0

curated = EasyDict()
noisy = EasyDict()
curated.y = pd.read_csv(constants.CSV_TRN_CURATED)
noisy.y = pd.read_csv(constants.CSV_TRN_NOISY)
submission = pd.read_csv(constants.CSV_SUBMISSION)

print("Data in curated set, labels = %d " % (curated.y.count()[0]))
print("Data in curated set, labels = %d " % (noisy.y.count()[0]))
print("Rows in submission = %d" % (submission.count()[0]))



## FIRST, read the names and lables of all files in Curated and Noisy

#curated set
for index, row in tqdm(curated.y.iterrows()):
    labels = row["labels"]
    fname = constants.TRN_CURATED + '/' + row["fname"]

    fnames_short.append(row['fname'])
    fnames_directory.append(constants.TRN_CURATED.split('/')[-1])
    fnames.append(fname)
    fnames_y.append(Util.get_label_num(labels, constants.target_labels))
    fnames_ids.append(counter)
    counter += 1


#noisy set


##for index, row in tqdm(noisy.y.iterrows()):
##    labels = row["labels"]
 ##   fname = constants.TRN_NOISY + '/' + row["fname"]
##
##    fnames_short.append(row['fname'])
##    fnames_directory.append(constants.TRN_NOISY.split('/')[-1])
##    fnames.append(fname)
##    fnames_y.append(Util.get_label_num(labels, constants.target_labels))
##    fnames_ids.append(counter)
##    counter += 1


## best 50s


## Get histograms of labels in noisy set
#This is for those data point with a signle label
df = noisy.y.copy()
df['singled'] = ~df.labels.str.contains(',')
singles_df = df[df.singled]

print("Rows in noisy set with a single label = %d" % (singles_df.count()[0]))

cat_gp = (singles_df.groupby(
    ['labels']).agg({
    'fname':'count'
}).reset_index()).set_index('labels')


labels = singles_df.labels.unique()
labels, len(labels)

import random
##select only 50 files fpr each class, from those with a single class

idxes_best50s = np.array([random.choices(singles_df[(singles_df.labels == l)].index, k=50)
                          for l in labels]).ravel()

print("Rows in best50s = %d" % (len(idxes_best50s)))

#idxes_best50s = set(idxes_best50s)  ## Remove repeated ids, NOTE: this might not be a good idea, as oversampling is good for imablanced datasets
                                    #  but this screws the generation of the hd5 file
best50s_df = singles_df.loc[idxes_best50s]

print("Unique rows in best50s = %d" % (best50s_df.count()[0]))


for index, row in tqdm(best50s_df.iterrows()):
    labels = row["labels"]
    fname = constants.TRN_NOISY + '/' + row["fname"]
    fnames_short.append(row['fname'])
    fnames_directory.append(constants.TRN_NOISY.split('/')[-1])

    fnames.append(fname)
    fnames_y.append(Util.get_label_num(labels, constants.target_labels))
    fnames_ids.append(counter)
    counter += 1

train_ids, valid_ids = sklearn.model_selection.train_test_split(fnames_ids, test_size=0.3)


## ###########################################################################
## SAve to disk
## ###########################################################################

with open(os.path.join(constants.WORK, 'fnames.pkl'), 'wb') as pickle_file:
    pickle.dump(fnames,  pickle_file)
with open(os.path.join(constants.WORK, 'fnames_y.pkl'), 'wb') as pickle_file:
    pickle.dump(fnames_y,  pickle_file)
with open(os.path.join(constants.WORK, 'fnames_ids.pkl'), 'wb') as pickle_file:
    pickle.dump(fnames_ids,  pickle_file)
with open(os.path.join(constants.WORK, 'fnames_short.pkl'), 'wb') as pickle_file:
    pickle.dump(fnames_short,  pickle_file)
with open(os.path.join(constants.WORK, 'fnames_directory.pkl'), 'wb') as pickle_file:
    pickle.dump(fnames_directory,  pickle_file)
with open(os.path.join(constants.WORK, 'train_ids.pkl'), 'wb') as pickle_file:
    pickle.dump(train_ids,  pickle_file)
with open(os.path.join(constants.WORK, 'valid_ids.pkl'), 'wb') as pickle_file:
    pickle.dump(valid_ids,  pickle_file)

print("====================")
print("Generated files")
print("fnames with shape %d" %(len(fnames)))
print("fnames_y with shape %d" %(len(fnames_y)))
print("fnames_iods with shape %d" %(len(fnames_ids)))
print("fnames_short with shape %d" %(len(fnames_short)))
print("fnames_directory with shape %d" %(len(fnames_directory)))
print("train_ids with shape %d" %(len(train_ids)))
print("valid_ids with shape %d" %(len(valid_ids)))