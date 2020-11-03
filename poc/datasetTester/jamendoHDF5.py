#!/usr/bin/env python3
import argparse
import numpy as np
import os
from tqdm import tqdm
import pickle
import h5py
from dataset import JamendoSpecFolder
import commons

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="Where are we running, 0=local, 1=kaggle, 2=triton, 3=cs-181",  type=int, default=0)
parser.add_argument("-l", "--limit", help="Limit",  type=int, default=100)
parser.add_argument("-o", "--output_filename", help="Output filename",  type=str, default='data/processed/jamendo.hdf5')
args = parser.parse_args()

## #############################################################################################################
## Main Code
## #############################################################################################################

if args.mode == 0:
    root = '/Volumes/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master'
elif args.mode == 2:
    root = '/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master'
elif args.mode == 3:
    root = '/m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master'

print(args)

output_filename = args.output_filename
output_filename = os.path.join(root, output_filename)
input_file = os.path.join(root, 'data/autotagging.tsv')

tracks, tags, extra = commons.read_file(input_file)

counter = 0

with h5py.File(output_filename, 'w') as f:
    grp_specs = f.create_group('specs')

    for k, v in tqdm(tracks.items()):
        spec = None

        counter += 1
        if counter > args.limit:  # early stopping for debugging
            break

        try:
            fpath = v['path']

            fn = os.path.join(root, 'data/raw_30s_specs', fpath[:-3]+'npy')
            spec = np.array(np.load(fn)).astype('float32')

            if spec is None:
                raise ValueError("Error, filename {} not found.".format(fpath))

            grp_specs.create_dataset(fpath, data=spec)

        except Exception as e:
            print(e)
            f.close()
            break


print("Processed %d files" % counter)
