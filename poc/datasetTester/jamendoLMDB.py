#!/usr/bin/env python3
import argparse
import numpy as np
import os
from tqdm import tqdm
import commons
import lmdb
import msgpack
import msgpack_numpy as mg
import math
import pandas as pd
import gc
# Useful
# https://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions
#
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="Where are we running, 0=local, 1=kaggle, 2=triton, 3=cs-181",  type=int, default=0)
parser.add_argument("-l", "--limit", help="Limit",  type=int, default=100)
parser.add_argument("-o", "--output_filename", help="Output filename",  type=str, default='data/processed/')
parser.add_argument("-c", "--chunk_size", help="Chunk Size",  type=int, default=1000)
parser.add_argument("-t", "--use_raw_audio", help="Set to 1 to use raw audio files, 0 for mel specs",  type=int, default=0)
parser.add_argument("--skip_subdirs", help="List of subdirectories to skip, example: '1 2 3'",  type=str, default="[]")
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
chunk_size = 1000

tracks, tags, extra = commons.read_file(input_file)

tmp = [x[1] for x in tracks.items()]  # get second element of tuple
tracks_df = pd.DataFrame(tmp)

def processVanilla():
    with lmdb.open(output_filename, map_size=int(1e13), writemap=False) as env:
        def split_dict_equally(input_dict, chunks=2):
            "Splits dict by keys. Returns a list of dictionaries."
            # prep with empty dicts
            return_list = [dict() for idx in range(chunks)]
            idx = 0
            for k, v in input_dict.items():
                return_list[idx][k] = v
                if idx < chunks - 1:  # indexes start at 0
                    idx += 1
                else:
                    idx = 0
            return return_list

        def process():
            chunk_counter = 0
            counter = 0
            chunk_total = math.ceil(len(tracks) / args.chunk_size)
            #for batch in chunks(tracks.items(), args.chunk_size):
            for batch in split_dict_equally(tracks, chunk_total):
                chunk_counter += 1
                print("------- Chunk ------- {} / {}".format(chunk_counter, chunk_total))

                with env.begin(write=True) as txn:
                    for k, v in tqdm(batch.items()):
                        datum = None

                        counter += 1
                        if counter > args.limit:  # early stopping for debugging
                            return counter

                        fpath = v['path']

                        if args.use_raw_audio == 1:  # Raw audio files
                            fn = os.path.join(root, 'data/processed/audio_npy', fpath[:-3] + 'npy')
                            if not os.path.exists(fn):
                                counter -= 1
                                continue  # Skip files processed in the splits. TODO, add support for all files
                            datum = np.array(np.load(fn)).astype('float32')
                        else:  # Precomputed Mel spectrograms
                            fn = os.path.join(root, 'data/raw_30s_specs', fpath[:-3] + 'npy')
                            datum = np.array(np.load(fn)).astype('float32')

                        if datum is None:
                            raise ValueError("Error, filename {} not found.".format(fpath))

                        #txn.put(fpath.encode('ascii'), np.ascontiguousarray(spec))
                        txn.put(fpath.encode('ascii'), msgpack.packb(datum, default=mg.encode))
            return counter

        counter = process()
    return counter

def process_subdirectories():
    # Get list of subdirectories
    tmp = tracks_df.loc[:, 'path']  # Get column path, as series
    tmp2 = tmp.apply(lambda x: x[0:1])  # extract first  character, whcih are subdirs
    # print(tmp2.loc[0:5])

    # how many rows for each sub?
    # tmp2.loc[:].value_counts()

    subdirs = tmp2.unique()  # Get unique values
    tracks_df['subdir'] = tmp2  # add column of subdirs

    envs = []
    counter = 0
    for sub in subdirs:
        if sub in args.skip_subdirs.split():
            continue
        batch = tracks_df.loc[tracks_df['subdir'] == sub]  # select rows with this subdir
        print('-------SubDir {} ------ {} rows'.format(sub, batch.shape[0]))

        # Another way to find out how many rows
        # tmp = tracks_df['subdir'] == sub
        # tmp.value_counts()

        sub_path = os.path.join(output_filename, sub)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        with lmdb.open(sub_path, map_size=int(1e13), writemap=False) as env:
            with env.begin(write=True) as txn:
                for idx, row in tqdm(batch.iterrows()):
                    datum = None

                    counter += 1
                    if counter > args.limit:  # early stopping for debugging
                        break

                    fpath = row['path']

                    if args.use_raw_audio == 1:  # Raw audio files
                        fn = os.path.join(root, 'data/processed/audio_npy', fpath[:-3] + 'npy')
                        if not os.path.exists(fn):
                            counter -= 1
                            continue  # Skip files that are not present in the split TODO, add support for all files
                        datum = np.array(np.load(fn)).astype('float32')
                    else:  # Precomputed Mel spectrograms
                        fn = os.path.join(root, 'data/raw_30s_specs', fpath[:-3] + 'npy')
                        datum = np.array(np.load(fn)).astype('float32')

                    if datum is None:
                        raise ValueError("Error, filename {} not found.".format(fpath))

                    txn.put(fpath.encode('ascii'), msgpack.packb(datum, default=mg.encode))

                    del datum
        gc.collect()
    return counter

#ct = processVanilla()  # Single LMDB
ct = process_subdirectories() # One LMDB for each subdir

print("Processed %d files" % ct)
