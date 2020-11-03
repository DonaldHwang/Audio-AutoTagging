#!/usr/bin/env python3
import argparse
import numpy as np
import os
from tqdm import tqdm
from dataset import JamendoAudioFolder_audtorch, JamendoAudioFolder_torchaudio
import torch


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--start_id", help="Start id",  type=int, default=0)
parser.add_argument('--platform', help="Where are we running, 0=local, 1=kaggle, 2= triton", type=int, default=3)
parser.add_argument('--subset', type=str, default='top50tags')
parser.add_argument('--chunk_size', type=int, default=10)
parser.add_argument('--output_path', type=str, default='data/processed/yolo/')
parser.add_argument('--mode', type=str, default='train', help="{'train', 'validation', 'test'}")
parser.add_argument('--mono', type=bool, default=False, help="Save as mono files")
parser.add_argument('--filetype', type=str, default='npy', help='{npy, pt}')
config = parser.parse_args()

## #############################################################################################################
## Main Code
## #############################################################################################################


# ==============================================================
# Config and parameters

if config.platform == 0:
    root = '/Volumes/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master'
elif config.platform == 2:
    root = '/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master'
elif config.platform == 3:
    root = '/m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master'

split = 0
start_id = config.start_id * config.chunk_size
stop_id = start_id + config.chunk_size
full_output_path = os.path.join(root, config.output_path)

dataset = JamendoAudioFolder_audtorch(root,
                                      config.subset,
                                      split,
                                      config.mode,
                                      transform=None,
                                      return_fname=True,
                                      )

if not os.path.exists(full_output_path):
    os.mkdir(full_output_path)

def worker_procAudio(audio, fname, output_path, type):
    # sample = audio data read from dataset
    # fname = filenmae, e.g. '14/1234.mp3'
    # output_path = path where to save files
    # type = {'npy', 'pt'}

    if type == 'npy':
        tmpPath = os.path.join(root, output_path, fname[:-3] + 'npy')

        if not os.path.exists(os.path.dirname(tmpPath)):
            os.mkdir(os.path.dirname(tmpPath))
        np.save(tmpPath, audio)
    elif type == 'pt':
        tmpPath = os.path.join(root, output_path, fname[:-3] + 'pt')

        if not os.path.exists(os.path.dirname(tmpPath)):
            os.mkdir(os.path.dirname(tmpPath))

        torch.save(
            audio,
            os.path.join(tmpPath)
        )

    return 1

# ==============================================================
# Start

print("========== WORKER %04d ---> %04d" % (start_id, stop_id))

result = []
pbar = tqdm(range(start_id, stop_id, 1))
ctr = 0
for i in pbar:
    ctr += 1
    audio, _, fname = dataset[i]

    if config.mono:
        audio = np.mean(audio, axis=0)  # (channels, time), mono = mean of all channels
        audio = np.expand_dims(audio, axis=0)

    result.append(worker_procAudio(audio, fname, full_output_path, config.filetype))

    pbar.set_description("Processing ids [{} -- {}], Step [{}/{}]"
                         .format(start_id, stop_id, ctr, config.chunk_size))

print("========== WORKER %04d ---> %04d         Processed  %d files" % (start_id, stop_id, len(result)))







