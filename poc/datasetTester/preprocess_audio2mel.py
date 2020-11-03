#!/usr/bin/env python3
import argparse
import numpy as np
import os
from tqdm import tqdm
from dataset import JamendoAudioFolder_audtorch, JamendoAudioFolder_torchaudio, JamendoAudioFolder_npy
import torch
from ATPP import AudioTaggingPreProcessing as ATPP
import audtorch.transforms as tforms2
import math

'''
Preprocess audio files to mel spectrograms.
'''

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--start_id", help="Start id",  type=int, default=0)
parser.add_argument('--platform', help="Where are we running, 0=local, 1=kaggle, 2= triton", type=int, default=3)
parser.add_argument('--subset', type=str, default='top50tags')
parser.add_argument('--chunk_size', type=int, default=10)
parser.add_argument('--output_path', type=str, default='data/processed/specs/')
parser.add_argument('--mode', type=str, default='train', help="{'train', 'validation', 'test'}")
parser.add_argument('--mono', type=bool, default=False, help="Save as mono files")
parser.add_argument('--filetype', type=str, default='npy', help='{npy, pt}')

# Parameters for spectrograms
spec_group = parser.add_argument_group('spectrogram')
spec_group.add_argument('--resampling_rate', type=int, default=44100,
                        help="Sampling rate to be used.")
spec_group.add_argument('--hop_length_ms', type=int, default=25,
                        help="Hop length in milliseconds.")
spec_group.add_argument('--fmin', type=int, default=20,
                        help="Minimum frequency for the spectrograms.")
spec_group.add_argument('--fmax', type=int, default=22050,
                        help="Maximum frequency for the spectrograms.")
spec_group.add_argument('--n_mels', type=int, default=96,
                        help="Number of mel bins.")
spec_group.add_argument('--n_fft', type=int, default=2048,
                        help="FFT size.")
spec_group.add_argument('--use_mels', type=bool, default=True,
                        help="Use mel filters.")
spec_group.add_argument('--max_length', type=int, default=-44100,
                        help="Desired length in samples for the input audio. Positive values will use a random crop of"
                             "the audio file.")


def ms_to_samples(ms, sampling_rate):
    return math.ceil(ms / 1000 * sampling_rate)

config = parser.parse_args()

config.hop_length = ms_to_samples(config.hop_length_ms, config.resampling_rate)

## #############################################################################################################
## Main Code
## #############################################################################################################

print("Preprocessing - audio2mel")
print(config)
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
                                      return_fname=True,
                                      transform=tforms2.Compose([
                                           tforms2.RandomCrop(config.max_length),
                                           tforms2.Downmix(1),
                                           tforms2.Normalize()]
                                           ),
                                      )

## TODO use the audio dataset, not the npy

# dataset = JamendoAudioFolder_npy(root,
#                                  config.subset,
#                                  split,
#                                  config.mode,
#                                  trim_to_size=config.max_length,
#                                  return_fname=True,
#                                  transform=tforms2.Compose([
#                                            tforms2.Downmix(1),
#                                            tforms2.Normalize()]
#                                            ),
#                                  )

if not os.path.exists(full_output_path):
    os.mkdir(full_output_path)

## TODO Save config to file in ouput directory

def worker_procAudio(audio, fname, output_path, type):
    # sample = audio data read from dataset
    # fname = filenmae, e.g. '14/1234.mp3'
    # output_path = path where to save files
    # type = {'npy', 'pt'}


    #tmpPath = os.path.join(root, fname)
    #wave = ATPP.read_audio(tmpPath, config.resampling_rate, config.max_length, fixedLength=True)

    audio = audio.squeeze()  # For librosa, mono audio should have 1 dim, i.e (time)

    spec = ATPP.audio_to_spectrogram(audio, config.use_mels,
                                     config.resampling_rate, config.hop_length, config.n_fft,
                                     config.n_mels, config.fmin, config.fmax)

    if type == 'npy':
        tmpPath = os.path.join(root, output_path, fname[:-3] + 'npy')

        if not os.path.exists(os.path.dirname(tmpPath)):
            os.mkdir(os.path.dirname(tmpPath))
        np.save(tmpPath, spec)
    elif type == 'pt':
        tmpPath = os.path.join(root, output_path, fname[:-3] + 'pt')

        if not os.path.exists(os.path.dirname(tmpPath)):
            os.mkdir(os.path.dirname(tmpPath))

        torch.save(
            spec,
            os.path.join(tmpPath)
        )

    ## TODO display some statistics about the specs, to make sure they are correct
    ## TODO display spec shape
    ## TODO maybe visualize the specs

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







