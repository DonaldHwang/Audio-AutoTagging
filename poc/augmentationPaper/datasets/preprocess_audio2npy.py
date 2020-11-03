import os
import argparse
import numpy as np
import torch
import torchaudio
from torchaudio import transforms
from magnatagatune_dataset import MTTDataset
from tqdm import tqdm


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--start_id", help="Start id",  type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='MTTdataset')
    parser.add_argument('--data_limit', type=int, default=-1)
    parser.add_argument('--resampling_rate', type=int, default=16000, help="Sampling rate to resample and save file as. Use -1 to keep the dataset samplign rate")
    parser.add_argument('--filetype', type=str, default='npy', help='{npy, pt}')
    parser.add_argument('--mono', type=int, default=1, help='Saver as mono, 1 = true')
    config = parser.parse_args()

    print("")
    print("======== PreProcess Audio2Npy =======")
    print("")

    # Print the experiment config
    ctr = 0
    for k, v in vars(config).items():
        ctr += 1
        if ctr % 10 == 0: print(' ')
        print('{} \t {}'.format(k.ljust(15, ' '), v))
    print("")

    return config


def get_dataset(config):
    if config.dataset == 'MTTdataset':
        dataset = MTTDataset(directory='/m/cs/work/falconr1/datasets/MTT/',
                             raw_directory='raw/',
                             processed_directory='processed/',
                             prefetch_files=False,
                             read_raw_audio=True,
                             use_synonyms=False,
                             data_limit=config.data_limit)

    return dataset


def worker_proccess_audio(audio, fname, output_path, type='npy'):
    """Saves an audio file read as either npy or torch.tensor to disk.

    sample = audio data read from dataset
    fname = filenmae, e.g. '0/williamson-a_few_things_to_hear_before_we_all_blow_up-10-a-30-59.mp3'
    output_path = path where to save files
    type = {'npy', 'pt'}

    Returns 1 if there were no errors.
    """

    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()

    assert fname.find('.' ) < 0, 'Fname should not contain the extension.'

    if type == 'npy':
        tmpPath = os.path.join(output_path, fname + '.npy')

        if not os.path.exists(os.path.dirname(tmpPath)):
            os.mkdir(os.path.dirname(tmpPath))

        if os.path.exists(tmpPath):  # Delete previous file
            os.remove(tmpPath)

        np.save(tmpPath, audio)
    elif type == 'pt':
        tmpPath = os.path.join(output_path, fname + '.pt')

        if not os.path.exists(os.path.dirname(tmpPath)):
            os.mkdir(os.path.dirname(tmpPath))

        torch.save(
            audio,
            os.path.join(tmpPath)
        )

    return 1

def main():

    # ==============================================================
    # Config and parameters
    config = get_params()

    dataset = get_dataset(config)

    if config.resampling_rate != -1:
        tforms = transforms.Resample(orig_freq=dataset.fs, new_freq=config.resampling_rate)
    else:
        tforms = None

    start_id = config.start_id * config.chunk_size
    stop_id = start_id + config.chunk_size
    full_output_path = os.path.join(dataset.root, dataset.processed_directory)

    if not os.path.exists(full_output_path):
        os.mkdir(full_output_path)

    # ==============================================================
    # Start

    print("========== WORKER %04d ---> %04d" % (start_id, stop_id))

    result = []
    pbar = tqdm(range(start_id, stop_id, 1))
    ctr = 0
    for i in pbar:
        ctr += 1
        audio, _, fname = dataset[i]

        result.append(worker_proccess_audio(audio, fname, full_output_path, config.filetype))

        pbar.set_description("Processing ids [{} -- {}], Step [{}/{}]"
                             .format(start_id, stop_id, ctr, config.chunk_size))

    print("========== WORKER %04d ---> %04d         Processed  %d files" % (start_id, stop_id, len(result)))


if __name__ == "__main__":
    main()

