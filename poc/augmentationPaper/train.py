#!/usr/bin/env python
#from datasets import magnatagatune_dataset
from datasets.magnatagatune_dataset import MTTDataset
from datasets.mnist_wrapper import MYMNIST
from datasets.jamendo_audio_dataset import JamendoAudioDataset
from datasets.speech_commands_dataset import SpeechCommandsDataset
from solver import Solver
import torch
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as tforms_vision
import torchvision
from parameters import get_parameters
from torchaudio import transforms as tforms_torch
import myTransforms as tforms_mine
import utils
import os, shutil, math
from tqdm import tqdm
import seaborn as sns
import numpy as np
from datetime import datetime
sns.set()


def get_data(config):
    if config.dataset == 'MTTDataset':
        # directory='/m/cs/work/falconr1/datasets/MTT/'
        dataset = MTTDataset(directory=config.dataset_path,
                             prefetch_files=config.prefetch_files,
                             read_raw_audio=False,
                             data_limit=config.data_limit,
                             trim_audio=config.trim_audio_pre,
                             top_tags=config.dataset_tags)

    elif config.dataset == 'SpeechCommands':
        dataset = SpeechCommandsDataset(root_directory=config.dataset_path,
                                        raw_directory='raw/',
                                        processed_directory='processed/',
                                        prefetch_files=config.prefetch_files,
                                        read_raw_audio=True,
                                        data_limit=config.data_limit,
                                        trim_audio=config.trim_audio_pre)

    elif config.dataset == 'MNIST':
        # Image processing
        train_transform = tforms_vision.Compose([
            tforms_vision.Resize(config.n_mels),
            tforms_vision.CenterCrop((config.n_mels, config.max_length_frames)),
            tforms_vision.ToTensor(),
            ])

        # root='/m/cs/work/falconr1/datasets/'
        dataset = MYMNIST(root='/m/cs/work/falconr1/datasets/',
                          train=True,
                          transform=train_transform,
                          target_transform=utils.OneHot(10),
                          download=False)

    elif config.dataset == 'JamendoAudioDataset':
        # root_directory='/m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master/'
        resampler = get_resampling_transform(config) if config.new_fs != config.original_fs else None
        dataset = JamendoAudioDataset(root_directory='/m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master/',
                                      mode='train',
                                      subset='top50tags',
                                      split=0,
                                      mono=True,
                                      trim_to_size=config.trim_audio_pre,
                                      transform=resampler,
                                      data_limit=config.data_limit)

    # Split into subsets
    if len(dataset) / 4 <= 1:
        train_len, valid_len, test_len = 2,1,1
    else:
        train_len = math.floor(len(dataset) * 0.7)
        valid_len = math.floor(len(dataset) * 0.1)
        test_len = len(dataset) - (train_len + valid_len)
    train_set, valid_set, test_set = data.random_split(dataset, [train_len, valid_len, test_len])

    data_loader_train = data.DataLoader(dataset=train_set,
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers)

    data_loader_valid = data.DataLoader(dataset=valid_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=config.num_workers)

    data_loader_test = data.DataLoader(dataset=test_set,
                                       batch_size=config.batch_size,
                                       shuffle=False,
                                       num_workers=config.num_workers)

    print("")
    print("Data loaders:")
    print("Train length = {}".format(len(data_loader_train.dataset)))
    print("Valid length = {}".format(len(data_loader_valid.dataset)))
    print("Test length = {}".format(len(data_loader_test.dataset)))
    print("")
    return data_loader_train, data_loader_valid, data_loader_test


def get_time_frequency_transform(config):
    """
    Returns a nn.Sequential block to do a time-frequency transform, and crop to the desired size.
    The spectrogram has shape: [batch, channels, freq_bins, frames]

    :param config:
    :return:
    """
    if config.use_mels:
        transformer = nn.Sequential(
            tforms_torch.MelSpectrogram(sample_rate=config.new_fs,
                                        n_fft=config.n_fft,
                                        win_length=config.win_length,
                                        hop_length=config.hop_length,
                                        f_min=float(config.fmin),
                                        f_max=float(config.fmax),
                                        pad=0,
                                        n_mels=config.n_mels
                                        ),
            #utils.make_module(tforms_mine.RandomCrop)(config.max_length_frames),
            tforms_mine.RandomCrop((1,
                                    config.n_mels if config.use_mels else config.n_fft // 2 + 1,
                                    config.max_length_frames),
                                   value=0),
            tforms_torch.AmplitudeToDB(stype='power', top_db=80),
            tforms_mine.ReScaleSpec([-1, 1]),

           )
    else:
        transformer = nn.Sequential(
            tforms_torch.Spectrogram(n_fft=config.n_fft,
                                     win_length=config.win_length,
                                     hop_length=config.hop_length,
                                     pad=0,
                                     power=2,
                                     normalized=True),
            tforms_mine.RandomCrop((1,
                                    config.n_mels if config.use_mels else config.n_fft // 2 + 1,
                                    config.max_length_frames),
                                   value=0),
            tforms_mine.AmplitudeToDB(stype='power', top_db=80),
            #utils.make_module(tforms_mine.RandomCrop)(config.max_length_frames),
            tforms_mine.ReScaleSpec([-1, 1]),
        )

    return transformer


def get_resampling_transform(config):
    '''
    Torchaudio has no support for batches when resmapling.
    :param config:
    :return:
    '''
    return nn.Sequential(tforms_torch.Resample(orig_freq=config.original_fs, new_freq=config.new_fs))


def main():
    # Reproducibility
    np.random.seed(12345)
    torch.manual_seed(12345)

    # Preparation
    config = get_parameters()

    # Logging configuration
    writer = None
    if config.tensorboard:
        path_tensorboard = f'{config.logging_dir}/{config.experiment_description}'
        if config.debug_mode:  # Clear tensorboard when debugging
            if os.path.exists(path_tensorboard):
                shutil.rmtree(path_tensorboard)
        writer = SummaryWriter(path_tensorboard)

    data_loader_train, data_loader_valid, data_loader_test = get_data(config)

    if config.use_time_freq:
        transforms = get_time_frequency_transform(config)
    else:
        transforms = None

    # =====================================================================
    # Visualize some data
    tmp_audio = None
    tmp_spec = None
    tmp_data, targets, _ = data_loader_train.dataset[0]  # audio is [channels, timesteps]

    # Is the data audio or image?
    if len(tmp_data.shape) == 2:
        tmp_audio = tmp_data
    else:
        tmp_spec = tmp_data

    if config.use_time_freq:
        tmp_spec = transforms(tmp_audio)    # spec is [channels, freq_bins, frames]

    if tmp_spec is not None:
        utils.show_spectrogram(tmp_spec, config)

    if writer is not None:
        if tmp_audio is not None:
            # Store 5 secs of audio
            ind = tmp_audio.shape[-1] if tmp_audio.shape[-1] <= 5 * config.original_fs else 5 * config.original_fs
            writer.add_audio('input_audio', tmp_audio[:,0:ind], None, config.original_fs)

            tmp_audios = []
            fnames = []
            for i in range(4):
                aud, _, fn = data_loader_train.dataset.dataset[i]
                fnames.append(fn)
                tmp_audios.append(aud)
            writer.add_figure('input_waveform', utils.show_waveforms_batch(tmp_audios, fnames, config), None)

        # Analyze some spectrograms
        if tmp_spec is not None:
            img_tform = tforms_vision.Compose([
                        tforms_vision.ToPILImage(),
                        tforms_vision.ToTensor(),
                        ])

            writer.add_image('input_spec', img_tform(tmp_spec), None)  # Raw tensor
            writer.add_figure('input_spec_single', utils.show_spectrogram(tmp_spec, config), None)  # Librosa

            if config.use_time_freq:
                tmp_specs = []
                fnames = []
                for i in range(4):
                    aud, _, fn = data_loader_train.dataset.dataset[i]
                    tmp_specs.append(transforms(aud))
                    fnames.append(fn)

                writer.add_figure('input_spec_batch', utils.show_spectrogram_batch(tmp_specs, fnames, config), None)
                writer.add_figure('input_spec_histogram', utils.get_spectrogram_histogram(tmp_specs), None)
                del tmp_specs, fnames, aud, fn, i


    # Class Histograms
    if not config.dataset_skip_class_hist:
        fig_classes = utils.get_class_histograms(data_loader_train.dataset,
                                                 data_loader_valid.dataset,
                                                 data_loader_test.dataset,
                                                 one_hot_encoder=utils.OneHot if config.dataset=='MNIST' else None,
                                                 data_limit=200 if config.debug_mode else None)
        if writer is not None:
            writer.add_figure('class_histogram', fig_classes, None)


    # =====================================================================
    # Train and Test
    solver = Solver(data_loader_train, data_loader_valid, data_loader_test,
                    config, writer, transforms)
    solver.train()
    scores, true_class, pred_scores = solver.test()

    # =====================================================================
    # Save results

    np.save(open(os.path.join(config.result_dir, 'true_class.npy'), 'wb'), true_class)
    np.save(open(os.path.join(config.result_dir, 'pred_scores.npy'), 'wb'), pred_scores)

    utils.compare_predictions(true_class, pred_scores, config.result_dir)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
