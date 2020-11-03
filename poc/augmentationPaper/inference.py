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
from torchvision import transforms as tforms_vision
from parameters import get_parameters
from torchaudio import transforms as tforms_torch
import myTransforms as tforms_mine
import utils
import os, shutil, math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
from argparse import ArgumentParser
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



    # Split into a single subset
    dataset = data.random_split(dataset, [len(dataset)])

    data_loader = data.DataLoader(dataset=dataset[0],
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)

    print("")
    print("Data loaders:")
    print("Inference dataset length = {}".format(len(data_loader.dataset)))
    print("")
    return data_loader


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
            utils.make_module(tforms_mine.RandomCrop)(config.max_length_frames),
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
            tforms_mine.AmplitudeToDB(stype='power', top_db=80),
            utils.make_module(tforms_mine.RandomCrop)(config.max_length_frames),
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

def set_args():
    parser = ArgumentParser(description='Inference')
    parser.add_argument('--result_dir', type=str, default='results_debug', help='path to results')
    parser.add_argument('--model_file', type=str, default='best_model.pth', help='filename of model')
    parser.add_argument('--params_file', type=str, default='params.yaml', help='filename of params')

    return parser.parse_args()


class PrettySafeLoader(yaml.SafeLoader):
    '''
    https://stackoverflow.com/questions/9169025/how-can-i-add-a-python-tuple-to-a-yaml-file-using-pyyaml
    '''
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)


def main():
    # Reproducibility
    np.random.seed(12345)
    torch.manual_seed(12345)

    # Preparation
    config = set_args()
    with open(os.path.join(config.result_dir, config.params_file)) as f:
        # Load model params. Overwrite with infer params in case a key is overlapping
        #config.__dict__.__init__({**yaml.safe_load(f), **config.__dict__})
        config.__dict__.__init__({**yaml.load(f, Loader=PrettySafeLoader), **config.__dict__})

    # config.__dict__['dataset_tags'] = 5
    config.data_limit = 2000
    config.num_workers = 0
    config.batch_size = 4

    data_loader = get_data(config)

    if config.use_time_freq:
        transforms = get_time_frequency_transform(config)
    else:
        transforms = None

    true_class_saved = np.load(os.path.join(config.result_dir, 'true_class.npy'))
    pred_scores_saved = np.load(os.path.join(config.result_dir, 'pred_scores.npy'))

    # From saved results:
    scores = utils.compute_evaluation_metrics(np.copy(true_class_saved), np.copy(pred_scores_saved),
                                                    data_loader.dataset.dataset.tags_list, filepath=None,
                                                    class_threshold=0.2, multilabel=True, verbose=True)


    # With no training:
    solver = Solver(data_loader, data_loader, data_loader,
                    config, None, transforms)
    scores, true_class, pred_scores = solver.test()

    # With training:
    solver = Solver(data_loader, data_loader, data_loader,
                    config, None, transforms, os.path.join(config.result_dir, 'model', config.model_file))

    scores, true_class, pred_scores = solver.test()

    # =====================================================================
    # Visualize some data

    # Class Histograms
    if not config.dataset_skip_class_hist:
        fig, ax = plt.subplots(2,1)
        ax[0].bar(np.arange(true_class_saved.shape[1]), np.sum(true_class_saved, axis=0))
        ax[0].set_title('True_class')
        ax[1].bar(np.arange(pred_scores_saved.shape[1]), np.sum(pred_scores_saved, axis=0))
        ax[1].set_title('Pred_scores')
        plt.show()

if __name__ == '__main__':
    main()
