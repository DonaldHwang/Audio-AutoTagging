#!/usr/bin/env python
from datasets.magnatagatune_dataset import MTTDataset
from parameters import get_parameters
from torch.utils import data
import utils
import torch.nn as nn
from torchaudio import transforms
import myTransforms as tforms_mine


def get_time_frequency_transform(config):
    """
    Returns a nn.Sequential block to do a time-frequency transform, and crop to the desired size.
    The spectrogram has shape: [batch, channels, freq_bins, frames]

    :param config:
    :return:
    """
    if config.use_mels:
        transformer = nn.Sequential(
            transforms.MelSpectrogram(sample_rate=config.resampling_rate,
                                      n_fft=config.n_fft,
                                      win_length=config.win_length,
                                      hop_length=config.hop_length,
                                      f_min=float(config.fmin),
                                      f_max=float(config.fmax),
                                      pad=0,
                                      n_mels=config.n_mels
                                      ),
            utils.make_module(tforms_mine.RandomCrop)(config.max_length_frames),
            transforms.AmplitudeToDB(stype='power', top_db=80),
           )
    else:
        transformer = nn.Sequential(
            transforms.Spectrogram(n_fft=config.n_fft,
                                   win_length=config.win_length,
                                   hop_length=config.hop_length,
                                   pad=0,
                                   power=2,
                                   normalized=True),
            tforms_mine.AmplitudeToDB(stype='power', top_db=80),
            utils.make_module(tforms_mine.RandomCrop)(config.max_length_frames),
        )

    return transformer


def get_data(config):
    if config.dataset == 'MTTDataset':
        dataset = MTTDataset(directory=config.dataset_path,
                             prefetch_files=config.prefetch_files,
                             read_raw_audio=False,
                             data_limit=config.data_limit)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    return data_loader, None


def main():
    # Preparation
    config = get_parameters()
    transformer = get_time_frequency_transform(config)

    data_loader_train, data_loader_valid = get_data(config)
    audio, target, fname = data_loader_train.dataset[0]
    batch_audio, batch_target, batch_fname = next(iter(data_loader_train))  # Get single batch

    spec = transformer(audio)
    batch_specs = transformer(batch_audio)

    # Sanity check for shapes
    assert spec.shape[0] == 1, 'Wrong size for spec'
    assert spec.shape[1] == config.n_fft // 2 + 1, 'Wrong size for spec'
    assert spec.shape[2] == config.max_length_frames, 'Wrong size for spec'

    assert batch_specs.shape[0] == config.batch_size, 'Wrong size for batch_spec'
    assert batch_specs.shape[1] == 1, 'Wrong size for batch_spec'
    assert batch_specs.shape[2] == config.n_fft // 2 + 1, 'Wrong size for batch_spec'
    assert batch_specs.shape[3] == config.max_length_frames, 'Wrong size for batch_spec'

    # Manual plotting
    import matplotlib.pyplot as plt
    fig = plt.figure()
    im = plt.imshow(spec[0].numpy()[::-1, :], cmap=plt.get_cmap('magma'))
    fig.colorbar(im)
    plt.show()

    # Using librosa
    utils.show_spectrogram(spec, config )
    print(spec)

    # Librosa, by batch
    utils.show_spectrogram_batch(batch_specs, batch_fname, config, gridSize=4)

if __name__ == '__main__':
    main()





