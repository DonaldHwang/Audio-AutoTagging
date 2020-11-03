#!/usr/bin/env python3'
from dataset import JamendoSpecFolder, JamendoAudioFolder_npy, JamendoAudioFolder_torchaudio
import torch
from torchvision import transforms
import torchvision.datasets as dset
import time
from matplotlib import pyplot as plt
import numpy as np
import argparse
import torchaudio
from torchaudio import transforms as tforms
from audtorch import datasets as dsets
from audtorch.collate import Seq2Seq
import audtorch.transforms as tforms2
import os
from psutil import cpu_count
import myTransfroms as myTforms
import gc
from ATPP import AudioTaggingPreProcessing, Util
import math
from enum import Enum
from itertools import cycle

def get_result_dir_path(experiment_description: str, root: str = './results'):
    """Returns path where to save training results of a experiment specific result.

    Args:
        root: root path of where to save
        experiment_description: "epoch=50-batch=128-arch=FCN-data=FULL"

    Create the directory, "result-20190604_13_40_52-epoch=50-batch=128-arch=FCN-data=FULL"

    Return directory path(str):
        "result-20190604_13_40_52-epoch=50-batch=128-arch=FCN-data=FULL"

    """
    from datetime import datetime
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%d_%H_%M_%S")

    path = f"result-{date_time}-{experiment_description}"
    path = os.path.join(root, path)
    try:
        os.makedirs(path)
    except OSError:
        if os.path.exists(path):
            print("Path already exists")
        else:
            print(f"Couldn't create {path}.")
            path = root
    else:
        print(f"Save weights to {path}")
    finally:
        return path

class TformsSet(Enum):
    TorchAudio = 0
    Audtorch = 1
    MySet = 2


def get_train_transforms(config: object, set: TformsSet = TformsSet.Audtorch) -> object:
    if config.use_mels:
        if set == TformsSet.TorchAudio:
            trans = transforms.Compose([
                tforms2.Crop((441000, 441000 + 441000)),
                tforms.MelSpectrogram(sample_rate=config.resampling_rate,
                                      n_fft=config.n_fft,
                                      win_length=config.hop_length,
                                      hop_length=config.hop_length,
                                      f_min=float(config.fmin),
                                      f_max=float(config.fmax),
                                      pad=0,
                                      n_mels=config.n_mels
                                      ),
                tforms.AmplitudeToDB(stype='power', top_db=80),
                # transforms.ToPILImage(),
                # transforms.RandomCrop((96, 256), pad_if_needed=True,
                #                      padding_mode='reflect'),
                # transforms.ToTensor(),
            ])
        elif set == TformsSet.Audtorch:  ## no real mel spectrogram in audtorch
            trans = tforms2.Compose([
                myTforms.ToNumpy(),
                tforms2.Crop((441000, 441000 + 441000)),
                # tforms2.Normalize(),
                tforms2.Spectrogram(window_size=config.hop_length,
                                    hop_size=config.hop_length,
                                    fft_size=config.n_fft,
                                    ),
                tforms2.Log(),
                myTforms.ToTensor(),
                tforms.AmplitudeToDB(stype='magnitude', top_db=80)
            ])
        elif set == TformsSet.MySet:
            trans = tforms2.Compose([
                tforms2.Crop((441000, 441000 + 441000)),
                myTforms.Spectrogram(config)
            ])
    else:
        if set == TformsSet.TorchAudio:
            trans = transforms.Compose([
                                         tforms2.Crop((441000, 441000+441000)),
                                         tforms.Spectrogram(n_fft=config.n_fft,
                                                            win_length=config.hop_length,
                                                            hop_length=config.hop_length,
                                                            pad=0,
                                                            power=2,
                                                            normalized=True),
                                         tforms.AmplitudeToDB(stype='power', top_db=80),
                                         # tforms.MelSpectrogram(sample_rate=config.resampling_rate,
                                         #                       n_fft=config.n_fft,
                                         #                       win_length=config.hop_length,
                                         #                       hop_length=config.hop_length,
                                         #                       f_min=float(config.fmin),
                                         #                       f_max=float(config.fmax),
                                         #                       pad=0,
                                         #                       n_mels=config.n_mels),

                                         #transforms.ToPILImage(),
                                         #transforms.RandomCrop((96, 256), pad_if_needed=True,
                                         #                      padding_mode='reflect'),
                                         #transforms.ToTensor(),
                                     ])
        elif set == TformsSet.Audtorch:
            trans = tforms2.Compose([
                                    myTforms.ToNumpy(),
                                    tforms2.Crop((441000, 441000 + 441000)),
                                    #tforms2.Normalize(),
                                    tforms2.Spectrogram(window_size=config.hop_length,
                                                        hop_size=config.hop_length,
                                                        fft_size=config.n_fft,
                                                        ),
                                    myTforms.ToTensor(),
                                    tforms.AmplitudeToDB(stype='magnitude', top_db=80)
                                    ])
        elif set == TformsSet.MySet:
            trans = tforms2.Compose([
                                    tforms2.Crop((441000, 441000 + 441000)),
                                    myTforms.Spectrogram(config)
                                    ])
    return trans

def get_DataLoader(config, transforms : TformsSet = TformsSet.Audtorch):
    if config.platform == 0:
        root = '/Volumes/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master'
    elif config.platform == 2:
        root = '/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master'
    elif config.platform == 3:
        root = '/m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master'

    subset = config.subset
    split = 0
    mode = 'train'

    if config.dataset == 'JamendoAudioFolder_npy':
        dataset = JamendoAudioFolder_npy(root,
                                         subset,
                                         split,
                                         mode,
                                         trim_to_size=config.trim_size,
                                         return_fname=True,
                                         transform=get_train_transforms(config, transforms)
                                         )
    elif config.dataset == 'JamendoAudioFolder_torchaudio':
        dataset = JamendoAudioFolder_torchaudio(root,
                                         subset,
                                         split,
                                         mode,
                                         trim_to_size=config.trim_size,
                                         return_fname=True,
                                         normalize=True,
                                         transform=get_train_transforms(config, transforms)
                                         )

    subset_indices = np.random.choice(range(len(dataset)), config.data_limit, replace=False)
    subset_indices = np.array([14154,  9914, 24252, 23957,  8028, 27894, 21738, 19022])
    subset_indices = np.array([14154, 14154, 14154, 14154, 14154, 14154, 14154, 14154])  # This is to compare different tforms sets, where I want ot have the same file

    print('------ Dataset length = {}, using {} samples.'.format(len(dataset), len(subset_indices)))

    if config.collate_fn == 'seq2seq':
        collate = Seq2Seq([-1,-1], batch_first=None, sort_sequences=False)
        #collate = Seq2Seq_short([-1, -1], batch_first=None, sort_sequences=False)
    else:
        collate = torch.utils.data.dataloader.default_collate

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config.batch_size,
                                             # shuffle=True,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices),
                                             collate_fn=collate,
                                             drop_last=True,
                                             )

    return dataloader

def compareTforms(config):
    '''
    Here I compare different transfromations sets for spectrograms, using (torchaudio, audtorch, and my own custom
    spectrogram using librosa. This codes is applied to a sample audio file from the librispeech dataset.

    This code was done mostly to post as an issue in github. As a minimal working example.
    '''
    config.use_mels = False
    config.win_length = 400
    config.hop_length = 400
    config.n_fft = 2048
    config.resampling_rate = 16000
    augment1 = tforms2.Compose([
                            myTforms.ToTensor(),
                            tforms.Spectrogram(n_fft=2048,
                                               win_length=400,  # 400 samples @ 16k = 25 ms,
                                               hop_length=400,
                                               pad=0,
                                               power=2,
                                               normalized=False),
                            tforms.AmplitudeToDB(stype='power', top_db=80)
                            ])

    augment2 = tforms2.Compose([
                            tforms2.Spectrogram(window_size=400,  # 400 samples @ 16k = 25 ms
                                                hop_size=400,
                                                fft_size=2048
                                                ),
                            myTforms.ToTensor(),
                            tforms.AmplitudeToDB(stype='magnitude', top_db=80)
                            ])

    augment3 = tforms2.Compose([
                            myTforms.Spectrogram(config)
                            ])

    data1 = dsets.LibriSpeech(root='/m/cs/work/falconr1/datasets/librespeech/LibriSpeech', sets='dev-clean',
                              download=False, transform=augment1)
    data2 = dsets.LibriSpeech(root='/m/cs/work/falconr1/datasets/librespeech/LibriSpeech', sets='dev-clean',
                              download=False, transform=augment2)
    data3 = dsets.LibriSpeech(root='/m/cs/work/falconr1/datasets/librespeech/LibriSpeech', sets='dev-clean',
                              download=False, transform=augment3)


    plt.figure(figsize=(16, 8))

    titles = ['torchaudio', 'audtorch', 'myset']
    for i, data in enumerate([data1, data2, data3]):
        spec, label = data[0]

        if isinstance(spec, torch.Tensor):
            spec = spec.numpy()

        plt.subplot(1, 3, i+1)
        plt.imshow(spec.squeeze(), interpolation='nearest', cmap='inferno', origin='lower', aspect='auto')
        plt.colorbar()
        plt.title(titles[i])

    plt.savefig(os.path.join('./results', 'Test_Output_compare_specs.png'))
    plt.show()

def analyzeDatasets(config):
    '''
    This function compares the preprocessing methods of the different transforms sets using the Jamendo dataset with
    my own dataloaders. It works pretty ok.
    '''
    dataloader_torchaudio = get_DataLoader(config, TformsSet.TorchAudio)
    dataloader_audtorch = get_DataLoader(config, TformsSet.Audtorch)
    dataloader_myset = get_DataLoader(config, TformsSet.MySet)

    for i, sample in enumerate(zip(dataloader_torchaudio, dataloader_audtorch, dataloader_myset)):
        sample_torchaudio, sample_audtorch, sample_myset = sample  # unpack different dataloaders (specs, labels, fnames, raw_audios)
        spec_torchaudio = sample_torchaudio[0][0,:,:,:].detach().numpy().squeeze()
        spec_audtorch = sample_audtorch[0][0,:,:,:].numpy().squeeze()
        spec_myset = sample_myset[0][0,:,:,:].numpy().squeeze()

        # Look at them specs:
        plt.figure(figsize=(16, 8))
        plt.subplot(1,3,1)
        plt.imshow(spec_torchaudio, interpolation='nearest', cmap='inferno', origin='lower')
        plt.colorbar()
        plt.title('torchaudio')

        plt.subplot(1,3,2)
        plt.imshow(spec_audtorch, interpolation='nearest', cmap='inferno', origin='lower')
        plt.colorbar()
        plt.title('audtorch')

        plt.subplot(1,3,3)
        plt.imshow(spec_myset, interpolation='nearest', cmap='inferno', origin='lower')
        plt.colorbar()
        plt.title('myset')


        plt.savefig(os.path.join('./results', 'Test_Output_compare_specs.png'))
        plt.show()

        # Look at them audio files
        tmp = torch.cat((sample_torchaudio[3][0, :, :], sample_audtorch[3][0, :, :]), 0)
        tmp = torch.cat((tmp, sample_myset[3][0, :, :]), 0)
        tmp = torch.cat((tmp, sample_myset[3][0, :, :]), 0)  # repeat last just to have 4 files

        batch = {'audio': tmp,
                 'labels': None,
                 'fname': [sample_torchaudio[2][0], sample_audtorch[2][0], sample_myset[2][0], sample_myset[2][0]]}

        AudioTaggingPreProcessing.show_waveforms_from_audio_batch(batch, config, filename='Test_waveforms',
                                                                  filepath='./results')
        break



def main(config):
    num_epochs = config.num_epochs
    log_every = 30

    dataloader = get_DataLoader(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Variables to keep track of time
    t_total = 0
    t_epoch = 0
    t_step = 0
    last_time = time.time()
    last_epoch = time.time()
    all_time = []
    all_time_epoch = []

    total_step = len(dataloader)
    print("Beginning of training loop.")

    for j in range(num_epochs):
        step = 0
        batch_shape = None
        for sample in dataloader:
            if config.collate_fn == 'seq2seq':
                data, _, labels, _ = sample
            else:
                data, labels, fname, raw_audio = sample

            step += 1
            t = time.time()
            t_step = t - last_time
            t_total += t_step
            last_time = t
            all_time.append(t_step)

            x = data.to(device)
            y = labels.to(device)

            tmp = data[1, :, :, :]
            AudioTaggingPreProcessing.show_spectrogram(tmp, config, filepath='./results/')

            tmp2 = {'spectrogram': data,
                    'labels': labels,
                    'fname': fname}

            AudioTaggingPreProcessing.show_spectrogram_batch(tmp2, config, filepath='./results')

            tmp3 = {'audio': raw_audio,
                    'labels': labels,
                    'fname': fname}

            AudioTaggingPreProcessing.show_waveforms_from_audio_batch(tmp3, config, filepath='./results')

            spec = AudioTaggingPreProcessing.audio_to_spectrogram(raw_audio[1,:,:], config)
            AudioTaggingPreProcessing.show_spectrogram(spec, config, filepath='./results/', filename='Output_single_spec_mine.png')

            batch_shape = data.shape
            if step % log_every == 0:
                print("Epoch [{}/{}], Step [{}/{}],  t_step = {:.4f}, data_shape = {}".format(
                    j, num_epochs, step, len(dataloader), t_step, data.shape
                ))

            del sample, data, labels, x, y

        t = time.time()
        t_epoch = t - last_epoch
        last_epoch = t
        all_time_epoch.append(t_epoch)

        print("---- Summary ----- Epoch [{}/{}], t_epoch = {:.4f}, t_total = {:.4f}, data_shape = {}".format(
            j + 1, num_epochs, t_epoch, t_total, batch_shape
        ))

        gc.collect()

    ## Plot time for each step
    t = np.linspace(0, num_epochs, num=len(all_time), endpoint=False)

    plt.figure(figsize=(16, 8))
    plt.plot(t, all_time, 'r-+', label='Time per step')
    plt.legend()
    plt.savefig(os.path.join(config.result_path, 'Test_Output_time_per_step.png'))
    plt.xlabel('Epochs')
    plt.ylabel('Time (s)')
    plt.show()

    ## Plot time for each epoch
    t = np.linspace(0, num_epochs, num=len(all_time_epoch), endpoint=False)

    plt.figure(figsize=(16, 8))
    plt.plot(t, all_time_epoch, 'r-+', label='Time per epoch')
    plt.savefig(os.path.join(config.result_path, 'Test_Output_time_per_epoch.png'))
    plt.xlabel('Epochs')
    plt.ylabel('Time (s)')
    plt.show()




if __name__ == '__main__':
    # This is copied from Seyoung's code. I am not sure what it does.
    N_JOBS = cpu_count()
    os.environ['MKL_NUM_THREADS'] = str(N_JOBS)
    os.environ['OMP_NUM_THREADS'] = str(N_JOBS)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--platform', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='JamendoAudioFolder_npy')
    parser.add_argument('--data_limit', type=int, default=100)
    parser.add_argument('--subset', type=str, default='top50tags')
    parser.add_argument('--collate_fn', type=str, default='default', help="{seq2seq, default}")
    parser.add_argument('--trim_size', type=int, default='-1', help="[-1, inf], how many samples to read in audio")

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
    spec_group.add_argument('--max_length', type=int, default=44100,  ## for the specs??? Npot sure what it does
                            help="Desired length in samples for the input audio. Positive values will use a random crop of"
                                 "the audio file.")

    def ms_to_samples(ms, sampling_rate):
        return math.ceil(ms / 1000 * sampling_rate)


    config = parser.parse_args()
    config.hop_length = ms_to_samples(config.hop_length_ms, config.resampling_rate)


    experiment_description = "dset={}__" \
                             "batch={}__" \
                             "limit={}__" \
                             "workers={}__" \
                             "collate={}".format(config.dataset,
                                                 config.batch_size,
                                                 config.data_limit,
                                                 config.num_workers,
                                                 config.collate_fn)

    print("")
    print("======== Experiment =======")
    print("")
    print(config)
    config.result_path = get_result_dir_path(experiment_description)
    #compareTforms(config)
    analyzeDatasets(config)
    #main(config)
