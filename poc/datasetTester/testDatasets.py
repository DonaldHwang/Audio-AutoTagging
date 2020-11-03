#!/usr/bin/env python3'
from dataset import JamendoSpecFolder, JamendoSpecHDF5, JamendoSpecLMDB, JamendoSpecLMDBsubdir, \
    JamendoAudioFolder_audtorch, JamendoAudioFolder_torchaudio, JamendoAudioFolder_npy, Seq2Seq_short, \
    JamendoAudioFolder_torch
import torch
from torchvision import transforms
import torchvision.datasets as dset
import time
from matplotlib import pyplot as plt
import numpy as np
import argparse
import torchaudio
from torchaudio import transforms as tforms
from audtorch.collate import Seq2Seq
import audtorch.transforms as tforms2
import os
from psutil import cpu_count
import myTransfroms as myTforms
import gc


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

def get_Transforms():
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((96, 256), pad_if_needed=True, padding_mode='reflect'),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
        #transforms.Normalize([-48.78], [19.78])
    ])

    return train_transforms

def get_DataLoader(config):
    train_transforms = get_Transforms()

    if config.platform == 0:
        root = '/Volumes/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master'
    elif config.platform == 2:
        root = '/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master'
    elif config.platform == 3:
        root = '/m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master'

    subset = config.subset
    split = 0
    mode = 'train'


    if config.dataset == 'JamendoSpecFolder':
        dataset = JamendoSpecFolder(root,
                                    subset,
                                    split,
                                    mode,
                                    spec_folder='data/processed/spec_npy',
                                    transform=train_transforms)

    elif config.dataset == 'JamendoSpecHDF5':
        dataset = JamendoSpecHDF5(root,
                                  subset,
                                  split,
                                  mode,
                                  train_transforms,
                                  hdf5_filename='data/processed/jamendo.hdf5')
    elif config.dataset == 'JamendoSpecLMDB':
        dataset = JamendoSpecLMDB(root,
                                  subset,
                                  split,
                                  mode,
                                  train_transforms,
                                  lmdb_path='data/processed/triton')
    elif config.dataset == 'JamendoSpecLMDBsubdir':
        dataset = JamendoSpecLMDBsubdir(root,
                                  subset,
                                  split,
                                  mode,
                                  train_transforms,
                                  lmdb_path='data/processed/chunks')
    elif config.dataset == 'fake':
        dataset = dset.FakeData(image_size=(1, 96, 1366),
                                transform=transforms.Compose([
                                    transforms.RandomCrop((96, 256), pad_if_needed=True, padding_mode='reflect'),
                                    transforms.ToTensor()
                                ]))
    elif config.dataset == 'SVHN':
        dataset = dset.SVHN(root='/m/cs/work/falconr1/datasets/SVHN',
                            transform=transforms.Compose([
                                    transforms.RandomCrop((96, 256), pad_if_needed=True, padding_mode='reflect'),
                                    transforms.ToTensor()
                                ]),
                            download=True)
    elif config.dataset == 'JamendoAudioFolder_torchaudio':
        dataset = JamendoAudioFolder_torchaudio(root,
                                              subset,
                                              split,
                                              mode,
                                              transform=transforms.Compose([
                                                  tforms.MelSpectrogram(sr=44100,
                                                                        n_fft=512,
                                                                        ws=256,
                                                                        hop=256,
                                                                        f_min=20.0,
                                                                        f_max=8000,
                                                                        pad=0,
                                                                        n_mels=96),
                                                  transforms.ToPILImage(),
                                                  transforms.RandomCrop((96, 256), pad_if_needed=True, padding_mode='reflect'),
                                                  transforms.ToTensor(),
                                                ])
                                              )
    elif config.dataset == 'JamendoAudioFolder_audtorch':
        dataset = JamendoAudioFolder_audtorch(root,
                                              subset,
                                              split,
                                              mode,
                                              ## transform=tforms2.RandomCrop(size=256*44100),
                                              # transform=tforms2.Compose([
                                              #     tforms2.Downmix(1),
                                              #     tforms2.Normalize(),
                                              #     tforms2.Spectrogram(window_size=256,
                                              #                         hop_size=256,
                                              #                         fft_size=512),
                                              #     tforms2.Log(),
                                              #     # tforms2.LogSpectrogram(window_size=256,
                                              #     #                        hop_size=256,
                                              #     #                        normalize=True),
                                              #     myTforms.Debugger(),
                                              #     myTforms.CFL2FLC(),
                                              #     transforms.ToPILImage(),
                                              #     transforms.RandomCrop((96, 256), pad_if_needed=True, padding_mode='reflect'),
                                              #     transforms.ToTensor(),
                                              #   ])
                                              )
    elif config.dataset == 'JamendoAudioFolder_npy':
        dataset = JamendoAudioFolder_npy(root,
                                         subset,
                                         split,
                                         mode,
                                         trim_to_size=config.trim_size,
                                         ###transform=tforms2.Downmix(1),
                                         #transform=tforms2.RandomCrop(size=30*44100),
                                         # transform=tforms2.Compose([
                                         #     tforms2.Downmix(1),
                                         #     tforms2.Normalize(),
                                         #     tforms2.Spectrogram(window_size=256,
                                         #                         hop_size=256,
                                         #                         fft_size=512),
                                         #     tforms2.Log(),
                                         #     # tforms2.LogSpectrogram(window_size=256,
                                         #     #                        hop_size=256,
                                         #     #                        normalize=True),
                                         #     myTforms.Debugger(),
                                         #     myTforms.CFL2FLC(),
                                         #     transforms.ToPILImage(),
                                         #     transforms.RandomCrop((96, 256), pad_if_needed=True, padding_mode='reflect'),
                                         #     transforms.ToTensor(),
                                         #   ])
                                         )
    elif config.dataset == 'JamendoAudioFolder_torch':
        dataset = JamendoAudioFolder_torch(root,
                                         subset,
                                         split,
                                         mode,
                                         ###transform=tforms2.Downmix(1),
                                         transform=tforms2.RandomCrop(size=30*44100),
                                         # transform=tforms2.Compose([
                                         #     tforms2.Downmix(1),
                                         #     tforms2.Normalize(),
                                         #     tforms2.Spectrogram(window_size=256,
                                         #                         hop_size=256,
                                         #                         fft_size=512),
                                         #     tforms2.Log(),
                                         #     # tforms2.LogSpectrogram(window_size=256,
                                         #     #                        hop_size=256,
                                         #     #                        normalize=True),
                                         #     myTforms.Debugger(),
                                         #     myTforms.CFL2FLC(),
                                         #     transforms.ToPILImage(),
                                         #     transforms.RandomCrop((96, 256), pad_if_needed=True, padding_mode='reflect'),
                                         #     transforms.ToTensor(),
                                         #   ])
                                         )

    subset_indices = np.random.choice(range(len(dataset)), config.data_limit, replace=False)

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

# if running iwth python -m memory_profile
# @profile
def test_Vanilla(config):
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
                data, labels = sample

            step += 1
            t = time.time()
            t_step = t - last_time
            t_total += t_step
            last_time = t
            all_time.append(t_step)

            x = data.to(device)
            y = labels.to(device)

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
            j+1, num_epochs, t_epoch, t_total, batch_shape
        ))

        gc.collect()


    ## Plot time for each step
    t = np.linspace(0, num_epochs, num=len(all_time), endpoint=False)

    plt.figure(figsize=(16,8))
    plt.plot(t, all_time, 'r-+', label='Time per step')
    plt.legend()
    plt.savefig(os.path.join(config.result_path, 'Test_Output_time_per_step.png'))
    plt.xlabel('Epochs')
    plt.ylabel('Time (s)')
    plt.show()

    ## Plot time for each epoch
    t = np.linspace(0, num_epochs, num=len(all_time_epoch), endpoint=False)

    plt.figure(figsize=(16,8))
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
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--platform', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='JamendoAudioFolder_npy')
    parser.add_argument('--data_limit', type=int, default=100)
    parser.add_argument('--subset', type=str, default='top50tags')
    parser.add_argument('--collate_fn', type=str, default='default', help="{seq2seq, default}")
    parser.add_argument('--trim_size', type=int, default='-1', help="[-1, inf], how many samples to read in audio")

    config = parser.parse_args()

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
    test_Vanilla(config)
