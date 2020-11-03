#!/usr/bin/env python3'
from dataset import JamendoSpecFolder, JamendoAudioFolder_npy, JamendoAudioFolder_torchaudio
import torch
from torch import nn
from torchvision import transforms as tforms_vision
import torchvision.datasets as dset
import time
from matplotlib import pyplot as plt
import numpy as np
import argparse
import torchaudio
from torchaudio import transforms as tforms_torch
from audtorch import datasets as dsets
from audtorch.collate import Seq2Seq
import audtorch.transforms as tforms_aud
import os
from psutil import cpu_count
import myTransfroms as tforms_mine
import gc
from ATPP import AudioTaggingPreProcessing, Util
import math
from enum import IntEnum
from itertools import cycle
from model import FCN, ConvBlock
from torchsummary import summary
from tqdm import tqdm
import warnings
import json
from easydict import EasyDict as edict
import seaborn as sns
sns.set()



def check_outputs(outputs, multilabel=False):
    """
    Validates that the outpus has valid values for all data points.
    In multiclass=False mode , the all columns should sum to 1 (softmax).
    In multiclass=True mode, all columns shouuld be in the range of 0 to 1 (sigmoid)
    """

    if torch.is_tensor(outputs):
        outputs = outputs.cpu().numpy()

    if np.any(np.isnan(outputs)):
        return False

    if not multilabel:
        # Check that the sum of all colums == 1, or close to it
        tmp = outputs.sum(axis=1)
        tmp2 = np.isclose(tmp, np.ones_like(tmp))
        return np.all(tmp2)
    else:
        # check that no column is negative or larger than 1, for all data samples
        tmp = np.invert(np.logical_or(outputs < 0, outputs > 1))
        return np.all(tmp)


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


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def str2bool(v):
    """
    Converts a string value to boolean.
    :param v:
    :return:
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class TformsSet(IntEnum):
    '''
    Enum for sets of transformations (augmentaqtion methods).

    https://stackoverflow.com/questions/43968006/support-for-enum-arguments-in-argparse
    '''
    TorchAudio = 0
    Audtorch = 1
    MySet = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return TformsSet[s]
        except KeyError:
            raise ValueError()


def get_train_transforms(config: object, transforms_set: TformsSet = TformsSet.Audtorch) -> object:
    if config.use_mels:
        if transforms_set == TformsSet.TorchAudio:
            trans = tforms_vision.Compose([
                tforms_torch.Resample(orig_freq=44100, new_freq=config.resampling_rate),
                tforms_torch.MelSpectrogram(sample_rate=config.resampling_rate,
                                      n_fft=config.n_fft,
                                      win_length=config.hop_length,
                                      hop_length=config.hop_length,
                                      f_min=float(config.fmin),
                                      f_max=float(config.fmax),
                                      pad=0,
                                      n_mels=config.n_mels
                                      ),
                tforms_torch.AmplitudeToDB(stype='power', top_db=80),
                #tforms_aud.RandomCrop(config.max_length_frames),  # Raises "Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead."

            ])
        elif transforms_set == TformsSet.MySet:  # this works
            trans = tforms_aud.Compose([
                tforms_torch.Resample(orig_freq=44100, new_freq=config.resampling_rate),
                tforms_mine.Spectrogram(config),
                tforms_aud.RandomCrop(config.max_length_frames)
            ])
    else:
        if transforms_set == TformsSet.TorchAudio:  # this works
            trans = tforms_aud.Compose([
                                         tforms_torch.Resample(orig_freq=44100, new_freq=config.resampling_rate),

                                         tforms_torch.Spectrogram(n_fft=config.n_fft,
                                                                  win_length=config.hop_length,
                                                                  hop_length=config.hop_length,
                                                                  pad=0,
                                                                  power=2,
                                                                  normalized=True),
                                         tforms_torch.AmplitudeToDB(stype='power', top_db=80),
                                         tforms_aud.RandomCrop(config.max_length_frames)
                                     ])
        elif transforms_set == TformsSet.MySet:  # this works
            trans = tforms_aud.Compose([
                                    tforms_torch.Resample(orig_freq=44100, new_freq=config.resampling_rate),
                                    tforms_mine.Spectrogram(config),
                                    tforms_aud.RandomCrop(config.max_length_frames)
                                    ])
    return trans


def get_train_transforms_audio_only(config: object, transforms_set: TformsSet = TformsSet.Audtorch) -> object:
    trans = tforms_torch.Resample(orig_freq=44100, new_freq=config.resampling_rate)

    return trans


def get_dataloader(config, transforms : TformsSet = TformsSet.Audtorch):
    if config.platform == 0:
        root = '/Volumes/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master'
    elif config.platform == 2:
        root = '/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master'
    elif config.platform == 3:
        root = '/m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master'


    if config.dataset == 'JamendoAudioFolder_npy':
        dataset_train = JamendoAudioFolder_npy(root,
                                         config.subset,
                                         config.split,
                                         mode='train',
                                         trim_to_size=config.trim_size,
                                         return_fname=True,
                                         transform=get_train_transforms_audio_only(config, transforms)
                                         )
        dataset_test = JamendoAudioFolder_npy(root,
                                               config.subset,
                                               config.split,
                                               mode='test',
                                               trim_to_size=config.trim_size,
                                               return_fname=True,
                                               transform=get_train_transforms_audio_only(config, transforms)
                                               )
    elif config.dataset == 'JamendoAudioFolder_torchaudio':
        dataset_train = JamendoAudioFolder_torchaudio(root,
                                         config.subset,
                                         config.split,
                                         mode='train',
                                         trim_to_size=config.trim_size,
                                         return_fname=True,
                                         normalize=True,
                                         transform=get_train_transforms_audio_only(config, transforms)
                                         )
        dataset_test = JamendoAudioFolder_torchaudio(root,
                                                      config.subset,
                                                      config.split,
                                                      mode='train',
                                                      trim_to_size=config.trim_size,
                                                      return_fname=True,
                                                      normalize=True,
                                                      transform=get_train_transforms_audio_only(config, transforms)
                                                      )

    subset_indices_train = np.random.choice(range(len(dataset_train)), config.data_limit, replace=False)
    subset_indices_test = np.random.choice(range(len(dataset_test)), min(config.data_limit, len(dataset_test)),
                                            replace=False)


    print('------ Train dataset length = {}, using {} samples.'.format(len(dataset_train), len(subset_indices_train)))
    print('------ Test dataset length = {}, using {} samples.'.format(len(dataset_test), len(subset_indices_test)))

    if config.collate_fn == 'seq2seq':
        collate = Seq2Seq([-1,-1], batch_first=None, sort_sequences=False)
        #collate = Seq2Seq_short([-1, -1], batch_first=None, sort_sequences=False)
    else:
        collate = torch.utils.data.dataloader.default_collate

    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                                             batch_size=config.batch_size,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices_train),
                                             collate_fn=collate,
                                             drop_last=True,
                                             )

    test_dataloader = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=config.batch_size,
                                                   # shuffle=True,
                                                   num_workers=config.num_workers,
                                                   pin_memory=True,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices_test),
                                                   collate_fn=collate,
                                                   drop_last=True,
                                                   )

    return train_dataloader, test_dataloader


def make_module(cls):
    class ModuleChild(nn.Module, cls):
        def __init__(self, *args, **kwargs):
            nn.Module.__init__(self)
            cls.__init__(self, *args, **kwargs)

        def forward(self, signal):
            return cls.__call__(self, signal)

        def __repr__(self):
            return cls.__repr__(self)

    return ModuleChild


def get_time_frequency_transform(config):
    """
    Returns a nn.Sequeutnial bock to do a time-frequency transform, and crop to the desired size.
    :param config:
    :return:
    """
    if config.tforms == TformsSet.TorchAudio:
        if config.use_mels:
            transformer = nn.Sequential(
                #tforms_torch.Resample(orig_freq=44100, new_freq=config.resampling_rate),  # raises "RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same"
                tforms_torch.MelSpectrogram(sample_rate=config.resampling_rate,
                                            n_fft=config.n_fft,
                                            win_length=config.hop_length,
                                            hop_length=config.hop_length,
                                            f_min=float(config.fmin),
                                            f_max=float(config.fmax),
                                            pad=0,
                                            n_mels=config.n_mels
                                            ),
                make_module(tforms_mine.RandomCrop)(config.max_length_frames),
                tforms_torch.AmplitudeToDB(stype='power', top_db=80),
                #tforms_mine.AmplitudeToDB(stype='power', top_db=80),

                #make_module(tforms_mine.ToTensor)(),
                #make_module(tforms_aud.RandomCrop)(config.max_length_frames) # raises Can't call numpy() on Variable that requires grad. Use var.detach().numpy()
            )
        else:
            transformer = nn.Sequential(
                tforms_torch.Spectrogram(n_fft=config.n_fft,
                                         win_length=config.hop_length,
                                         hop_length=config.hop_length,
                                         pad=0,
                                         power=2,
                                         normalized=True),
                tforms_mine.AmplitudeToDB(stype='power', top_db=80),
                make_module(tforms_aud.RandomCrop)(config.max_length_frames)
            )
    elif config.tforms == TformsSet.Audtorch:
        if config.use_mels:
            transformer = nn.Sequential(
                make_module(tforms_aud.Spectrogram)(window_size=config.hop_length,
                                     hop_size=config.hop_length,
                                     fft_size=config.n_fft,),
                make_module(tforms_aud.RandomCrop)(config.max_length_frames),
                make_module(tforms_mine.SpecToMelSpec)(config),  # Returns (power) mel spectrogram
                make_module(tforms_mine.MagnitudeToDb)(stype='power', ref='np.max'),  # Normalizes by max of each spec
                make_module(tforms_mine.ToTensor)(),
                ####tforms_torch.AmplitudeToDB(stype='power', top_db=80),  # does not work
            )
        else:
            transformer = nn.Sequential(
                make_module(tforms_aud.Spectrogram)(window_size=config.hop_length,
                                                    hop_size=config.hop_length,
                                                    fft_size=config.n_fft, ),  # returns energy
                make_module(tforms_aud.RandomCrop)(config.max_length_frames),
                make_module(tforms_mine.MagnitudeToDb)(stype='magnitude', ref='np.max'),  # magnitude, because STFT is energy
                make_module(tforms_mine.ToTensor)(),
                # tforms_torch.AmplitudeToDB(stype='magnitude', top_db=80),  # kinda works, mag and power, but still not shure
            )

    return transformer


def do_time_frequency_transform(data, transformer, config):
    """
    Applies the time-frequency transform to some data. The difference is that the TorchAudio based one does not support
    batches, so it has to be run in a for loop. The transformer is a nn.Sequential block.

    :param data:
    :param transformer:
    :param config:
    :return:
    """
    if config.tforms == TformsSet.TorchAudio:
        tmpList = []
        for i in range(data.shape[0]):
            tmp = data[i,:,:]
            tmpList.append(transformer(tmp))

        # Stack adds a dim, so the shape is (batch, channel, freq_bins, frames) after doing STFT
        out = torch.stack(tmpList, dim=0)
    else:
        out = transformer(data)

    return out

def main(config):
    ## ================ Load data =============================================================
    dataloader_train, dataloader_test = get_dataloader(config, config.tforms)

    data, labels, fname, raw_audio = next(iter(dataloader_train))

    tn = tforms_mine.ApplyReverb(config.resampling_rate)

    tmp = tn(data[0])



    print("Data shape pre module = %s" % str(data.shape))

    transformer = get_time_frequency_transform(config)

    out = do_time_frequency_transform(data, transformer, config)

    print("Data shape after transformer = %s" % str(out.shape))


    # Visualize some of the data
    tmp = out

    tmp2 = {'spectrogram': tmp,
            'labels': labels,
            'fname': fname}

    AudioTaggingPreProcessing.show_spectrogram_batch(tmp2, config, filepath=config.result_path)

    tmp3 = {'audio': raw_audio,
            'labels': labels,
            'fname': fname}

    AudioTaggingPreProcessing.show_waveforms_from_audio_batch(tmp3, config, filepath=config.result_path, saveWavs=False)


    ## ================ Model =================================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # input shape (batch, 1, 128, 1366) ss
    model = FCN(ConvBlock, output_shape=dataloader_train.dataset.num_classes,
               max_pool=[(2,4), (2,4), (2,4), (3,5), (4,4)]).to(device)

    summary(model, input_size=(1, 128, 1366))

    # input shape (batch, 1, 128, 256)
    # model = FCN(ConvBlock, output_shape=dataloader_train.dataset.num_classes,
    #             max_pool=[(2, 2), (2, 2), (2, 4), (3, 3), (4, 4)],
    #             filters_num=[64, 128, 256, 512, 1024]).to(device)
    #
    # summary(model, input_size=(1, 128, 256))

    # Loss and optimizer
    learning_rate = 0.001
    criterion = model.loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Another hacky way to get the iterator for the validation set
    def loopy(dl):
        while True:
            for x in iter(dl): yield x

    ## ================ Training loop =========================================================
    t_sum = 0
    t_epoch = 0
    last_time = time.time()
    last_epoch = time.time()
    all_time = []
    all_time_epoch = []

    total_step = len(dataloader_train)
    loss_history_train = []

    train_loss_step = 0.0
    valid_loss_step = 0.0
    train_loss_epoch = 0.0
    valid_loss_epoch = 0.0
    train_loss_history = []
    valid_loss_history = []

    train_steps = 0
    valid_steps = 0

    myIter = loopy(dataloader_test)

    for epoch in range(config.num_epochs):
        train_loss_epoch = 0.0
        valid_loss_epoch = 0.0
        train_loss_step = 0.0
        valid_loss_step = 0.0

        # Training loop
        pbar = tqdm(enumerate(dataloader_train))
        for i, sample in pbar:
            t = time.time()
            t_diff = t - last_time
            t_sum += t_diff
            last_time = t
            all_time.append(t_diff)

            audio = sample[0]  # Sample is (transformed_audio, labels, fname, raw_audio)
            labels = sample[1].to(device)

            # Forward pass
            if config.tforms == TformsSet.TorchAudio:  # Torchaudio supports full GPU
                audio = audio.to(device)
                specs = do_time_frequency_transform(audio, transformer.to(device), config) # time-freq transform
                outputs = model(specs)
            else:  # Audtorch uses numpy, so no support for GPU
                specs = do_time_frequency_transform(audio, transformer, config)  # time-freq transform
                outputs = model(specs.to(device))

            loss = criterion(outputs, labels)

            train_loss_epoch += loss.item()
            train_loss_step = loss.item()

            train_steps += 1
            train_loss_history.append(train_loss_step)  # Append losses for plotting

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_steps % config.print_every == 0:
                model.eval()

                with torch.no_grad():
                    # Hacky way to get a single validation batch, see above for comments.
                    try:
                        sample = next(myIter)
                    except StopIteration:
                        myIter = loopy(dataloader_test)
                        sample = next(myIter)

                    audio = sample[0]
                    labels = sample[1].to(device)

                    # Forward pass | time-freq transform
                    if config.tforms == TformsSet.TorchAudio:  # Torchaudio supports full GPU
                        specs = do_time_frequency_transform(audio.to(device), transformer.to(device), config)
                    else:  # Audtorch uses numpy, so no support for GPU
                        specs = do_time_frequency_transform(audio, transformer, config).to(device)

                    outputs = model(specs)
                    loss = criterion(outputs, labels)

                    valid_loss_step = loss.item()
                    valid_loss_epoch += valid_loss_step

                    valid_steps += 1
                    valid_loss_history.append(valid_loss_step)  # Append losses for plotting

            pbar.set_description("Epoch [{}/{}], Step [{}/{}],   T_diff {} , T_total {} , train_Loss: {:.4f} , valid_Loss {:.4f}"
                                 .format(epoch + 1, config.num_epochs,
                                         i + 1, total_step,
                                         t_diff, t_sum,
                                         train_loss_step, valid_loss_step))

        t = time.time()
        t_epoch = t - last_epoch
        last_epoch = t
        all_time_epoch.append(t_epoch)
        print("--------- Epoch [{}/{}] Summary , time per epoch {} , train_loss {} , valid_loss {}"
              .format(epoch + 1, config.num_epochs, t_epoch,
                      train_loss_epoch / len(dataloader_train), valid_loss_epoch / max(valid_steps, 1)))

    ## ================ Plot training =========================================================
    ## Plot time for each step
    t = np.linspace(0, config.num_epochs, num=len(all_time), endpoint=False)

    plt.figure(figsize=(16,8))
    plt.plot(t, all_time, 'r-+', label='Time per step')
    plt.legend()
    plt.savefig(os.path.join(config.result_path, 'Output_time_per_step.png'))
    plt.xlabel('Epochs')
    plt.ylabel('Time (s)')
    plt.title("Train data = %d" %(len(dataloader_train)))
    plt.show()

    ## Plot time per epoch
    t = np.linspace(0, config.num_epochs, num=len(all_time_epoch), endpoint=False)

    plt.figure(figsize=(16,8))
    plt.plot(t, all_time_epoch, 'r-+', label='Time per step')
    plt.legend()
    plt.savefig(os.path.join(config.result_path, 'Output_time_per_epoch.png'))
    plt.xlabel('Epochs')
    plt.ylabel('Time (s)')
    plt.title("Train data = %d" %(len(dataloader_train)))
    plt.show()


    ## Plot loss
    t = np.linspace(0, config.num_epochs, num=train_steps, endpoint=False)
    t_valid = np.linspace(0, config.num_epochs, num=valid_steps, endpoint=False)

    plt.figure(figsize=(20,8))
    plt.plot(t, train_loss_history, 'r-+', label='Train')
    plt.plot(t_valid, valid_loss_history, 'b--o', label='Valid')
    plt.legend()
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.title("Train data = %d" %(len(dataloader_train)))
    plt.savefig(os.path.join(config.result_path, 'Output_loss.png'))
    plt.show()

    ## ================ Evaluation ============================================================
    def get_auc(y_true, y_preds, labels_list):
        from sklearn import metrics
        score_accuracy = 0
        score_lwlrap = 0
        score_roc_auc_macro = 0
        score_pr_auc_macro = 0
        score_roc_auc_micro = 0
        score_pr_auc_micro = 0
        score_mse = 0
        score_roc_auc_all = np.zeros((len(labels_list), 1)).squeeze()
        score_pr_auc_all = np.zeros((len(labels_list), 1)).squeeze()
        try:
            # for accuracy, lets pretend this is a single label problem, so assign only one label to every prediction
            score_accuracy = metrics.accuracy_score(y_true, indices_to_one_hot(y_preds.argmax(axis=1), dataloader_train.dataset.num_classes))
            score_lwlrap = metrics.label_ranking_average_precision_score(y_true, y_preds)
            score_mse = math.sqrt(metrics.mean_squared_error(y_true, y_preds))
            # Average precision is a single number used to approximate the integral of the PR curve
            # Macro is average over all clases, without considering class imbalances
            score_pr_auc_macro = metrics.average_precision_score(y_true, y_preds, average="macro")
            score_roc_auc_macro = metrics.roc_auc_score(y_true, y_preds, average="macro")
            # Micro, considers class imbalances
            score_pr_auc_micro = metrics.average_precision_score(y_true, y_preds, average="micro")
            score_roc_auc_micro = metrics.roc_auc_score(y_true, y_preds, average="micro")

        except ValueError as e:
            print("Soemthing wrong with evaluation")
            print(e)
        print("Accuracy =  %f" % (score_accuracy))
        print("Label ranking average precision =  %f" % (score_lwlrap))
        print("ROC_AUC macro score  = %f" % (score_roc_auc_macro))
        print("PR_AUC macro score = %f" % (score_pr_auc_macro))
        print("ROC_AUC_micro score  = %f" % (score_roc_auc_micro))
        print("PR_AUC_micro score = %f" % (score_pr_auc_micro))
        print("MSE score for train = %f" % (score_mse))

        # These are per tag
        try:
            score_roc_auc_all = metrics.roc_auc_score(y_true, y_preds, average=None)
            score_pr_auc_all = metrics.average_precision_score(y_true, y_preds, average=None)
        except ValueError as e:
            print("Something wrong with evaluation")
            print(e)

        print("")
        print("Per tag, roc_auc, pr_auc")
        for i in range(len(labels_list)):
            print('%s \t\t\t\t\t\t %.4f \t%.4f' % (labels_list[i], score_roc_auc_all[i], score_pr_auc_all[i]))


        tmp = {
            'accuracy': score_accuracy,
            'lwlrap': score_lwlrap,
            'mse': score_mse,
            'roc_auc_macro': score_roc_auc_macro,
            'pr_auc_macro': score_pr_auc_macro,
            'roc_auc_micro': score_roc_auc_micro,
            'pr_auc_micro': score_pr_auc_micro,
            'roc_auc_all': score_roc_auc_all,
            'pr_auc_all': score_pr_auc_all
        }

        scores = edict(tmp)

        try:
            plt.figure(figsize=(20, 8))
            plt.bar(np.arange(len(labels_list)), scores.roc_auc_all[:])
            plt.ylabel('ROC_AUC')
            plt.title("Train data = %d" % (len(dataloader_train)))
            plt.savefig(os.path.join(config.result_path, 'Output_scores_per_label.png'))
            plt.show()

            # Plot only a few of the scores
            aa = {'accuracy': scores.accuracy,
                  'lwlrap': scores.lwlrap,
                  'mse': scores.mse,
                  'roc_auc_macro': scores.roc_auc_macro,
                  'roc_auc_micro': scores.roc_auc_micro,
                  'pr_auc_macro': scores.pr_auc_macro,
                  'pr_auc_micro': scores.pr_auc_micro}

            plt.figure(figsize=(10, 6))
            plt.bar(range(len(aa)), list(aa.values()), align='center')
            plt.xticks(range(len(aa)), list(aa.keys()))
            plt.ylabel('Scores')
            plt.savefig(os.path.join(config.result_path, 'Outputs_scores.png'))
            plt.show()
        except IndexError as e:
            print(e)

        return scores

    del myIter

    model.eval()
    with torch.no_grad():
        total = 0
        numBatches = 0

        allPreds = np.empty((0, dataloader_test.dataset.num_classes), float)
        allLabels = np.empty((0, dataloader_test.dataset.num_classes), int)

        for batch in dataloader_test:
            audio = batch[0]
            labels = batch[1].cpu().numpy()

            # Forward pass | time-freq transform
            if config.tforms == TformsSet.TorchAudio:  # Torchaudio supports full GPU
                specs = do_time_frequency_transform(audio.to(device), transformer.to(device), config)
            else:  # Audtorch uses numpy, so no support for GPU
                specs = do_time_frequency_transform(audio, transformer, config).to(device)

            outputs = model.forward_with_evaluation(specs.to(device)).cpu().numpy()

            if not check_outputs(outputs, True):
                warnings.warn("Warning, the ouputs appear to have wrong values!!!!!")

            allPreds = np.append(allPreds, outputs, axis=0)
            allLabels = np.append(allLabels, labels, axis=0)

            total += labels.shape[0]
            numBatches += 1

    print("Evaluated on {} validation batches".format(numBatches))

    scores = get_auc(allLabels, allPreds, dataloader_test.dataset.taglist)

    # Save scores to disk
    import pickle

    with open(os.path.join(config.result_path, 'Scores.pkl')) as f:
        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)

    return

    # Load the results like this:
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # This is copied from Seyoung's code. I am not sure what it does.
    N_JOBS = cpu_count()
    os.environ['MKL_NUM_THREADS'] = str(N_JOBS)
    os.environ['OMP_NUM_THREADS'] = str(N_JOBS)

    secs = 44100 * 35
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--platform', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='JamendoAudioFolder_npy')
    #parser.add_argument('--tforms', type=lambda tform: TformsSet[tform], default='TorchAudio', choices=list(TformsSet))
    parser.add_argument('--tforms', type=TformsSet.from_string, default='TorchAudio', choices=list(TformsSet))
    parser.add_argument('--data_limit', type=int, default=100)
    parser.add_argument('--subset', type=str, default='top50tags')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--collate_fn', type=str, default='default', help="{seq2seq, default}")
    parser.add_argument('--trim_size', type=int, default=secs, help="[-1, inf], how many samples to read in audio before transformations")

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
    spec_group.add_argument('--n_mels', type=int, default=128,
                            help="Number of mel bins.")
    spec_group.add_argument('--n_fft', type=int, default=2048,
                            help="FFT size.")
    spec_group.add_argument('--use_mels', type=str2bool, default=False,
                            help="Use mel filters.")
    spec_group.add_argument('--max_length_frames', type=int, default=256,  ## for the specs
                            help="Desired length in frames for the specs. Positive values will use a random crop of"
                                 "the spectrogram file.")

    def ms_to_samples(ms, sampling_rate):
        return math.ceil(ms / 1000 * sampling_rate)


    config = parser.parse_args()
    config.hop_length = ms_to_samples(config.hop_length_ms, config.resampling_rate)

    experiment_description = "dset={}__" \
                             "batch={}__" \
                             "limit={}__" \
                             "tforms={}__" \
                             "collate={}".format(config.dataset,
                                                 config.batch_size,
                                                 config.data_limit,
                                                 config.tforms,
                                                 config.collate_fn)

    config.result_path = get_result_dir_path(experiment_description)
    #config.result_path = './results'

    print("")
    print("======== Experiment =======")
    print("")

    # Print the experiment config
    ctr = 0
    for k, v in vars(config).items():
        ctr += 1
        if ctr % 10 == 0: print(' ')
        print('{} \t {}'.format(k.ljust(15, ' '), v))
    print("")

    # Save the experiment config to JSON
    with open(os.path.join(config.result_path, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    # Do stuff
    main(config)
