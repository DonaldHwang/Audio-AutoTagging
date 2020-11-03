import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import math
import librosa
import librosa.display
import torch
import numpy as np
import os
from sklearn import metrics
from easydict import EasyDict as edict
from tqdm import tqdm
import warnings
import matplotlib


def make_module(cls):
    '''
    Forces a class to inherit from nn.Module.
    This is useful to be able to use custom transforms inside a nn.Sequential block.
    '''

    class ModuleChild(nn.Module, cls):
        def __init__(self, *args, **kwargs):
            nn.Module.__init__(self)
            cls.__init__(self, *args, **kwargs)

        def forward(self, signal):
            return cls.__call__(self, signal)

        def __repr__(self):
            return cls.__repr__(self)

    return ModuleChild


def show_spectrogram_batch(specs, config, gridSize=4,
                           saveFig=True, filepath=None, filename='Output_batch_spectrogram'):
    '''
    Provides 3 visualizations (mel or STFT spectrograms, time domain plots, and wave files) of a batch of
    samples and optionally saves them to disk :

    Args:
        specs - a torch.tensor of [batch, channels, freq_bins, frames]
        config - configuration dictionary
        dataPath - path to the audio files
        gridSize - number of plots, in case that the batch is too big
        filepath - path to save the image, if None, it will be saved to config.result_dir
        saveFig - flag for saving the plots to disk
        filename - name for the plot
    '''

    assert isinstance(specs, torch.Tensor), "Specs should be torch.tensor"

    batchSize = specs.shape[0]
    idx = np.random.choice(range(batchSize), gridSize, replace=False)

    assert math.sqrt(
        gridSize).is_integer(), "gridSize should be a squared number"  # For now, just to make easier plots

    ## Spectrograms
    fig = plt.figure(1, figsize=(32, 20))
    for j, i in enumerate(idx):
        plt.subplot(math.sqrt(gridSize), math.sqrt(gridSize), j + 1)
        tmp_spec = specs[i, :, :, :].detach().numpy().squeeze()
        if config.use_mels:
            librosa.display.specshow(tmp_spec, y_axis='mel', x_axis='time', sr=config.new_fs,
                                     hop_length=config.hop_length, fmax=config.fmax, fmin=config.fmin)
            plt.title('Normalized Mel spectrogram')
        else:
            librosa.display.specshow(tmp_spec, y_axis='log', x_axis='time', sr=config.new_fs,
                                     hop_length=config.hop_length)
            plt.title('STFT {foo:02d}'.format(foo=i))
        plt.colorbar(format='%+2.0f dB')

    if saveFig:
        filepath = config.result_dir if filepath is None else filepath
        plt.savefig(os.path.join(filepath, filename))
    plt.show()

    return fig


def show_spectrogram(spectrogram, config, fname='', use_stft_linear=True, saveFig=True,
                     filepath=None, filename='Output_single_spectrogram.png'):
    '''
    Plots the spectrogram (either mel or STFT) for a single sample and optionally saves it to disk.

    Args:
        spectrogram - 2d array for the spectrogram, either numpy array of torch tensor
        config - configuration dictionary
        fname - Filename for the plot title
        use_stft_linear - For STFT, whether to plot 'linear' or 'log' frequency axis
        saveFig - flag for saving the plots to disk
        filepath - path to the directory where the plot is saved
        filename - name for the plot
    '''

    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.numpy().squeeze()  # should be a (mel_bins, frames) after the squeeze

    fig = plt.figure(2, figsize=(16, 9))
    if config.use_mels:
        librosa.display.specshow(spectrogram,
                                 y_axis='mel',
                                 x_axis='time',
                                 sr=config.new_fs,
                                 hop_length=config.hop_length,
                                 fmax=config.fmax,
                                 fmin=config.fmin)
        plt.title('Normalized Mel spectrogram')
    else:
        x_axis = None
        title = None
        if use_stft_linear:
            x_axis = 'time'
            title = 'Normalized power STFT'
        else:
            x_axis = 'log'
            title = 'Normalized Log-frequency power STFT'

        librosa.display.specshow(spectrogram, y_axis='hz', x_axis=x_axis, sr=config.new_fs,
                                 hop_length=config.hop_length)
        plt.title(title + fname)
    plt.colorbar(format='%+2.0f dB')

    # if saveFig:
    #     matplotlib.use('agg', warn=False)
    #     plt.savefig(os.path.join(filepath, filename))
    # else:
    #     matplotlib.use('TkAgg', warn=False)
    #     plt.interactive(False)

    if saveFig:
        filepath = config.result_dir if filepath is None else filepath
        plt.savefig(os.path.join(filepath, filename))
    plt.show()

    return fig


def show_waveforms_batch(audios, fnames, config, grid_size=4, filepath='./results_debug', filename='Output_batch_%s'):
    '''
    Plots the waveforms of a batch of audios.

    Args:
    audios - list of audio tensors
    fnames - list of fnames
    config - configuration dictionary
    grid_size - number of plots, in case that the batch is too big
    filepath - path to the directory where the plot is saved
    filename - name for the plot
    '''

    assert math.sqrt(grid_size).is_integer(), "gridSize should be a squared number"  # For now, just to make easier plots


    ## Time domain plots
    fig =plt.figure(3)
    for j, i in enumerate(range(4)):
        plt.subplot(math.sqrt(grid_size), math.sqrt(grid_size), j + 1)
        tmp_audio = audios[i]
        tmp_audio = tmp_audio if isinstance(tmp_audio, np.ndarray) else tmp_audio.numpy()

        fname = fnames[i]
        librosa.display.waveplot(tmp_audio.squeeze(), sr=config.new_fs)
        plt.title('%d %s' % (j, fname.split('.')[0].split('/')[-1]))

    if filepath is not None:
        plt.savefig(os.path.join(filepath, filename % ('waveforms.png')))
    plt.show()

    return fig

def scores_to_class(pred_scores, class_threshold=0.5, multilabel=True):
    """
    Classifies a tensor of sigmoid or softmax results into a binary tensor of 0s and 1s.
    The classification can be either single label, or multi label.

    Args:
        pred_scores - (np.array or torch.Tensor) in format [batch, classes], or [batch, classes, frames] for
    the ground truth. This is the raw scores (e.g. sigmoid or softmax output).
    """

    if multilabel:
        pred_scores[pred_scores < class_threshold] = 0
        pred_scores[pred_scores >= class_threshold] = 1
    else:
        tmp = np.zeros_like(pred_scores)
        ids = np.argmax(pred_scores, axis=-1)
        tmp[np.arange(pred_scores.shape[0])[:, None],
            np.arange(pred_scores.shape[1])[None, :],
            ids] = 1
        pred_scores = tmp
    return pred_scores


def compute_evaluation_metrics(true_class, pred_scores, labels_list, filepath='.',
                               class_threshold=0.2, multilabel=False, verbose=True):
    """
    Computes evaluation metrics for classification tasks.
    This method supports muulti-class, and multi-label cases. And formats when there is 1 label per observation, or 1
    label per time step (frames) per observation (e.g. sound event detection).
    The raw scores for the prediction will be classified for some metrics. This means, the scores will be transformed
    to 0s or 1s.

    Args:
    - true_class - (np.array or torch.Tensor) in format [batch, classes] or [batch, classes, frames] for
    the ground truth. Should be 0 or 1.
    - pred_scores - (np.array or torch.Tensor) in format [batch, classes], or [batch, classes, frames] for
    the predicted class scores (logits). This is the raw scores (e.g. sigmoid or softmax output).
    - labels_list - List of labels names (strings).
    - filepath - Where to save the plot
    - multlabel - (Boolean) True if there can be many classes per observation.
    - verbose - (Boolean) True to print to console the scores and warnings.

    Returns:
         Dictionary with all the metrics.
    """

    score_accuracy = 0   # Overall for all frames and observations
    score_accuracy_observation = 0  # Per observation, regardless of class
    score_events_recall = 0  # Frames with events
    score_lwlrap = 0
    score_roc_auc_macro = 0
    score_pr_auc_macro = 0
    score_roc_auc_micro = 0
    score_pr_auc_micro = 0
    score_mse = 0
    score_roc_auc_per_label = np.zeros((len(labels_list), 1)).squeeze()  # Per label
    score_pr_auc_per_label = np.zeros((len(labels_list), 1)).squeeze()  # Per label

    score_precision = 0
    score_recall = 0
    score_f1 = 0
    score_support = 0

    # ===================================================
    # Validate inputs

    if isinstance(true_class, torch.Tensor):
        true_class = true_class.detach().cpu().numpy()
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.detach().cpu().numpy()

    assert true_class.shape == pred_scores.shape, "ERROR: The shape of true_class, and pred_scores must be the same."

    if true_class.shape[1] > true_class.shape[0]:
        msg = "WARNING: The number of classes appears to be larger than the number of observations. " \
              "Inputs should be [batch, classes, frames], or [batch, classes]. " \
              "\n Input shape = {}".format(true_class.shape)
        if verbose:
            print(msg)

    # Ground truth should have only 0s or 1s
    assert ((true_class == 0) | (true_class == 1)).all(), "ERROR: Ground truth labels should be 0 or 1"

    # Classify, from scores (sigmoid output) to 1s and 0s
    pred_class = scores_to_class(pred_scores, class_threshold=class_threshold, multilabel=multilabel)

    assert ((pred_class == 0) | (pred_class == 1)).all(), "ERROR: Predictions should be 0 or 1"

    if len(true_class.shape) > 2:  # Are inputs [batch, classes, frames], or [batch, classes]?
        if true_class.shape[1] == 1:
            true_class = true_class.squeeze()
            pred_scores = pred_scores.squeeze()
            pred_class = pred_class.squeeze()
        else:
            is_many_frames = True

            if true_class.shape[1] > true_class.shape[2]:
                msg = "WARNING: The number of classes appears to be larger than the number of frames. " \
                      "Inputs should be [batch, classes, frames], or [batch, classes]. " \
                      "\n Input shape = {}".format(true_class.shape)
                if verbose:
                    print(msg)

            # Permute dimensions, sklearn expects [batch, frames, class], but we have [batch, class, frames]
            # Here we treat frames as obs.
            tmp_true_class = np.transpose(true_class, (0, 2, 1))
            tmp_pred_scores = np.transpose(pred_scores, (0, 2, 1))
            tmp_pred_class = np.transpose(pred_class, (0, 2, 1))
    else:
        is_many_frames = False

    # ===================================================
    # Evaluate performance
    with warnings.catch_warnings(record=True):
        if not is_many_frames:
            try:
                score_precision, score_recall, score_f1, score_support = metrics.precision_recall_fscore_support(true_class,
                                                                                                                 pred_class)
                score_accuracy = np.mean(sum(np.equal(true_class, pred_class))  /
                                                np.array( [true_class.shape[0] for i in range(true_class.shape[1])]))
                score_accuracy_observation = metrics.accuracy_score(true_class, pred_class)
                score_lwlrap = metrics.label_ranking_average_precision_score(true_class, pred_scores)
                score_mse = math.sqrt(metrics.mean_squared_error(true_class, pred_scores))

                # Average precision is a single number used to approximate the integral of the PR curve
                # Macro is average over all classes, without considering class imbalances
                score_pr_auc_macro = metrics.average_precision_score(true_class, pred_scores, average="macro")
                score_roc_auc_macro = metrics.roc_auc_score(true_class, pred_scores, average="macro")
                # Micro, considers class imbalances
                score_pr_auc_micro = metrics.average_precision_score(true_class, pred_scores, average="micro")
                score_roc_auc_micro = metrics.roc_auc_score(true_class, pred_scores, average="micro")
            except ValueError as e:
                if verbose:
                    print("ERROR: Something went wrong with evaluation: ")
                    print(e)

        else:  # many frames
            # Accuracy is computed for all classes and frames
            # This is the % of frames,classes that match the ground truth, even if there is no sound event in a frame
            # If most frames are empty (no class), this value can be misleading
            tmp_ids = np.equal(true_class, pred_class)
            score_accuracy = np.count_nonzero(tmp_ids) / true_class.size

            # Count events
            # Find how many frames have non 0s
            event_frames_true = np.any(true_class > 0, axis=1)
            event_frames_pred = np.any(pred_class > 0, axis=1)

            total_events_true = np.sum(np.sum(event_frames_true))
            total_events_pred = np.sum(np.sum(event_frames_pred))

            # This the % of frames with sound events that were predicted, even if the class is wrong
            score_events_recall = total_events_pred / total_events_true

            # Iterate all observations
            for h in range(true_class.shape[0]):
                try:
                    tmp_score_precision, tmp_score_recall, tmp_score_f1, tmp_score_support = metrics.precision_recall_fscore_support(
                        tmp_true_class[h, :, :], tmp_pred_class[h, :, :])

                    score_precision += tmp_score_precision
                    score_recall += tmp_score_recall
                    score_f1 += tmp_score_f1
                    score_support += tmp_score_support.astype(np.float)

                    score_accuracy_observation += metrics.accuracy_score(tmp_true_class[h, :, :], tmp_pred_class[h, :, :])

                    score_lwlrap += metrics.label_ranking_average_precision_score(tmp_true_class[h, :, :], tmp_pred_scores[h, :, :])
                    score_mse += math.sqrt(metrics.mean_squared_error(tmp_true_class[h, :, :], tmp_pred_scores[h, :, :]))
                    score_pr_auc_macro += metrics.average_precision_score(tmp_true_class[h, :, :], tmp_pred_scores[h, :, :],
                                                                          average="macro")
                    score_roc_auc_macro += metrics.roc_auc_score(tmp_true_class[h, :, :], tmp_pred_scores[h, :, :], average="macro")
                    score_pr_auc_micro += metrics.average_precision_score(tmp_true_class[h, :, :], tmp_pred_scores[h, :, :],
                                                                          average="micro")
                    score_roc_auc_micro += metrics.roc_auc_score(tmp_true_class[h, :, :], tmp_pred_scores[h, :, :], average="micro")

                except ValueError as e:
                    if verbose:
                        print("ERROR: Something went wrong with evaluation:")
                        print(e)
                    break

            score_precision /= true_class.shape[0]
            score_recall /= true_class.shape[0]
            score_f1 /= true_class.shape[0]
            score_support /= true_class.shape[0]
            score_accuracy_observation /= true_class.shape[0]
            score_lwlrap /= true_class.shape[0]
            score_roc_auc_macro /= true_class.shape[0]
            score_pr_auc_macro /= true_class.shape[0]
            score_roc_auc_micro /= true_class.shape[0]
            score_pr_auc_micro /= true_class.shape[0]
            score_mse /= true_class.shape[0]

    # ===================================================
    # Print Metrics

    # How many events we predicted?
    if verbose:
        print("=============================================")
        print("Evaluation")
        print("")
        print("Frames with events recall = {:.4f} ".format(score_events_recall))
        print("Accuracy per observation  =  %f" % (score_accuracy_observation))
        print("Accuracy for all frames, classes  =  %f" % (score_accuracy))
        print("Label ranking average precision =  %f" % (score_lwlrap))
        print("ROC_AUC macro score  = %f" % (score_roc_auc_macro))
        print("PR_AUC macro score = %f" % (score_pr_auc_macro))
        print("ROC_AUC_micro score  = %f" % (score_roc_auc_micro))
        print("PR_AUC_micro score = %f" % (score_pr_auc_micro))
        print("MSE score for train = %f" % (score_mse))
        print("")

        if is_many_frames:
            print("-- Precision : {}".format(score_precision))
            print("-- Recall : {}".format(score_recall))
            print("-- Support : {}".format(score_support))
            print("-- F1 : {}".format(score_f1))
        else:
            print(metrics.classification_report(true_class, pred_class, target_names=labels_list))

    tmp = {
        'accuracy': score_accuracy,
        'accuracy_observation': score_accuracy_observation,
        'frames_events_recall': score_events_recall,
        'precision': score_precision,
        'recall': score_recall,
        'f1': score_f1,
        'support': score_support,
        'lwlrap': score_lwlrap,
        'mse': score_mse,
        'roc_auc_macro': score_roc_auc_macro,
        'pr_auc_macro': score_pr_auc_macro,
        'roc_auc_micro': score_roc_auc_micro,
        'pr_auc_micro': score_pr_auc_micro,
    }

    scores = edict(tmp)

    try:
        # Plot only a few of the scores
        aa = {'accuracy': scores.accuracy,
              'accuracy_observation': score_accuracy_observation,
              'lwlrap': scores.lwlrap,
              'mse': scores.mse,
              'roc_auc_macro': scores.roc_auc_macro,
              'roc_auc_micro': scores.roc_auc_micro,
              'pr_auc_macro': scores.pr_auc_macro,
              'pr_auc_micro': scores.pr_auc_micro}

        plt.figure(4, figsize=(16, 6))
        plt.bar(range(len(aa)), list(aa.values()), align='center')
        plt.xticks(range(len(aa)), list(aa.keys()))
        plt.ylabel('Scores')
        plt.ylim(0, 1)
        if filepath is not None:
            plt.savefig(os.path.join(filepath, 'Outputs_scores.png'))
        plt.show()
    except IndexError as e:
        print(e)

    return scores


def grad_norm(parameters, norm_type=2):
    '''
    Returns the total norm of all the gradients in a model.
    '''
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == 'inf':
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm


def compare_predictions(true_class, pred_scores, filepath='./results_debug'):
    '''
    Plots some results to compare them easily.
    :param true_class:
    :param pred_scores:
    :return:
    '''

    cols = true_class.shape[-1]
    #cols = true_class.shape[-1] if true_class.shape[0] < 50 else true_class.shape[-1]
    rows = 100 if true_class.shape[0] > 100 else true_class.shape[0]

    fig, ax = plt.subplots(1,2, num=5)
    ax[0].imshow(true_class[0:rows, 0:cols], interpolation='none', vmin=0, vmax=1, cmap='magma',
                 extent=[0, cols, rows, 0])
    ax[0].set_title('Ground truth')
    ax[1].imshow(pred_scores[0:rows, 0:cols], interpolation='none', vmin=0, vmax=1, cmap='magma',
                 extent=[0, cols, rows, 0])
    ax[1].set_title('Prediction scores')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, 'Outputs_comparison.png'))
    plt.show()

    return fig


class OneHot:
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, x):
        """
        One-hot encoding
        :param x: Tensor (channels, timestep)
        :return: Tensor (channels, timestep, depth)
        """
        return torch.nn.functional.one_hot(torch.from_numpy(np.array([x])),
                                           num_classes=self.depth).squeeze().float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


def get_class_histograms(dataset_train: torch.utils.data.dataset.Subset,
                         dataset_valid: torch.utils.data.dataset.Subset,
                         dataset_test: torch.utils.data.dataset.Subset,
                         one_hot_encoder=None,
                         data_limit=500,
                         filepath='./results_debug'):
    '''
    Analyze a dataset and get a class histogram.
    :param dataset:
    :return:
    '''

    if dataset_valid is None:
        dataset_valid = dataset_train
    if dataset_test is None:
        dataset_test = dataset_valid

    if data_limit is None:
        data_limit = len(dataset_train.dataset)

    print("")
    print("Getting class histograms.")
    ctr = 0
    targets_train = np.zeros((min(len(dataset_train), data_limit), dataset_train.dataset.num_classes))
    print("Train data")
    for (_, targets, _) in tqdm(dataset_train):
        ctr += 1
        if ctr >= data_limit: break
        tmp = targets.unsqueeze(0).numpy() if one_hot_encoder is None else one_hot_encoder(targets.unsqueeze(0).numpy()).depth
        #targets_train = np.append(targets_train, tmp, axis=0)
        targets_train[ctr-1, :] = tmp

    ctr = 0
    targets_valid = np.zeros((min(len(dataset_valid), data_limit), dataset_valid.dataset.num_classes))
    print("")
    print("Valid data")
    for (_, targets, _) in tqdm(dataset_valid):
        ctr += 1
        if ctr >= data_limit: break
        tmp = targets.unsqueeze(0).numpy() if one_hot_encoder is None else one_hot_encoder(targets.unsqueeze(0).numpy()).depth
        #targets_valid = np.append(targets_valid, tmp, axis=0)
        targets_valid[ctr-1, :] = tmp

    ctr = 0
    targets_test = np.zeros((min(len(dataset_test), data_limit), dataset_test.dataset.num_classes))
    print("")
    print("Test data")
    for (_, targets, _) in tqdm(dataset_test):
        ctr += 1
        if ctr >= data_limit: break
        tmp = targets.unsqueeze(0).numpy() if one_hot_encoder is None else one_hot_encoder(targets.unsqueeze(0).numpy()).depth
        #targets_test = np.append(targets_test, tmp, axis=0)
        targets_test[ctr-1, :] = tmp

    bins = dataset_train.dataset.num_classes

    fig, ax = plt.subplots(3,1, num=6)
    tmp = np.sum(targets_train, axis=0)
    if np.any(tmp == 0): print('WARNING: There are {} classes with no samples in the train data.'.format(
        np.sum(tmp == 0)))
    ax[0].bar(np.arange(bins), tmp)
    ax[0].set_title('Train Classes')

    tmp = np.sum(targets_valid, axis=0)
    if np.any(tmp == 0): print('WARNING: There are {} classes with no samples in the valid data.'.format(
        np.sum(tmp == 0)))
    ax[1].bar(np.arange(bins), tmp)
    ax[1].set_title('Valid Classes')

    tmp = np.sum(targets_test, axis=0)
    if np.any(tmp == 0): print('WARNING: There are {} classes with no samples in the test data.'.format(
        np.sum(tmp == 0)))
    ax[2].bar(np.arange(bins), tmp)
    ax[2].set_title('Test Classes')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, 'Inputs_classes.png'))
    plt.show()
    return fig


def get_spectrogram_histogram(spectrograms: list):
    '''
    Analyze one or more spectrograms and gets the histograms of values.

    Args:
        spectrograms: (list of tensors [freq_bins, frames])

    :return:
    '''

    if spectrograms[0].abs().max() > 1:
        my_range = [-50,50]
    else:
        my_range = [-1, 1]

    max_specs = 4
    coords = [(0,0), (0,1), (1,0), (1,1)]
    fig, ax = plt.subplots(2,2, num=7)
    for i, spec in enumerate(spectrograms):
        if i >= max_specs : break
        ax[coords[i]].hist(spec.squeeze().view(1,-1), bins=100, range=my_range)
    plt.show()
    return fig


def show_spectrogram_batch(specs, fnames, config, gridSize=4,
                           filepath='./results_debug', filename='Output_batch_spectrogram'):
    '''
    Provides 3 visualizations (mel or STFT spectrograms, time domain plots, and wave files) of a batch of
    samples and optionally saves them to disk :

    Args:
        specs - a list of spectrograms
        fnames -  a list of fnames
        config - configuration dictionary
        dataPath - path to the audio files
        gridSize - number of plots, in case that the batch is too big
        saveFig - flag for saving the plots to disk
        filepath - path to the directory where the plot is saved
        filename - name for the plot
    '''

    batchSize = len(specs)
    idx = np.random.choice(range(batchSize), gridSize, replace=False)

    assert math.sqrt(
        gridSize).is_integer(), "gridSize should be a squared number"  # For now, just to make easier plots

    ## Spectrograms
    fig = plt.figure(8)
    for j, i in enumerate(idx):
        plt.subplot(math.sqrt(gridSize), math.sqrt(gridSize), j + 1)
        tmp_spec = specs[i]
        if isinstance(tmp_spec, torch.Tensor):
            tmp_spec = tmp_spec.squeeze().detach().numpy()
        if config.use_mels:
            librosa.display.specshow(tmp_spec, y_axis='mel', x_axis='time', sr=config.new_fs,
                                     hop_length=config.hop_length, fmax=config.fmax, fmin=config.fmin)
            plt.title('Normalized Mel spectrogram')
        else:
            librosa.display.specshow(tmp_spec, y_axis='log', x_axis='time', sr=config.new_fs,
                                     hop_length=config.hop_length)
            plt.title('STFT {foo:02d}'.format(foo=i))
        plt.colorbar(format='%+2.0f dB')

    if filepath is not None:
        plt.savefig(os.path.join(filepath, filename))
    plt.show()
    return fig


class NoamScheduler:
    def __init__(self, optimizer, init_lr, warmup_steps=4000):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.warmup_steps = float(warmup_steps)
        self._step = 0

    def step(self):
        """
        Noam scheme from tensor2tensor
        https://github.com/tensorflow/tensor2tensor/issues/280#issuecomment-339110329
        """
        self._step += 1.
        lr = self.init_lr * self.warmup_steps ** 0.5 * np.minimum(self._step * self.warmup_steps ** -1.5, self._step ** -0.5)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)


