#!/usr/bin/env python3
## #########################################################################
## Audio Tagging Pre Processing ATPP
## ########################################################################
import librosa
import math
import librosa.display
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

class AudioTaggingPreProcessing(object):
    __melBank__ = None

    @staticmethod
    def get_Mels(resampling_rate, n_fft, n_mels, fmin, fmax):
        '''
        returns a mel fitler bank, used to compute the mel spectrograms
        '''

        if AudioTaggingPreProcessing.__melBank__ is None:
            AudioTaggingPreProcessing.__melBank__ = librosa.filters.mel(sr=resampling_rate,
                                                                        n_fft=n_fft,
                                                                        n_mels=n_mels,
                                                                        fmin=fmin,
                                                                        fmax=fmax,
                                                                        htk=False,
                                                                        norm=1)

        return AudioTaggingPreProcessing.__melBank__

    @staticmethod
    def read_audio(pathname, resampling_rate, max_length, fixedLength=False):
        '''
        Loads a single audio file from disk and resamples, (optionally) fixes the length either by trimming or
        padding with silence, and normalizes it.

        Params:
            config - configuration dictionary
            pathname - full path to the audio file
            fixedLength - flag to fix length of audio file to the value determined in the configuraiton

        Returns:
            audio - np array of audio
        '''
        audio, sr = librosa.load(pathname, mono=True)

        # Resampling
        if resampling_rate != sr:
            audio = librosa.resample(audio, sr, resampling_rate)

        # Fix length
        if fixedLength:
            if audio.shape[0] < max_length:  # if too short, repeat many times
                times = math.floor(max_length / audio.shape[0])

                tmpAudio = np.array([])
                for i in range(times):
                    tmpAudio = np.append(tmpAudio, audio)

                if tmpAudio.shape[0] < max_length:  # Pad with silence on both ends
                    pad_length = max_length - tmpAudio.shape[0]
                    tmpAudio = np.pad(tmpAudio,
                                   pad_width=(math.floor(pad_length / 2), math.ceil(pad_length / 2)),
                                   mode="constant")

                audio = tmpAudio

            else:  # too big, grab random portion of the audio file
                res = audio.shape[0] - max_length
                start = math.floor(np.random.rand() * res)
                audio = audio[start : start + max_length]

        # Normalize audio
        audio = librosa.util.normalize(audio)
        return audio

    @staticmethod
    def audio_to_spectrogram(audio, use_mels, resampling_rate, hop_length, n_fft, n_mels, fmin, fmax):
        '''
        Computes the spectrogram (mel of STFT) of a previously loaded audio signal.

        Params:
            condig - Configuration dictionary.
            audio - audio signal (as np array)
        '''
        if use_mels:
            spec = librosa.core.stft(audio,
                                     hop_length=hop_length,
                                     win_length=hop_length,
                                     n_fft=n_fft,
                                     center=False
                                     )

            mel = AudioTaggingPreProcessing.get_Mels(resampling_rate, n_fft, n_mels, fmin, fmax)  # Mels are stored as static variable to speed up the process

            # Manually compute the mel spectrogram,
            # The mel spectrogram is the dot product of the mel matrix (shape n_mels x stft_bins)
            # and the power spectrogram of the audio signal
            # The power spectrogram is the squared magnitude spectrogram
            # power = np.abs(spec) ** 2

            # NOTE:
            # This spectrogram is normalized by the max value of each spectrogram, so the maximum value is always 0 db
            spec = librosa.power_to_db(mel.dot(np.abs(spec) ** 2), ref=np.max)
        else:
            spec = librosa.core.stft(audio,
                                     hop_length=hop_length,
                                     win_length=hop_length,
                                     n_fft=n_fft,
                                     center=False
                                     )

            # Spectrogram returns complex values (amplitude + phase)
            # abs to get magnitude
            # then we go to power and show in db
            # where the ref is the maximum value in each file
            spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)

        spec = spec.astype(np.float32)
        return spec


    @staticmethod
    def show_spectrogram(spectrogram, config, saveFig=True, filepath='', filename='Output_single_spectrogram.png'):
        '''
        Plots the spectrogram (either mel or STFT) for a single sample and optionally saves it to disk.

        Args:
            spectrogram - 2d array for the spectrogram, either numpy array of torch tensor
            config - configuration dictionary
            saveFig - flag for saving the plots to disk
            filepath - path to the directory where the plot is saved
            filename - name for the plot
        '''

        raise NotImplementedError

        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.numpy().squeeze()  # should be a (mel_bins, frames) after the squeeze

        plt.figure()
        if config.useMel:
            librosa.display.specshow(spectrogram,
                                     y_axis='mel',
                                     x_axis='time',
                                     sr=config.resampling_rate,
                                     hop_length=config.hop_length,
                                     fmax=config.fmax,
                                     fmin=config.fmin)
            plt.title('Normalized Mel spectrogram')
        else:
            librosa.display.specshow(spectrogram, y_axis='log', x_axis='time', sr=config.resampling_rate,
                                     hop_length=config.hop_length)
            plt.title('Normalized Log-frequency power STFT')
        plt.colorbar(format='%+2.0f dB')
        if saveFig:
            matplotlib.use('agg')
            plt.savefig(os.path.join(filepath, filename))
        else:
            matplotlib.use('TkAgg')
            plt.interactive(False)
        plt.show()

    @staticmethod
    def show_spectrogram_batch(batch, config, gridSize=4,
                               saveFig=True, filepath='', filename='Output_batch_spectrogram'):
        '''
        Provides 3 visualizations (mel or STFT spectrograms, time domain plots, and wave files) of a batch of
        samples and optionally saves them to disk :

        Args:
            batch - a batch dictionary with fields 'spectrogram', 'labels', 'fname'
            config - configuration dictionary
            dataPath - path to the audio files
            gridSize - number of plots, in case that the batch is too big
            saveFig - flag for saving the plots to disk
            filepath - path to the directory where the plot is saved
            filename - name for the plot
        '''

        raise NotImplementedError

        specs, labels, fnames = batch['spectrogram'], batch['labels'], batch['fname']
        batchSize = specs.shape[0]
        idx = np.random.choice(range(batchSize), gridSize, replace=False)

        assert math.sqrt(gridSize).is_integer(), "gridSize should be a squared number"  # For now, just to make easier plots

        ## Spectrograms
        plt.figure()
        for j, i in enumerate(idx):
            plt.subplot(math.sqrt(gridSize), math.sqrt(gridSize), j + 1)
            tmpSpec = specs[i, :, :, :].numpy().squeeze()
            if config.useMel:
                librosa.display.specshow(tmpSpec, y_axis='mel', x_axis='time', sr=config.resampling_rate,
                                         hop_length=config.hop_length, fmax=config.fmax, fmin=config.fmin)
                plt.title('Normalized Mel spectrogram')
            else:
                librosa.display.specshow(tmpSpec, y_axis='linear', x_axis='time', sr=config.resampling_rate,
                                         hop_length=config.hop_length)
                plt.title('Normalized Linear-frequency power STFT')
            plt.colorbar(format='%+2.0f dB')

        if saveFig:
            plt.savefig(os.path.join(filepath, filename))
        plt.show()

    @staticmethod
    def show_waveforms_batch(batch, config, dataPath, gridSize=4,
                               saveFig=True, filepath='', filename='Output_batch_%s'):
        '''
        Provides 3 visualizations (mel or STFT spectrograms, time domain plots, and wave files) of a batch of
        samples and optionally saves them to disk :

        Args:
            batch - a batch dictionary with fields 'spectrogram', 'labels', 'fname'
            config - configuration dictionary
            dataPath - path to the audio files
            gridSize - number of plots, in case that the batch is too big
            saveFig - flag for saving the plots to disk
            filepath - path to the directory where the plot is saved
            filename - name for the plot
        '''

        raise NotImplementedError

        specs, labels, fnames = batch['spectrogram'], batch['labels'], batch['fname']
        batchSize = specs.shape[0]
        idx = np.random.choice(range(batchSize), gridSize, replace=False)

        assert math.sqrt(
            gridSize).is_integer(), "gridSize should be a squared number"  # For now, just to make easier plots


        ## Time domain plots AND wave files
        plt.figure()
        for j, i in enumerate(idx):
            plt.subplot(math.sqrt(gridSize), math.sqrt(gridSize), j + 1)

            fname = fnames[i]
            audio = AudioTaggingPreProcessing.read_audio(config, os.path.join(dataPath, fname), config.fixedLength)
            librosa.display.waveplot(audio, sr=config.resampling_rate)

            plt.title('%d %s' % (j, fname.split('.')[0].split('/')[-1]))

            librosa.output.write_wav(os.path.join(filepath, filename % ('%d' % (j) + '.wav')), audio,
                                     config.resampling_rate)

        if saveFig:
            plt.savefig(os.path.join(filepath, filename % ('waveforms.png')))
        plt.show()

    @staticmethod
    def show_classes_batch(y_preds, y_true, config, dataPath, gridSize=16,
                               saveFig=True, filepath='', filename='Output_predictions.png'):
        '''
        Provides 3 visualizations (mel or STFT spectrograms, time domain plots, and wave files) of a batch of
        samples and optionally saves them to disk :

        Args:
            spectrogram - 2d array for the spectrogram, either numpy array of torch tensor
            config - configuration dictionary
            saveFig - flag for saving the plots to disk
            filepath - path to the directory where the plot is saved
            filename - name for the plot
        '''

        raise NotImplementedError

        if isinstance(y_preds, torch.Tensor):
            y_preds = y_preds.numpy().squeeze()  # should be a (batch, classes) after the squeeze

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy().squeeze()  # should be a (batch, classes) after the squeeze

        assert math.sqrt(gridSize).is_integer(), \
                        "gridSize should be a squared number"  # For now, just to make easier plots

        plt.figure()
        ids = np.random.randint(y_preds.shape[0], size=gridSize)
        if y_true is not None:
            for i in range(0, gridSize, 2):
                tmpIdx = ids[i]

                # Ground truth
                plt.subplot(math.sqrt(gridSize), math.sqrt(gridSize), i + 1)
                tmp = y_true[tmpIdx, :].reshape(10, 8)
                plt.imshow(tmp, vmin=0, vmax=1)

                # Predictions
                plt.subplot(math.sqrt(gridSize), math.sqrt(gridSize), i + 2)
                tmp = y_preds[tmpIdx, :].reshape(10, 8)
                plt.imshow(tmp, vmin=0, vmax=1)
        else:
            for i in range(0, gridSize, 1):
                tmpIdx = ids[i]

                # Predictions
                plt.subplot(math.sqrt(gridSize), math.sqrt(gridSize), i + 1)
                tmp = y_preds[tmpIdx, :].reshape(10, 8)
                plt.imshow(tmp, vmin=0, vmax=1)
        if saveFig:
            plt.savefig(os.path.join(filepath, filename))
        plt.show()


class Util(object):
    @staticmethod
    def get_label_num(labels, thisTargetLabels):
        '''
        Inputs:
        labels = string with labels of sound file e.g. "door,clap,stuff_happening"
        thisTargetLabels = string array of ALL availabel target labels

        Returns:
        target_arr = one hot encoding the target labels for this sound file
        '''
        lbs = labels.split(",")
        target_arr = np.zeros(80)
        for lb in lbs:
            if (lb in thisTargetLabels):
                i = thisTargetLabels.index(lb)
                target_arr[i] = 1
                break
        return target_arr

    @staticmethod
    def get_lambda(size=(1)):
        '''
        For mix up data augmentation, get a random sample from the beta distribution with a= 0,4 and b=0.4
        '''
        lamb = 0
        while lamb < 0.5 or lamb > 1:
            lamb = np.random.beta(0.4, 0.4, size=size)

        return lamb

    @staticmethod
    def getNpyFilename(fname, train=True):
        '''

        Args:
            fname -- full path and filename of the audio file, e.g. '/scratch/work/falconr1/audioTaggingPKG/data/train_curated/1228a986.wav'
            train - flag to mark we are operating in train mode

        Returns:
            f_spec -- filename for the spec , e.g. 'train_curated-1228a986_spec.npy'
            f_label -- filename for the labels, e.g. 'train_curated-1228a986_y.npy'
        '''
        # filenmaes like: /trn_curated-0019ef41_aug_0_spec
        tmp = '-'.join(fname.split('.')[-2].split('/')[-2:])
        f_spec = tmp  + '_spec.npy'

        # filenmaes like: /trn_curated-0019ef41_aug_0_y
        tmp = '-'.join(fname.split('.')[-2].split('/')[-2:])
        f_label = tmp  + '_y.npy'

        return f_spec, f_label