
#!/usr/bin/env python3'
import os
import numpy as np
import torchaudio
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import utils
import librosa
import torch.functional as f


def rad2deg(tensor):
    r"""Function that converts angles from radians to degrees.

    See :class:`~torchgeometry.RadToDeg` for details.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Example:
        >>> input = tgm.pi * torch.rand(1, 3, 3)
        >>> output = tgm.rad2deg(input)
    """
    pi = torch.Tensor([3.14159265358979323846])
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))

    return 180. * tensor / pi.to(tensor.device).type(tensor.dtype)

def plot_time_fft(audio,
                       fs,
                       nfft=4096,
                       ms=0.003):
    '''
    Plots a very short waveform, about 3 milliseocnds, and the FFT of that waveform.
    '''
    # ms is 0.003 = 3 milliseconds

    segment_size = math.floor(ms * fs)
    idx = np.random.randint(30000, 50000)
    segment = audio[:, idx: idx + segment_size]
    segment /= segment.abs().max()

    assert nfft >= segment_size, 'NFFT is too short.'
    audio_post = torch.stft(segment, n_fft=nfft,
                            hop_length=segment_size + 1,
                            win_length=segment_size,
                            normalized=True,
                            pad_mode='constant'
                            )

    fig, ax = plt.subplots(2,1)
    time_values = np.arange(0.0, ms - 1 / fs, 1 / fs)
    ax[0].plot(time_values, segment[0, :])
    ax[0].set_xlabel('time (secs)')
    ax[0].set_ylabel('amplitude')

    bin_size = (fs) / nfft
    freq_values = np.arange(0.0, fs / 2 + bin_size, bin_size)
    tmp = audio_post.numpy()[0, ::-1, :, 0]
    ax[1].plot(freq_values, 20 * np.log10(np.abs(tmp)))
    ax[1].set_ylabel('freq (Hz)')
    ax[1].set_xscale('log')
    fstep = int(nfft / 5.0)
    frequency_ticks = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    ax[1].set_xticks(frequency_ticks)
    ax[1].set_xticklabels([ str(i) for i in frequency_ticks])
    ax[1].set_xlim([100, 20000])
    plt.show()


def plot_time_sfft(audio,
                  fs,
                  nfft=4096,
                  hop_length=0.02,
                  win_length=0.04):
    '''
    Plots a very STFT, log power spectra and phase, manually.
    '''

    audio /= audio.abs().max()
    hop_length = math.floor(hop_length * fs)
    win_length = math.floor(win_length * fs)
    length_seconds = audio.shape[-1] / fs

    audio_post = torch.stft(audio, n_fft=nfft,
                            hop_length=hop_length,
                            win_length=win_length,
                            normalized=True,
                            pad_mode='constant',
                            )

    fig, ax = plt.subplots(3, 1)
    time_values = np.arange(0.0, length_seconds, 1 / fs)
    ax[0].plot(time_values, audio[0, :])
    ax[0].set_xlabel('time (secs)')
    ax[0].set_ylabel('amplitude')

    bin_size = (fs) / nfft
    frequency_ticks_labels = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    frequency_ticks = [i / bin_size for i in frequency_ticks_labels]
    spec = audio_post[0, :, :, 0].abs()
    phase = audio_post[0, :, :, 1]

    log_power_spec = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)(spec)
    ax[1].imshow(log_power_spec.numpy()[:,:], interpolation='nearest', aspect='auto', cmap='magma')
    #ax[1].invert_yaxis()
    ax[1].set_ylim(ax[1].get_ylim()[::-1])
    ax[1].set_yticks(frequency_ticks)
    ax[1].set_yticklabels([str(i) for i in frequency_ticks_labels])

    #ax[1].set_ylim([100, 20000])

    ax[2].matshow(phase, interpolation='nearest', aspect='auto', cmap='magma')
    ax[2].set_ylim(ax[1].get_ylim()[::-1])

    plt.show()

def plot_stft_librosa(audio,
                      fs,
                      nfft=4096,
                      win_length=0.04,
                      hop_length=0.02):

    audio /= audio.abs().max()
    hop_length = math.floor(hop_length * fs)
    win_length = math.floor(win_length * fs)
    length_seconds = audio.shape[-1] / fs


    audio_post = torch.stft(audio, n_fft=nfft,
                            hop_length=hop_length,
                            win_length=win_length,
                            normalized=True,
                            pad_mode='constant',
                            )
    spec = audio_post[0, :, :, 0].abs()
    log_power_spec = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)(spec)
    log_power_spec = log_power_spec.numpy()

    plt.figure(2)
    librosa.display.specshow(log_power_spec, y_axis='log', x_axis='time', sr=fs,
                             hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.show()



class MultiResolutionSpectralLoss(torch.nn.Module):
    ''' Multi resolution Spectral Loss
    Based on the DDSP paper
    Computes the L1 loss of the STFT at differente nfft sizes
    Ther are two terms, L1loss of the magnitude SFTT, and L1loss of the log magnitude

    '''
    def __init__(self, nfft=[2048,1024,512,256,128,64], hop_size=0.75, fs=44100, alpha=1.0):
        super().__init__()
        self.nfft = nfft
        self.hop_size = hop_size
        self.alpha = alpha

    def forward(self, output, target):
        losses = []
        for nfft in self.nfft:
            spec_transform = torchaudio.transforms.Spectrogram(n_fft=nfft,
                                                               win_length=nfft,
                                                               hop_length=math.floor(nfft * self.hop_size),
                                                               power=1,
                                                               normalized=True)
            log_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)

            tmp_loss = nn.L1Loss()(spec_transform(output), spec_transform(target)) + \
                       self.alpha * nn.L1Loss()(log_transform(spec_transform(output)), log_transform(spec_transform(target)))
            losses.append(tmp_loss)

        return torch.sum(torch.Tensor(losses))


def test_STFTloss():
    filename1 = 'Single_502_1_RIR.wav'
    filename2= 'Single_503_1_RIR.wav'
    root_dir = '/m/cs/work/falconr1/datasets/MusicSamples'

    audio1, fs = torchaudio.load(os.path.join(root_dir, filename1), normalization=True)
    audio2, fs = torchaudio.load(os.path.join(root_dir, filename2), normalization=True)

    # Truncate so that both files have the same timesteps
    max_length = min(audio1.shape[-1], audio2.shape[-1])
    audio1 = audio1[:,0:max_length]
    audio2 = audio2[:, 0:max_length]

    criterion = MultiResolutionSpectralLoss()
    loss1 = criterion(audio1, audio1)
    loss2 = criterion(audio2, audio2)
    loss_mix1 = criterion(audio1, audio2)
    loss_mix2 = criterion(audio2, audio1)

    assert np.isclose(loss1, 0), 'Loss for audio 1 is too big'
    assert np.isclose(loss2, 0), 'Loss for audio 2 is too big'

    assert loss1 < loss_mix1, 'Self loss for audio 1 is larger than the losss compared to a different audio'
    assert loss2 < loss_mix2, 'Self loss for audio 2 is larger than the losss compared to a different audio'

    assert np.isclose(loss_mix1, loss_mix2), 'Loss should be symmetric'

    # Testing Batches
    batch = torch.stack((audio1, audio2))
    loss_batch = criterion(batch, batch)

    assert np.isclose(loss_batch, 0), 'Loss for batch nshould be close to 0'

def explore_muklti_resolution_STFT():
    filename1 = 'Single_502_1_RIR.wav'
    filename1 = 'Suzanne Vega 1987 - Toms Diner (acapella).mp3'
    root_dir = '/m/cs/work/falconr1/datasets/MusicSamples'

    audio1, fs = torchaudio.load(os.path.join(root_dir, filename1), normalization=True)
    audio = audio1[:,0:math.floor(20 * fs)]
    nfft = [2048,1024,512,256,128,64]
    hop_size = 0.75

    fig, ax = plt.subplots(1, len(nfft))
    ctr = 0
    for this_nfft in nfft:
        spec_transform = torchaudio.transforms.Spectrogram(n_fft=this_nfft,
                                                           win_length=this_nfft,
                                                           hop_length=math.floor(this_nfft * hop_size),
                                                           power=1,
                                                           normalized=True)
        log_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)

        spec = log_transform(spec_transform(audio1))

        bin_size = (fs) / this_nfft
        frequency_ticks_labels = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        frequency_ticks = [i / bin_size for i in frequency_ticks_labels]
        spec = spec[0, :, :]

        #time_values = np.arange(0.0, length_seconds, 1 / fs)
        ax[ctr].imshow(spec.numpy()[:, :], interpolation='nearest', aspect='auto', cmap='magma')
        # ax[1].invert_yaxis()
        ax[ctr].set_ylim(ax[ctr].get_ylim()[::-1])
        #ax[ctr].set_yticks(frequency_ticks)
        #ax[ctr].set_yticklabels([str(i) for i in frequency_ticks_labels])
        ctr += 1
    plt.show()

    return 0



def main():

    filename = 'Suzanne Vega 1987 - Toms Diner (acapella).mp3'
    #filename = 'genelec_sweep1.mp3'
    root_dir = '/m/cs/work/falconr1/datasets/MusicSamples'

    nfft = 4096
    secs = 30
    hop_length = 0.01
    win_length = 0.01

    audio, fs = torchaudio.load(os.path.join(root_dir, filename), normalization=True)
    hop_length = math.floor(hop_length * fs)
    win_length = math.floor(win_length * fs)

    # Grab 60 s of
    audio = audio[:, 0 : secs * fs]

    plot_time_fft(audio, fs)
    plot_time_sfft(audio, fs)
    plot_stft_librosa(audio, fs)

    audio_post = torch.stft(audio, n_fft=nfft,
                          hop_length=hop_length,
                          win_length=win_length,
                          normalized=True
                          )

    fig, ax = plt.subplots(2, 1)
    log_power_spec = 20 * torch.log10(audio_post[0, ::-1, :, 0].abs())
    log_power_spec = log_power_spec.clamp(-80, 0)
    ax[0].matshow(log_power_spec.numpy()[::-1,:])

    phase_spec = audio_post[0, ::-1, :, 1]
    ax[1].matshow(phase_spec.numpy()[::-1,:])
    plt.show()


    tform = nn.Sequential(torchaudio.transforms.Spectrogram(n_fft=2048,
                                                            win_length=math.floor(0.04 * fs),
                                                            hop_length=math.floor(0.02 * fs),
                                                            power=None,
                                                            normalized=True))
    rir_post_2 = tform(rir)
    log_power_spec_2 = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)(rir_post_2[:, :, :, 0])
    log_power_spec_2 = torch.repeat_interleave(log_power_spec_2, 30, dim=-1)
    phase_spec_2 = torch.repeat_interleave(rir_post_2[:, :, :, 1], 30, dim=-1)
    phase_spec_3 = rad2deg(phase_spec_2)

    fig, ax = plt.subplots(3, 1)
    ax[0].matshow(log_power_spec_2[0, :, :])
    ax[1].matshow(phase_spec_2[0, :, :])
    ax[2].matshow(phase_spec_3[0, :, :])
    plt.show()



if __name__ == '__main__':
    explore_muklti_resolution_STFT()
    test_STFTloss()

    #main()
    #test_fft()


