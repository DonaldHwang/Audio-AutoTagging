import numpy as np
import math
import torch
import librosa
import scipy
from torch import nn
import torch.nn.functional as func

__mel_bank__ = None

def apply_reverb(audio, ir):
    '''
    Convolves an audio signal with an impulse response to get an audio signal with reverberation.

    :param audio:
    :param reverb:
    :return:
    '''

    if isinstance(ir, np.ndarray):
        ir = torch.Tensor(ir)

        if len(ir.shape) < 2:
            ir = ir.unsqueeze(0)  # should be (channels, timesteps)

    #tmp = func.conv1d(audio, ir, bias=None )
    print(ir.shape)
    short = ir[:, 0:16800]
    tmp = func.conv1d(audio.view(1, audio.shape[0], audio.shape[1]), short.view(1, short.shape[1]).repeat(1, 1, 1),
                      bias=None, padding=short.shape[1] - 1)


    #tmp = func.conv1d(audio.view(1, audio.shape[0], audio.shape[1]), ir.view(1, ir.shape[1]).repeat(1, 1, 1), bias=None)
    return tmp


def get_Mels(resampling_rate, n_fft, n_mels, fmin, fmax):
    '''
    returns a mel fitler bank, used to compute the mel spectrograms
    '''

    global __mel_bank__
    if __mel_bank__ is None:
        __mel_bank__ = librosa.filters.mel(sr=resampling_rate,
                                           n_fft=n_fft,
                                           n_mels=n_mels,
                                           fmin=fmin,
                                           fmax=fmax,
                                           htk=False,
                                           norm=1)

    return __mel_bank__


def spectrogram(audio, hop_length, win_length, use_mels, n_fft, n_mels, resampling_rate, fmin=20, fmax=20000):
    '''
        Computes the spectrogram (mel of STFT) of a previously loaded audio signal.

        Params:
            config - Configuration dictionary.
            audio - audio signal (as np array)

        returns:
            spec - Spectrogram (np.array)
        '''

    if isinstance(audio, torch.Tensor):
        audio = audio.numpy().squeeze()  # should be a (channels, tiemsteps)

    audio = audio.squeeze()

    if use_mels:
        spec = librosa.core.stft(audio,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 n_fft=n_fft,
                                 center=False
                                 )

        mel = get_Mels(resampling_rate, n_fft, n_mels, fmin, fmax)  # Mels are stored as static variable to speed up the process

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
    spec = np.expand_dims(spec, axis=0)  # Add dimension for channel, final shape (batch, channels, freq_bins, frames)
    spec = torch.tensor(spec)
    return spec


def SpecToMel(spec, stype='energy', resampling_rate=44100, n_fft=2048, n_mels=128, fmin=20, fmax=20000):
    """
    Returns a mel-spectrgoram from a regular spectrogram (STFT).

    If the input is energy (e.g. output of STFT in audtorch), then use sttype='energy', so that we compute the power
    before doing the dot product with the mels.

    :param spec:
    :param resampling_rate:
    :param n_fft:
    :param n_mels:
    :param fmin:
    :param fmax:
    :return:
    """
    # spec is (batch, channels, stft_bins, frames),
    # mel is (n_mels, stft_bins)
    # mel spec is the dot product, along the stft_bins, for all specs in the batch
    # NOT TESTED for multi channel inputs
    axis = 2

    # Mels are stored as static variable to speed up the process
    mel = get_Mels(resampling_rate, n_fft, n_mels, fmin, fmax)

    # Mel spec ios the dot product of mel basis and the power spectrogram
    if stype == 'energy':
        spec = np.power(spec, 2)

    melspec = np.apply_along_axis(mel.dot, axis, spec)

    return melspec


def magnitude_to_db(spec, stype='power', ref='np.max'):
    """
    Transforms a magnitude or power spectrogram into a normalized db scale using librosa. If using ref np.max, each
    individual spectrogram will be in the range -80 to 0 db. (The lower limit varies).
    :param spec:
    :param type:
    :param ref:
    :return:

    If power, the spectrogram must already by a power spectrogram. This means that:
    out_complex = librosa.stft(waveform)  # this is complex
    magnitude, phase =  librosa.magphase(out_compelx)

    magnitude is energy, when power = 1
    To get power, power = 2

    This is the same as :
    np.pow(np.abs(out_complex), 2))
    """

    axis = 2

    additional_args = {'ref': np.max}

    if stype == 'power':  # this means the spec is alredy power
        spec_db = np.apply_along_axis(librosa.power_to_db, axis, spec, **additional_args)
    elif stype == 'magnitude':
        spec_db = np.apply_along_axis(librosa.amplitude_to_db, axis, spec, **additional_args)

    return spec_db


#@torch.jit.ignore
@torch.jit.script
def amplitude_to_DB(x, multiplier, amin, db_multiplier, top_db=None):
    # type: (Tensor, float, float, float, Optional[float]) -> Tensor
    r"""Turns a tensor from the power/amplitude scale to the decibel scale.

    This output depends on the maximum value in the input tensor, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        x (torch.Tensor): Input tensor before being converted to decibel scale
        multiplier (float): Use 10. for power and 20. for amplitude
        amin (float): Number to clamp ``x``
        db_multiplier (float): Log10(max(reference value and amin))
        top_db (Optional[float]): Minimum negative cut-off in decibels. A reasonable number
            is 80. (Default: ``None``)

    Returns:
        torch.Tensor: Output tensor in decibel scale
    """
    x_db = multiplier * torch.log10(torch.clamp(x, min=amin))
    x_db -= multiplier * db_multiplier

    # My change:
    # Each spectogram is normalized by its maximum value (10log10( x ./ max(x))
    # So that the range goes from -top_db to 0 dB
    # Without this, the range for every spec can vary
    x_db -= multiplier * torch.log10(torch.max(x))
    if top_db is not None:
        new_x_db_max = torch.tensor(float(x_db.max()) - top_db,
                                    dtype=x_db.dtype, device=x_db.device)
        x_db = torch.max(x_db, new_x_db_max)

    return x_db


def debugger(signal):
    return signal


def CFL2FLC(specgram):
    if not isinstance(specgram, np.ndarray):
        raise TypeError('specgram should be a ndarray. Got {}'.format(type(specgram)))

    return np.transpose(specgram, (1,2,0))


def EqShelvingCookbok(freq: int, gain: float, Q: float, fs: int = 44100, type: str = 'peak'):
    '''
    Returns the coefficients for a low pass, high pass, or peak biquad filters, using the methods from:
    http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

    Args:
        freq (int): frequency in Hz
        gain (float): gain in dB, e.g. -6, 20. Positive values can be dangerous.
        Q (float): Quality factor.
        fs (int): Samplign frequency in Hz.
        type (str, optional): Type of filter. Options {lowpas, highpass, peak}

    Returns:
        topCoefficients (list of floats): Coefficients for the numerator (e.g. [b0, b1, b2])
        botCoefficients (list of floats): Coefficients for the denominator (e.g. [a0, a1, a2])

    Note:
        All coefficients are normalized, divided by a0.

    Example:
    '''
    ## TODO: add exmaple

    G = math.pow(10, gain / 40)
    w0 = 2 * math.pi * freq / fs
    alpha = math.sin(w0) / 2 * Q

    if type == 'lowpass':
        pass
    elif type == 'highpass':
        pass
    elif type == 'peak':
        b0 = 1 + alpha * G
        b1 = -2 * math.cos(w0)
        b2 = 1 - alpha * G
        a0 = 1 + alpha / G
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha / G
    else:
        raise ValueError('Unsupported filter type {}, was exctecting \{lowpas, highpass, peak\}'.format(
            type
        ))

    topCoefficients = [b0, b1, b2] / a0
    botCoefficients = [a0, a1, a2] / a0

    return topCoefficients, botCoefficients


def crop(signal, idx, *, axis=-1):
    r"""Crop signal along an axis.
    Based on the audtorch.functional
    Args:
        signal (Torch Tensor): audio signal
        idx (int or tuple): first (and last) index to return
        axis (int, optional): axis along to crop. Default: `-1`
    Note:
        Indexing from the end with `-1`, `-2`, ... is allowed. But you cannot
        use `-1` in the second part of the tuple to specify the last entry.
        Instead you have to write `(-2, signal.shape[axis])` to get the last
        two entries of `axis`, or simply `-1` if you only want to get the last
        entry.
    Returns:
        Torch Tensor: cropped signal
    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> crop(a, 1)
        array([[2],
               [4]])
    """
    # Ensure idx is iterate able
    if isinstance(idx, int):
        idx = [idx]
    # Allow for -1 like syntax for index
    length = signal.shape[axis]
    idx = [length + i if i < 0 else i for i in idx]
    # Add stop index for single values
    if len(idx) == 1:
        idx = [idx[0], idx[0] + 1]

    # Split into three parts and return middle one
    # In torch.split, idx must be the sizes of all chunks
    myIdx = [idx[0], idx[1] - idx[0], length - idx[1]]
    tmp = torch.split(signal, myIdx, dim=-1)
    return tmp[1]
