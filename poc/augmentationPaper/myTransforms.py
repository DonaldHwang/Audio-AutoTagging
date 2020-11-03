import myFunctional as F
import torch
from audtorch import transforms as tforms_aud
from torchaudio import transforms as tforms_torch
from typing import Optional
import math
import os
import numpy as np
from typing import List
from torch import nn

class Debugger(object):
    '''
    Empty transformn, useful to debug and analyze the output of other transforms.

    '''

    def __init__(self):
        super().__init__()

    def __call__(self, signal):
        return F.debugger(signal)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class CFL2FLC(object):
    '''
    Transforms a numpy object from C x F x L to F x L x C, (C = channels, F = freqwuencies, L = length).

    This is useful, for example, to pass an spectrogram to PIL image and use the tochvision transforms.
    '''

    def __init__(self):
        super().__init__()

    def __call__(self, specgram):
        return F.CFL2FLC(specgram)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class ToNumpy(object):
    '''
    Transforms a Torch.Tensor to a numpy()
    '''

    def __init__(self):
        super().__init__()

    def __call__(self, tensor):
        return tensor.numpy()

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class ToTensor(object):
    '''
    Transforms a numpy array to Torch.Tensor
    '''

    def __init__(self):
        super().__init__()

    def __call__(self, numpy):
        return torch.Tensor(numpy)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class ToLC(object):
    def __init__(self):
        super().__init__()

    def __call__(self, numpy):
        return numpy.transpose()
    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class Spectrogram(object):
    '''
    Gets a spectrgoram in either STFT or MelSpectrogram, using my code and librosa.
    '''

    def __init__(self, config):
        self.hop_length = config.hop_length
        self.win_length = config.hop_length
        self.use_mels = config.use_mels
        self.n_fft = config.n_fft
        self.n_mels = config.n_mels
        self.resampling_rate = config.resampling_rate
        self.fmin = config.fmin
        self.fmax = config.fmax
        super().__init__()

    def __call__(self, audio):
        return F.spectrogram(audio,
                             self.hop_length,
                             self.win_length,
                             self.use_mels,
                             self.n_fft,
                             self.n_mels,
                             self.resampling_rate,
                             self.fmin,
                             self.fmax)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class SpecToMelSpec(object):
    '''
    Transforms a STFT into a mel-spectrogram
    '''

    def __init__(self, config, stype='energy'):
        self.stype = 'energy'
        self.n_fft = config.n_fft
        self.n_mels = config.n_mels
        self.resampling_rate = config.resampling_rate
        self.fmin = config.fmin
        self.fmax = config.fmax
        super().__init__()

    def __call__(self, spec):
        return F.SpecToMel(spec,
                           self.stype,
                           self.resampling_rate,
                           self.n_fft,
                           self.n_mels,
                           self.fmin,
                           self.fmax)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class MagnitudeToDb(object):
    """
    Transforms an amplitude or power spectrograsm to db scale using librosa.
    """
    def __init__(self, stype='power', ref='np.max'):
        self.stype = stype
        self.ref = ref
        super().__init__()

    def __call__(self, spec):
        return F.magnitude_to_db(spec,
                           self.stype,
                           self.ref,
                           )

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class RandomCrop_aud(tforms_aud.RandomCrop):
    def __call__(self, signal):
        input_size = signal.shape[self.axis]
        output_size = self.size

        # Pad or replicate if signal is too short
        if input_size < output_size:
            self.expand.size = output_size
            signal = self.expand(signal)
            input_size = output_size

        # Pad if random crop parameter is fixed and signal is too short
        if self.fix_randomization and input_size < self.idx[1]:
            self.expand.size = self.idx[1]
            signal = self.expand(signal)
            input_size = self.idx[1]

        if not self.fix_randomization:
            self.idx = self.random_index(input_size, output_size)

        return F.crop(signal, self.idx, axis=self.axis)


class RandomCrop(nn.Module):
    ''' Returns a random crop of a spectrogram.

    Loosely based on audtorch RandomCrop.
    '''

    def __init__(self, size, method='pad', value=-80):
        super(RandomCrop, self).__init__()

        self.size = size
        self.method = method
        self.value = value

    def forward(self, x):
        input_size = x.shape[-3:]  #  Assuming x ix [batch, channel, height, width]
        output_size = self.size

        idxs = []  # [[idx, idx + output_size[i]], [idx, idx + output_size[i]]
        padding = []  # padding is (padding_left, padding_right, padding_top, padding_right)
        for i in range(1,3):  #
            #  Spec too short, pad with 0s
            if input_size[i] < output_size[i]:
                tmp = (output_size[i] - input_size[i]) / 2
                if tmp % 2 == 0:
                    padding.extend([int(tmp), int(tmp)])  # symmetrical case
                else:
                    padding.extend([math.floor(tmp), math.floor(tmp)+1])  # asymmetrical
                idxs.append([0, 0 + output_size[i]])
            else:
                padding.extend([0,0])  # No padding needed for this dim

                difo = input_size[i] - output_size[i]
                idx = 0 if difo == 0 else np.random.randint(0, difo)
                idxs.append([idx, idx + output_size[i]])

        padding[:2], padding[2:] = padding[2:], padding[:2]  # Swap dimensions for pytorch...
        padder = torch.nn.ConstantPad2d(tuple(padding), self.value)
        x = padder(x)
        x = x[:,
            idxs[0][0] : idxs[0][1],
            idxs[1][0] : idxs[1][1]]

        return x


class Expand(object):
    r"""Expand signal.
    Ensures that the signal matches the desired output size by padding or
    replicating it.
    * :attr:`size` controls the size of output signal
    * :attr:`method` controls whether to replicate signal or pad it
    * :attr:`axis` controls axis of expansion
    The expansion is done by one of these methods:
        * ``'pad'`` expand the signal by adding trailing zeros
        * ``'replicate'`` replicate the signal to match the specified size.
          If result exceeds specified size after replication, the signal will
          then be cropped
    Args:
        size (int): desired length of output signal in samples
        method (str, optional): expansion method. Default: `pad`
        axis (int, optional): axis along to crop. Default: `-1`
    Shape:
        - Input: :math:`(*, N_\text{in}, *)`
        - Output: :math:`(*, N_\text{out}, *)`, where :math:`N_\text{in}` is
          the input length of the axis to expand and :math:`N_\text{out}` is
          the output length as given by `size`.
          :math:`*` can be any additional number of dimensions.
    Example:
        >>> a = np.array([[1, 2, 3]])
        >>> t = Expand(6)
        >>> print(t)
        Expand(size=6, method=pad, axis=-1)
        >>> t(a)
        array([[1, 2, 3, 0, 0, 0]])
    """

    def __init__(
            self,
            size,
            *,
            method='pad',
            axis=-1,
    ):
        super().__init__()
        self.size = size
        self.method = method
        self.axis = axis

    def __call__(self, signal):
        input_size = signal.shape[self.axis]
        output_size = self.size

        if input_size < output_size:
            if self.method == 'replicate':
                signal = F.replicate(signal, output_size // input_size + 1,
                                     axis=self.axis)
                signal = F.crop(signal, (0, output_size), axis=self.axis)
            elif self.method == 'pad':
                signal = F.pad(signal, (0, output_size - input_size),
                               axis=self.axis)
        return signal

    def __repr__(self):
        options = 'size={0}, method={1}, axis={2}'.format(
            self.size, self.method, self.axis)
        return '{0}({1})'.format(self.__class__.__name__, options)


class AmplitudeToDB(torch.jit.ScriptModule):
    r"""Turns a tensor from the power/amplitude scale to the decibel scale.

    This output depends on the maximum value in the input tensor, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        stype (str): scale of input tensor ('power' or 'magnitude'). The
            power being the elementwise square of the magnitude. (Default: ``'power'``)
        top_db (float, optional): minimum negative cut-off in decibels.  A reasonable number
            is 80. (Default: ``None``)
    """
    __constants__ = ['multiplier', 'amin', 'ref_value', 'db_multiplier']

    def __init__(self, stype='power', top_db=None, ref='max'):
        super(AmplitudeToDB, self).__init__()
        self.stype = torch.jit.Attribute(stype, str)
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = torch.jit.Attribute(top_db, Optional[float])
        self.multiplier = 10.0 if stype == 'power' else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    @torch.jit.script_method
    def forward(self, x):
        r"""Numerically stable implementation from Librosa
        https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html

        Args:
            x (torch.Tensor): Input tensor before being converted to decibel scale

        Returns:
            torch.Tensor: Output tensor in decibel scale
        """
        return F.amplitude_to_DB(x, self.multiplier, self.amin, self.db_multiplier, self.top_db)

class ApplyReverb(nn.Module):
    '''
        Applies reverb (ir) to an audios signal.

        fs = sampling frequnecy (IRs will be resampled)
    '''

    rirs: List[np.array]

    def __init__(self, fs):
        super(ApplyReverb, self).__init__()

        dir = '/m/cs/scratch/sequentialml/datasets/RIRs/razr'
        fn = 'BRIRs_23-Nov-2019_19-26-56.mat'
        fn = os.path.join(dir, fn)

        self.resampler = tforms_torch.Resample(orig_freq=48000, new_freq=fs)

        import h5py
        tmp_rirs = []
        with h5py.File(fn, 'r') as f:
            #for k, v in f.items():
             #   arrays[k] = np.array(v)

            for i in range(0, len(f['allIRs'])):
                tmp = np.array(f[f['allIRs'][i][0]]).transpose()
                tmp = np.mean(tmp, axis=1)  # mono
                tmp = torch.Tensor(tmp)

                if len(tmp.shape) < 2:
                    tmp = tmp.unsqueeze(0)  # shape is (channels, timesteps)

                tmp = self.resampler(tmp)
                tmp_rirs.append(tmp)

        self.rirs = tmp_rirs

        print("Loaded {} RIRs ", len(self.rirs))


    #@torch.jit.script_method
    def forward(self, x):
        return F.apply_reverb(x, self.rirs[0])


class ReScaleSpec(nn.Module):
    '''
    Normalizers a spectrogram so that each spectrgoram is divided by the abs().max().
    '''
    def __init__(self, range):
        super(ReScaleSpec, self).__init__()

        self.range = range

        print("Rescaling spectrograms to range: {}" .format(self.range))

    # @torch.jit.script_method
    def forward(self, x):
        #GANsynth normalizaton:

        means = x.mean(dim=-1, keepdims=True)
        stds = x.std(dim=-1, keepdims=True)
        x = (x - means) / (stds + 1e-10)
        x = x.clamp(-3, 3)
        y = x / 3



        #x = x / x.abs().max()  # TODO: this works for a single spec

        # TODO trying to solve it for batches
        # max_freq = x.abs().max(dim=-2, keepdims=True)[0]
        # max_time_freq = max_freq.max(dim=-1, keepdims=True)[0]
        #
        # y = x / max_time_freq

        assert not torch.isnan(y).any(), 'NAN in data'

        return y


class TrimAudio(nn.Module):
    def __init__(self, output_size):
        self.output_size = output_size

    def forward(self, x):
        raise NotImplementedError
        input_size = x.shape[-1]
        output_size = self.output_size

        # Pad or replicate if signal is too short
        if input_size < output_size:
            raise NotImplementedError
            self.expand.size = output_size
            signal = self.expand(signal)
            input_size = output_size


        if self.output_size > 0:
            output_size = self.output_size
            input_size = audio.shape[-1]  # Assuming last axis is time

            if input_size < output_size:
                to_pad = math.floor((output_size - input_size) / 2)
                audio = np.pad(audio, [(0, 0), (to_pad, to_pad)], 'constant', constant_values=0)
            else:
                idx = np.random.randint(0, input_size - output_size)
                audio = audio[:, idx:idx + output_size]

        # Pad if random crop parameter is fixed and signal is too short
        if input_size < self.idx[1]:
            self.expand.size = self.idx[1]
            signal = self.expand(signal)
            input_size = self.idx[1]

        if not self.fix_randomization:
            self.idx = self.random_index(input_size, output_size)

        return F.crop(signal, self.idx, axis=self.axis)