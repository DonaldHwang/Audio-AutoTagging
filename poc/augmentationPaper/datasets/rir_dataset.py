#!/usr/bin/env python3'
from torch.utils.data import Dataset
import os
import numpy as np
import torchaudio
import torch
from tqdm import tqdm, trange
import torch.utils.data.dataloader
import warnings
import math
import h5py
import json
import scipy.io
import matplotlib.pyplot as plt


class RIRsDataset():
    '''
    Torch dataset that reads Room Impulses Responses generated in Matlab, and stored in a single mat file, using HDF5
    format.

    For the meta, as of 10.03.2020:
    The Meta is a dictionary with the following fields:
        -- id
        -- fs
        -- stats (list):
            -- 0 - T60 - [freq_bands, channels]
            -- 1 - DRR - [1, channels]
            -- 2 - C50 - [1, channels]
            -- 3 - EDT - [freq_bands, channels]
            -- 4 - LDT - [freq_bands, channels]
        -- total_vol
        -- rooms (dict)
            -- boxsize (list) - depth, width, height
            -- materials (list) - [freq_bands, walls] though not sure about the walls order.
            -- srcpos
            -- recpos
            -- recdir (in degrees)

    Each item returns:
        RIR - (torch.Tensor) - [channels, timesteps]
        stats - (np.ndarray) - [stats, ] -  A vector of stats, as selected in stats_for_conditioning
        geometry - (np.ndarray) - [47, ] - All geometrical features as vector
    '''
    def __init__(self, root_directory='/m/cs/scratch/sequentialml/datasets/RIRs/razr',
                 fname_rir='BRIRs_23-Nov-2019_19-26-56.mat',
                 fname_meta='meta_23-Nov-2019_19-26-56.json',
                 resampler=None,
                 prefetch_files=False,
                 data_limit=100,
                 use_mono=True,
                 stats_for_conditioning=['t60_band', 'edt_band', 'ldt_band', 'drr_broad', 'c50_broad'],
                 encode_stats=True):

        self.root_directory = root_directory
        self.fname_rir = fname_rir
        self.fname_meta = fname_meta
        self.resampler = resampler  # tforms_torch.Resample(orig_freq=48000, new_freq=fs)
        self.prefetch_files = prefetch_files
        self.data_limit = data_limit
        self.use_mono = use_mono
        self.encode_stats = encode_stats

        self.stats_for_conditioning = stats_for_conditioning

        ctr = 0
        if self.prefetch_files:
            print('Loading RIRs')
            tmp_rirs = []
            with h5py.File(os.path.join(self.root_directory, self.fname_rir), 'r') as f:
                for i in tqdm(range(0, len(f['allIRs']))):
                    if i >= self.data_limit:
                        break
                    tmp = np.array(f[f['allIRs'][i][0]]).transpose()
                    if self.use_mono:
                        #  tmp = np.mean(tmp, axis=1)  # mono
                        tmp = tmp[:,0]  # Keep left channel
                    tmp = torch.Tensor(tmp)

                    if len(tmp.shape) < 2:
                        tmp = tmp.unsqueeze(0)  # shape is [channels, timesteps]
                    else:
                        raise NotImplementedError

                    if self.resampler is not None:
                        tmp = self.resampler(tmp)
                    tmp_rirs.append(tmp)

            self.rirs = tmp_rirs
            self.stats = {
                't60_broad' : [],  # [batch, channels, freq_bands]
                't60_band' : [],
                'edt_broad': [],
                'edt_band': [],
                'ldt_broad': [],
                'ldt_band': [],
                'drr_broad': [],
                'c50_broad': [],
            }
            self.geometry = []

            with open(os.path.join(self.root_directory, self.fname_meta), 'r') as f:
                ctr = 0
                meta = json.load(f)

                for dat in meta:
                    t60s = np.array(dat['stats'][0]).transpose()[0 : 1 if self.use_mono else tmp.shape[0]]  # [channels, freq_bands]
                    edts = np.array(dat['stats'][3]).transpose()[0 : 1 if self.use_mono else 2]  # [channels, freq_bands]
                    ldts = np.array(dat['stats'][4]).transpose()[0: 1 if self.use_mono else 2]  # [channels, freq_bands]
                    drr = np.expand_dims(np.array(dat['stats'][1]).transpose()[0: 1 if self.use_mono else 2], axis=0)  # [channels, freq_bands]
                    c50 = np.expand_dims(np.array(dat['stats'][2]).transpose()[0: 1 if self.use_mono else 2], axis=0)  # [channels, freq_bands]


                    yolo = np.copy(np.mean(t60s, axis=1, keepdims=True))
                    self.stats['t60_broad'].append(np.mean(t60s, axis=1, keepdims=True))
                    self.stats['t60_band'].append(t60s)
                    self.stats['edt_broad'].append(np.mean(edts, axis=1, keepdims=True))
                    self.stats['edt_band'].append(edts)
                    self.stats['ldt_broad'].append(np.mean(ldts, axis=1, keepdims=True))
                    self.stats['ldt_band'].append(ldts)
                    self.stats['drr_broad'].append(drr)
                    self.stats['c50_broad'].append(c50)

                    tmp_dat = []
                    tmp_dat.extend(dat['rooms']['boxsize'])
                    tmp_dat.extend(np.reshape(np.array(dat['rooms']['materials']), (-1)).tolist())
                    tmp_dat.extend(dat['recdir'])
                    tmp_dat.extend(dat['recpos'])
                    tmp_dat.extend(dat['scrpos'])

                    self.geometry.append(tmp_dat)

                self.stats['t60_broad'] = self._ToNormLinearFromLog(np.array(self.stats['t60_broad']))
                self.stats['t60_band'] = self._ToNormLinearFromLog(np.array(self.stats['t60_band']))
                self.stats['edt_broad'] = self._ToNormLinearFromLog(np.array(self.stats['edt_broad']))
                self.stats['edt_band'] = self._ToNormLinearFromLog(np.array(self.stats['edt_band']))
                self.stats['ldt_broad'] = self._ToNormLinearFromLog(np.array(self.stats['ldt_broad']))
                self.stats['ldt_band'] = self._ToNormLinearFromLog(np.array(self.stats['ldt_band']))
                self.stats['drr_broad'] = self._ToNormLinearFromLinear(np.array(self.stats['drr_broad']))
                self.stats['c50_broad'] = self._ToNormLinearFromLinear(np.array(self.stats['c50_broad']))

                self.geometry = np.array(self.geometry)

            print('RIRS loaded')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root_directory)
        fmt_str += '    Stats for conditiong {}\n'.format(self.stats_for_conditioning)

        return fmt_str

    def __len__(self):
        assert len(self.rirs) == self.stats['t60_broad'].shape[0], 'Wrong shape in RIRs data.'
        assert len(self.rirs) == self.geometry.shape[0], 'Wrong shape in RIRs data.'
        return len(self.rirs)

    def __getitem__(self, item):
        if not self.use_mono:
            raise NotImplementedError
        if not self.prefetch_files:
            raise NotImplementedError

        rir = self.rirs[item]

        tmp = []
        for str in self.stats_for_conditioning:
            tmp.extend(self.stats[str][item, 0, :])

        stats = np.array(tmp)
        geometry = self.geometry[item, :]

        return rir, stats, geometry

    def _ToNormLinearFromLinear(self, linear_values, input_range=[-20, 20]):
        '''
        Returns a normalize value for a param is linear or is already in log scale.
        For example, DRR in range -20 to 20 db, returns [-1 to 1].
        '''
        assert len(input_range) == 2, 'Wrong output_range, got {}'.format(input_range)
        assert input_range[0] < input_range[1], 'Wrong output_range, got {}'.format(input_range)

        tmp = linear_values / abs(max(input_range))
        return tmp

    def _ToLinearFormNormLinear(self, norm_linear_values, output_range=[-20, 20]):
        '''
        Returns an unnormalized value for a param that is linear or is already in log scale.
        For example, DRR normalized to [-1, 1] returns range -20 to 20 db.
        '''
        assert len(output_range) == 2, 'Wrong output_range, got {}'.format(output_range)
        assert output_range[0] < output_range[1], 'Wrong output_range, got {}'.format(output_range)

        tmp = norm_linear_values * abs(max(output_range))
        #tmp = (output_range[1] + abs(output_range[0])) * norm_linear_values + output_range[0]
        return tmp

    def _ToNormLinearFromLog(self, log_values, input_range=[0,5]):
        '''
        Maps values in log scale to a normalize linear scale.
        For example, seconds in range [0 , 5] secs are mapped to [-1, 1].
        The seconds are in log scale, just like Hertz.

        :param log_values:
        :param input_range:
        :return:
        '''

        norm_range = [-1,1]
        assert len(input_range) == 2, 'Wrong output_range, got {}'.format(input_range)
        assert input_range[0] < input_range[1], 'Wrong output_range, got {}'.format(input_range)

        b = math.log((input_range[0] + 1e-8)/ input_range[1] ) / (norm_range[0] - norm_range[1])
        a = (input_range[0] + 1e-8) / math.exp(b * norm_range[0])

        tmp = (np.log(log_values) - math.log(a)) / b
        return tmp

    def _ToLogFromNormLinear(self, log_values, input_range=[0, 5]):
        '''
        Maps values to log scale some normalize linear values.
        For example, seconds in normalized range [-1, 1], are mapped to log scale in [0, 5] seconds.
        The seconds are in log scale, just like Hertz.

        :param log_values:
        :param input_range:
        :return:
        '''

        norm_range = [-1, 1]
        assert len(input_range) == 2, 'Wrong output_range, got {}'.format(input_range)
        assert input_range[0] < input_range[1], 'Wrong output_range, got {}'.format(input_range)

        b = math.log((input_range[0] + 1e-8)/ input_range[1] ) / (norm_range[0] - norm_range[1])
        a = (input_range[0] + 1e-8) / math.exp(b * norm_range[0])

        tmp = a * np.exp(log_values * b)
        return tmp


class RIRsDataset_single_files(RIRsDataset):
    '''
    RIRsDataset that reads individual files instead of a collection.
    '''
    def __init__(self, directory_root='/m/cs/scratch/sequentialml/datasets/RIRs/razr',
                 directory_rir = 'multi_rir_hrtf_true__noise_80__normalize_true',
                 fname_rir_template='{:04d}_RIRs.wav',
                 fname_meta_template='{:04d}_meta.json',
                 fname_curves_template='{:04d}_decay_curves.mat',
                 resampler=None,
                 prefetch_files=False,
                 data_limit=100,
                 use_mono=True,
                 stats_for_conditioning=['t60_band', 'edt_band', 'ldt_band', 'drr_broad', 'c50_broad', 'level_band', 'power_broad', 'noise_snr_broad'],
                 encode_stats=True,
                 use_normalized_stats=True ):

        self.directory_root = directory_root
        self.directory_rir = directory_rir
        self.fname_rir_template = fname_rir_template
        self.fname_meta_template = fname_meta_template
        self.fname_curves_template = fname_curves_template
        self.resampler = resampler  # tforms_torch.Resample(orig_freq=48000, new_freq=fs)
        self.prefetch_files = prefetch_files
        self.data_limit = data_limit
        self.use_mono = use_mono
        self.encode_stats = encode_stats
        self.use_normalized_stats = use_normalized_stats

        self.stats_for_conditioning = stats_for_conditioning
        self.geometry = []
        self.stats = {
            't60_broad': [],  # [batch, channels, freq_bands]
            't60_band': [],
            'edt_broad': [],
            'edt_band': [],
            'ldt_broad': [],
            'ldt_band': [],
            'drr_broad': [],
            'c50_broad': [],
            'level_broad': [],
            'level_band': [],
            'power_broad': [],
            'noise_snr_broad': [],
        }

        ctr = 0
        ids = [i for i in range(1,data_limit)]
        if self.prefetch_files:
            print('Prefetching files')
            tmp_rirs = []
            tmp_curves = []
            tmp_meta = []

            for id in ids:
                try:
                    # Meta
                    with open(os.path.join(self.directory_root, self.directory_rir, self.fname_meta_template.format(id)), 'r') as f:
                        this_json = json.load(f)

                    t60s = np.array(this_json['stats'][0]).transpose()[
                           0: 1 if self.use_mono else tmp.shape[0]]  # [channels, freq_bands]
                    edts = np.array(this_json['stats'][3]).transpose()[0: 1 if self.use_mono else 2]  # [channels, freq_bands]
                    ldts = np.array(this_json['stats'][4]).transpose()[0: 1 if self.use_mono else 2]  # [channels, freq_bands]
                    lev = np.array(this_json['stats'][5]).transpose()[0: 1 if self.use_mono else 2]  # [channels, freq_bands]
                    drr = np.expand_dims(np.array(this_json['stats'][1]).transpose()[0: 1 if self.use_mono else 2],
                                         axis=0)  # [channels, freq_bands]
                    c50 = np.expand_dims(np.array(this_json['stats'][2]).transpose()[0: 1 if self.use_mono else 2],
                                         axis=0)  # [channels, freq_bands]
                    sig_power = np.expand_dims(np.array(this_json['sig_power']).transpose()[0: 1 if self.use_mono else 2],
                                         axis=0)
                    noise_snr = np.expand_dims(np.expand_dims(np.array(this_json['noise_snr']).transpose(), axis=-1),
                                         axis=0)

                    # this_meta['t60_broad'] = np.mean(t60s, axis=1, keepdims=True)
                    # this_meta['t60_band'] = t60s
                    # this_meta['edt_broad'] = np.mean(edts, axis=1, keepdims=True)
                    # this_meta['edt_band'] = edts
                    # this_meta['ldt_broad'] = np.mean(ldts, axis=1, keepdims=True)
                    # this_meta['ldt_band'] = ldts
                    # this_meta['drr_broad'] = drr
                    # this_meta['c50_broad'] = c50
                    # this_meta['drr_broad'] = sig_power
                    # this_meta['c50_broad'] = noise_snr

                    self.stats['t60_broad'].append(np.mean(t60s, axis=1, keepdims=True))
                    self.stats['t60_band'].append(t60s)
                    self.stats['edt_broad'].append(np.mean(edts, axis=1, keepdims=True))
                    self.stats['edt_band'].append(edts)
                    self.stats['ldt_broad'].append(np.mean(ldts, axis=1, keepdims=True))
                    self.stats['ldt_band'].append(ldts)
                    self.stats['drr_broad'].append(drr)
                    self.stats['c50_broad'].append(c50)
                    self.stats['level_broad'].append(np.mean(lev, axis=1, keepdims=True))
                    self.stats['level_band'].append(lev)
                    self.stats['power_broad'].append(sig_power)
                    self.stats['noise_snr_broad'].append(noise_snr)

                    tmp_dat = []
                    tmp_dat.extend(this_json['rooms']['boxsize'])
                    tmp_dat.extend(np.reshape(np.array(this_json['rooms']['materials']), (-1)).tolist())
                    tmp_dat.extend(this_json['recdir'])
                    tmp_dat.extend(this_json['recpos'])
                    tmp_dat.extend(this_json['scrpos'])

                    self.geometry.append(tmp_dat)

                    # Curves
                    this_curves = scipy.io.loadmat(os.path.join(self.directory_root, self.directory_rir, self.fname_curves_template.format(id)))
                    this_curves = torch.Tensor(this_curves['decay_curves'])  # Matlab saves curves as [frequency, timesteps, channels]
                    this_curves = this_curves.permute(2, 0, 1) # Transpose to |channel, frequency, timesteps]

                    if self.use_mono:
                        this_curves = this_curves[0:1,:,:]

                    # RIR (wav)
                    this_rir, fs = torchaudio.load(os.path.join(self.directory_root, self.directory_rir, self.fname_rir_template.format(id)))
                    if self.use_mono:
                        this_rir = this_rir[0:1, :]  # Keep left channel

                    if len(this_rir.shape) < 2:
                        this_rir = this_rir.unsqueeze(0)  # shape is [channels, timesteps]

                    if self.resampler is not None:
                        this_rir = self.resampler(this_rir)

                    assert this_rir.shape[-1] == this_curves.shape[-1], 'The timestemp length for rir and curves shoudl be the same.'
                except Exception as e:
                    warnings.warn("Error loading file {}" .format(id))
                    print(e)
                    continue

                #tmp_meta.append(this_meta)
                tmp_curves.append(this_curves)
                tmp_rirs.append(this_rir)

            self.rirs = tmp_rirs
            self.curves = tmp_curves

            self.geometry = np.array(self.geometry)
            self.stats['t60_broad'] = np.array(self.stats['t60_broad'])
            self.stats['t60_band'] = np.array(self.stats['t60_band'])
            self.stats['edt_broad'] =np.array(self.stats['edt_broad'])
            self.stats['edt_band'] = np.array(self.stats['edt_band'])
            self.stats['ldt_broad'] = np.array(self.stats['ldt_broad'])
            self.stats['ldt_band'] = np.array(self.stats['ldt_band'])
            self.stats['drr_broad'] = np.array(self.stats['drr_broad'])
            self.stats['c50_broad'] = np.array(self.stats['c50_broad'])
            self.stats['level_broad'] = np.array(self.stats['level_broad'])
            self.stats['level_band'] = np.array(self.stats['level_band'])
            self.stats['power_broad'] = np.array(self.stats['power_broad'])
            self.stats['noise_snr_broad'] = np.array(self.stats['noise_snr_broad'])

            if use_normalized_stats:
                self.__normalize_stats__()
            print('RIRS loaded')

    def __normalize_stats__(self):
        self.stats_norm = {}
        eps = np.finfo(np.float32).eps
        for stat in self.stats_for_conditioning:
            if stat =='noise_snr_broad':  # Ignore noise_snr, as it is fixed for all
                self.stats_norm[stat] = self.stats[stat]
                continue

            tmp_mean = np.mean(self.stats[stat], axis=0)
            tmp_std = np.std(self.stats[stat], axis=0)

            self.stats_norm[stat] = (self.stats[stat] - tmp_mean) / (tmp_std + eps)

    def __len__(self):
        assert len(self.rirs) == len(self.curves), 'Wrong shape in RIRs data.'
        assert len(self.rirs) == self.stats['t60_broad'].shape[0], 'Wrong shape in RIRs data.'
        assert len(self.rirs) == self.geometry.shape[0], 'Wrong shape in RIRs data.'

        return len(self.rirs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.directory_rir)

        return fmt_str

    def __getitem__(self, item):
        if not self.use_mono:
            raise NotImplementedError
        if not self.prefetch_files:
            raise NotImplementedError

        rir = self.rirs[item]
        curves = self.curves[item]

        tmp = []
        for str in self.stats_for_conditioning:
            dat = self.stats_norm[str][item, 0, :] if self.use_normalized_stats else self.stats
            tmp.extend(dat)

        stats = np.array(tmp)
        geometry = self.geometry[item, :]

        return rir, stats, geometry, curves


def main():
    dataset = RIRsDataset(prefetch_files=True)

    print(dataset)

    rir, stats, geometry = dataset[0]

    print('RIR with shape {}'.format(rir.shape))
    print('Stats with shape {}'.format(stats.shape))
    print(stats)
    print("Room geometry with shape {}".format(geometry.shape))
    print(geometry)

    # Test the mappings
    data_range = [-20, 20]
    data_points = 100
    sample = data_range[0] + (data_range[1] - data_range[0]) * np.random.rand(data_points,1)
    norm_sample = dataset._ToNormLinearFromLinear(sample)

    assert np.allclose(dataset._ToLinearFormNormLinear(norm_sample), sample), 'Wrong normalization for linear.'
    assert np.all(norm_sample <= 1) and np.all(norm_sample >= -1), 'Wrong normalization for linear.'

    data_range = [0.1, 5]  # in seconds
    sample = data_range[0] + (data_range[1] - data_range[0]) * np.random.rand(data_points,1)
    norm_sample = dataset._ToNormLinearFromLog(sample, data_range)


    assert np.allclose(dataset._ToLogFromNormLinear(norm_sample, data_range), sample), 'Wrong normalization for log.'
    assert np.all(norm_sample <= 1) and np.all(norm_sample >= -1), 'Wrong normalization for log.'



def test_fft():
    dataset = RIRsDataset(prefetch_files=True)

    rir, stats, geometry = dataset[0]
    fs = 48000

    rir = rir / rir.abs().max()

    rir_post = torch.stft(rir, n_fft=2048,
                          hop_length=math.floor(0.02 * fs),
                          win_length=math.floor(0.040 * fs),
                          normalized=True
                          )

    import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(2, 1)
    log_power_spec = 20 * torch.log10(rir_post[0, :, :, 0].permute(1, 0).abs())
    log_power_spec = log_power_spec.clamp(-80, 0)
    ax[0].matshow(log_power_spec)

    phase_spec = rir_post[0, :, :, 1]
    ax[1].matshow(phase_spec)
    plt.show()

    import torch.nn as nn
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


def test_RIR_single():
    dataset = RIRsDataset_single_files(prefetch_files=True)

    print(dataset)

    rir, stats, geometry, curves = dataset[0]

    print('RIR with shape {}'.format(rir.shape))
    print('Stats with shape {}'.format(stats.shape))
    print(stats)
    print("Room geometry with shape {}".format(geometry.shape))
    print(geometry)

    plt.figure()
    for i in range(6):
        plt.plot(curves[0, i, :])
    plt.show()


if __name__ == '__main__':
    #main()
    #test_fft()
    test_RIR_single()