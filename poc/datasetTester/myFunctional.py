import numpy as np

def debugger(signal):
    return signal

def CFL2FLC(specgram):
    if not isinstance(specgram, np.ndarray):
        raise TypeError('specgram should be a ndarray. Got {}'.format(type(specgram)))

    return np.transpose(specgram, (1,2,0))

