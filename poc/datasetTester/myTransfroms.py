import myFunctional as F

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

