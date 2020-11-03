from scipy import signal
import numpy as np
from random import gauss
import math
import matplotlib.pyplot as plt

def get_template(samples=8000, padding_samples=1000, decay_samples=4000):
    if decay_samples < 0:
        decay_samples = math.floor(samples / 5)

    mu, sigma = 0.0, 1.0
    template = np.array([gauss(mu, sigma) for i in range(samples)])

    tau2 = -(decay_samples - 1) / np.log(0.01)
    decay = signal.exponential(samples, 0, tau2, False)
    template /= np.max(np.abs(template))

    tmp = np.zeros((padding_samples,))
    #template = np.concatenate([tmp, decay * template], axis=0)
    template = np.concatenate([tmp, decay], axis=0)  # only decay curve
    return template


if __name__ == '__main__':
    powers = []
    for j in range(2000):
        template = get_template(8000, 3500, 4000)
        powers.append(np.sum(template**2))

    print('AVG Power')
    print(np.mean(np.array(powers)))

    plt.figure()
    plt.plot(template)
    plt.show()

