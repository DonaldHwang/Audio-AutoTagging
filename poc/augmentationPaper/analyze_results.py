import numpy as np
import os
import utils
import matplotlib.pyplot as plt

results_dir = './results_debug'


true_class = np.load(os.path.join(results_dir, 'true_class.npy'))
pred_scores = np.load(os.path.join(results_dir, 'pred_scores.npy'))

utils.compare_predictions(true_class, pred_scores, results_dir)

a = 1
