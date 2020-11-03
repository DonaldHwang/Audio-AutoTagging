import scipy.io as sio
from os import path
import h5py
import numpy as np
import json
import pandas as pd

directory = '/Volumes/falconr1/data/Documents/00Code/razr/myCode'
filename = 'BRIRs.mat'
filenameJson = 'meta.json'

fullfile = path.join(directory, filename)

## Read the meta data
with open(path.join(directory, filenameJson)) as json_file:
    data = json.load(json_file)

meta = pd.DataFrame(data)
#meta = pd.DataFrame.from_records(data)
#meta = pd.DataFrame.from_dict(data)

## Read the BRIRs
BRIRs = []
with h5py.File(fullfile, 'r') as f:
    for column in f['allIRs']:
        tmp = np.array(f[column[0]][:]).transpose()
        BRIRs.append(tmp)

yolo = 1