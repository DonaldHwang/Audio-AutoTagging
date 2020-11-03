from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import os
import librosa
from PIL import Image
import pandas as pd
import numpy as np
import sounddevice as sd
import torch


## based on
## https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789534092/1/ch01lvl1sec13/loading-data
##

class ToyDataset(Dataset):

    def __init__(self, datapath, labelsFile, transforms):
        self.dataPath = datapath
        self.transform = transforms


        with open(os.path.join(self.dataPath, labelsFile)) as f:
            self.labels=[tuple(line) for line in csv.reader(f)]
        # check that all images files exist
        for i in range(len(self.labels)):
            assert os.path.isfile(datapath + '/' + self.labels[i][0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imageName, imageLabel=self.labels[idx][0:]
        imagePath, = os.path.join(self.dataPath, imageName)
        image = Image.open(open(imagePath, 'rb'))

        #Transforms the image if required
        if self.transform:
            image = self.transform(image)
        return((image, imageLabel))


class AudioDataset(Dataset):

    def __init__(self, datapath, labelsFile, targetLabels, transforms=None):
        self.dataPath = datapath
        self.targetLabels = targetLabels
        self.transform = transforms

        tmp = pd.read_csv(os.path.join(self.dataPath, labelsFile))
        self.fnames = tmp['fname'].tolist()
        self.labelsString = tmp['labels'].tolist()

        # check that all images files exist
        for i in range(len(self.labelsString)):
            try:
                assert os.path.isfile(datapath + '/' + self.fnames[i])
                assert len(self.fnames) == len(self.labelsString)
            except AssertionError as error:
                print('Error %d       %s ' % (i, self.fnames[i]))
                raise

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        '''

        :param idx:
        :return: Each item has:
        a tensor of the audio file
        a one hot encoded vector for the labels (based on  the tarfgetLabels provided)
        the string of the labels for this item
        the sample rate
        the diration in seconds
        '''
        audioName = self.fnames[idx]
        labelString = self.labelsString[idx]
        labels = self.getLabelVector(labelString, self.targetLabels)
        audioPath = os.path.join(self.dataPath, audioName)

        audio, sr = librosa.load(path=audioPath,
                                 mono=True)

        audio = librosa.util.normalize(audio)
        duration = len(audio) / sr

        # Transforms the image if required
        if self.transform:
            audio = self.transform(audio)
        return ((audio, labels, labelString, sr, duration))

    def getLabelVector(self, labels, targetLabels):
        '''
        Inputs:
        labels = string with labels of sound file e.g. "door,clap,stuff_happening"
        thisTargetLabels = string array of ALL availabel target labels

        Returns:
        target_arr = one hot encoding the target labels for this sound file
        '''
        lbs = labels.split(",")
        target_arr = np.zeros(len(targetLabels))
        for lb in lbs:
            if (lb in targetLabels):
                i = targetLabels.index(lb)
                target_arr[i] = 1
                break
        return target_arr



csvFile = '/Volumes/scratch/work/falconr1/audioTaggingPKG/data/train_curated.csv'
datapath = '/Volumes/scratch/work/falconr1/audioTaggingPKG/data/train_curated'
targetLabels = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark',
                     'Bass_drum', 'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell',
                     'Burping_and_eructation', 'Bus', 'Buzz', 'Car_passing_by', 'Cheering', 'Chewing_and_mastication',
                     'Child_speech_and_kid_speaking', 'Chink_and_clink', 'Chirp_and_tweet', 'Church_bell', 'Clapping',
                     'Computer_keyboard', 'Crackle', 'Cricket', 'Crowd', 'Cupboard_open_or_close',
                     'Cutlery_and_silverware', 'Dishes_and_pots_and_pans', 'Drawer_open_or_close', 'Drip',
                     'Electric_guitar', 'Fart', 'Female_singing', 'Female_speech_and_woman_speaking',
                     'Fill_(with_liquid)', 'Finger_snapping', 'Frying_(food)', 'Gasp', 'Glockenspiel', 'Gong',
                     'Gurgling', 'Harmonica', 'Hi-hat', 'Hiss', 'Keys_jangling', 'Knock', 'Male_singing',
                     'Male_speech_and_man_speaking', 'Marimba_and_xylophone', 'Mechanical_fan', 'Meow',
                     'Microwave_oven', 'Motorcycle', 'Printer', 'Purr', 'Race_car_and_auto_racing', 'Raindrop', 'Run',
                     'Scissors', 'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard', 'Slam',
                     'Sneeze', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush',
                     'Traffic_noise_and_roadway_noise', 'Trickle_and_dribble', 'Walk_and_footsteps',
                     'Water_tap_and_faucet', 'Waves_and_surf', 'Whispering', 'Writing', 'Yell', 'Zipper_(clothing)']



myDataset = AudioDataset(datapath, csvFile, targetLabels)
print(myDataset)
print(len(myDataset))

## Get first element
audio = myDataset[0]


## Soudn file
#ipd.Audio(audios[0], rate=audio[4])

sd.play(audio[0], audio[3]) #this works
sd.wait()

#sd.play(myDataset[1234][0], myDataset[1234][3])
#sd.wait()

