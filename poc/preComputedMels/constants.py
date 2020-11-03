#!/usr/bin/env python3
import os

## #########################################################################
## Constants
## ########################################################################

class Constants(object):
    __mode__ = 0
    #DATA = '/Volumes/scratch/work/falconr1/audioTaggingPKG/data'

    @property
    def mode(self):
        return self.__mode__

    @mode.setter
    def mode(self, value):
        self.__mode__ = value

    def __init__(self, mode=0):
        self.__mode__ = mode
        if self.__mode__ == 0:  # Macbook
            self.DATA = '/Users/falconr1/Documents/myData'
            self.DATA = '/Volumes/scratch/work/falconr1/audioTaggingPKG/data'
            self.DATA = '/Volumes/scratch/work/falconr1/datasets/freesound/raw/'

            self.WORK = '/Volumes/scratch/work/falconr1/audioTagging2019/work'
        elif self.__mode__ == 1:  # Kaggle
            self.DATA = '../input/'
        elif self.__mode__ == 2:  # Triton
            self.DATA = '/scratch/work/falconr1/audioTaggingPKG/data/'
            self.DATA = '/scratch/work/falconr1/datasets/freesound/raw/'

            self.WORK = '/scratch/work/falconr1/audioTagging2019/work'

        elif self.__mode__ == 3:  # CS-181
            self.DATA = '/m/cs/work/falconr1/audioTaggingPKG/data/'
            self.DATA = '/m/cs/work/falconr1/datasets/freesound/raw/'

            self.WORK = '/m/cs/work/falconr1/audioTagging2019/work'
        elif self.__mode__ == 4: # taito-gpu or taito at CSC
            self.DATA = '/wrk/falconpe/DONOTREMOVE/datasets/freesound'

            self.WORK = '/homeappl/home/falconpe/audiotagging2019/work'
        else:
            print("Mode should be 0, 1, 2, 3, or 4")
            raise Exception("Mode should be 0, 1, 2, 3, or 4")

        self.CSV_TRN_CURATED = os.path.join(self.DATA, 'train_curated.csv')
        self.CSV_TRN_NOISY = os.path.join(self.DATA, 'train_noisy.csv')
        self.CSV_SUBMISSION = os.path.join(self.DATA, 'sample_submission.csv')
        self.TRN_CURATED = os.path.join(self.DATA, 'train_curated')
        self.TRN_NOISY = os.path.join(self.DATA, 'train_noisy')
        self.TEST = os.path.join(self.DATA, 'test')

        self.WORK_SPECS = ('specs')
        self.WORK_LABELS = ('labels')
        self.IMG_TRN_CURATED = os.path.join(self.WORK, 'image/trn_curated')
        self.IMG_TRN_NOISY = os.path.join(self.WORK, 'image/trn_curated')
        self.IMG_TEST = os.path.join(self.WORK, 'image/test')

    target_labels = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark',
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

