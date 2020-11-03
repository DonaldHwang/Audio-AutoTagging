import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
import os
import sounddevice as sd

## Room definition
corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T  # [x,y]
room = pra.Room.from_corners(corners)


room = pra.ShoeBox([9, 7.5, 3.5], fs=44100, absorption=0.35, max_order=17)

fig, ax = room.plot()
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 4])

#room = pra.Room.from_corners(corners)
#room.extrude(2.8)

fig, ax = room.plot()
ax.set_xlim([0, 5])
ax.set_ylim([0, 3])
ax.set_zlim([0, 2])




#Create a 4 by 6 metres shoe box room
room = pra.ShoeBox([4,6])

# Add a source somewhere in the room
room.add_source([2.5, 4.5])

R = np.c_[
    [6.3, 4.87, 1.2],  # mic 1
    [6.3, 4.93, 1.2],  # mic 2
    ]

# the fs of the microphones is the same as the room
mic_array = pra.MicrophoneArray(R, room.fs)

# finally place the array in the room
room.add_microphone_array(mic_array)



## Add source
# specify signal source
fs = 44100
signal = np.zeros((1, fs*3))
signal[0] = 1
#fs, signal = wavfile.read("arctic_a0010.wav")

# add source to 2D room
#room = pra.Room.from_corners(corners, fs=fs)
my_source = pra.SoundSource([1, 1, 1], signal=signal, delay=1.3)
room.add_source(my_source)

fig, ax = room.plot()


## Add receiver
R = pra.circular_2D_array(center=[2.,2.], M=6, phi0=0, radius=0.1)
room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

fig, ax = room.plot()



# compute image sources
room.image_source_model(use_libroom=True)


room.plot_rir()
fig = plt.gcf()
fig.set_size_inches(20, 10)
plt.savefig(os.path.join('./' 'RIR.png'))
fig.savefig((os.path.join('./' 'RIR.png')))


##
room.compute_rir()
sd.play(room.rir[0], fs)

room.simulate()
print(room.mic_array.signals.shape)

