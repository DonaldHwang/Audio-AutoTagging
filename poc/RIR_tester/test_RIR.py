import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile

import pyroomacoustics as pra

fs = 16000
# room dimension
room_dim = [5, 4, 6]

# Create the shoebox
room = pra.ShoeBox(
    room_dim,
    absorption=0.05,
    fs=fs,import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile

import pyroomacoustics as pra

fs = 16000
# room dimension
room_dim = [10, 4, 15]

# Create the shoebox
room = pra.ShoeBox(
    room_dim,
    absorption=0.05,
    fs=fs,
    max_order=100,
    )

# source and mic locations
room.add_source([2, 3.1, 2])
room.add_microphone_array(
        pra.MicrophoneArray(
            np.array([[2, 1.5, 2]]).T,
            room.fs)
        )

# Run the image source model
room.image_source_model()

# Plot the result up to fourth order images
room.plot(img_order=4)

plt.figure()
room.plot_rir()
plt.show()
#sd.play(room.rir[0][0],room.fs)

my_rir = room.rir[0][0] / np.max(np.abs(room.rir[0][0]))
sd.play(my_rir, fs)
wavfile.write("my_rir.wav", fs, my_rir)
    max_order=100,
    )

# source and mic locations
room.add_source([2, 3.1, 2])
room.add_microphone_array(
        pra.MicrophoneArray(
            np.array([[2, 1.5, 2]]).T,
            room.fs)
        )

# Run the image source model
room.image_source_model()

# Plot the result up to fourth order images
room.plot(img_order=4)

plt.figure()
room.plot_rir()
plt.show()
#sd.play(room.rir[0][0],room.fs)

my_rir = room.rir[0][0] / np.max(np.abs(room.rir[0][0]))
sd.play(my_rir, fs)
wavfile.write("my_rir.wav", fs, my_rir)
