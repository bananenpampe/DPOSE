import requests
import ase.io
import random

# URL of the file you want to download


url = 'https://raw.githubusercontent.com/BingqingCheng/ab-initio-thermodynamics-of-water/master/training-set/dataset_1593.xyz'

response = requests.get(url)

if response.status_code == 200:

    with open('dataset_1593.xyz', 'wb') as file:
        file.write(response.content)
    print('File downloaded successfully!')
else:
    print('Failed to download the file. Status code:', response.status_code)


# Load the downloaded file
frames = ase.io.read('dataset_1593.xyz', index=':')

# convert units from hartree to eV
# convert positions and forces from Bohr to Angstrom
# convert cell from Bohr to Angstrom

from ase.units import Bohr, Hartree
import numpy as np

print(Bohr, Hartree)

#works for cubic cell
for frame in frames:
    frame.cell *= Bohr
    frame.positions *= Bohr
    frame.arrays["force"] *= Hartree/Bohr
    frame.info['TotEnergy'] *= Hartree

    #round cell to 6 decimals to make it consitent with orig data
    c = np.around(frame.cell, 6)
    frame.cell = c

#write this identifier because frame :1000 are from FPS and 1000: 
#are from PIMD - to keep track of the frames
    
for n,frame in enumerate(frames):
    frame.info["CONVERTED_ID"] = n

ase.io.write("water_converted.xyz", frames)

# make train - val - test split


frames = ase.io.read("water_converted.xyz", index=":")

SEED = 0
random.seed(SEED)
random.shuffle(frames)

# select a subset of the frames
frames_water_train = frames[:1274]
frames_water_val = frames[1274:1433]
frames_water_test = frames[1433:]

ase.io.write("train_frames.xyz", frames_water_train)
ase.io.write("validation_frames.xyz", frames_water_val)
ase.io.write("test_frames.xyz", frames_water_test)
