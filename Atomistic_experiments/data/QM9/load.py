# contains loading files for different water files

import os
import re
import tqdm
import ase.io
import numpy as np
from ase import Atoms

import sys
import ase.io
import random

# everything below is from and with permission from (Sergey Pozdnyakov)
# ------------- from: https://github.com/lab-cosmo/nice/blob/master/examples/qm9_small.ipynb !!! ------------


PROPERTIES_NAMES = ['tag', 'index', 'A', 'B', 'C', 'mu',
                    'alpha', 'homo', 'lumo', 'gap', 'r2',
                    'zpve', 'U0', 'U', 'H', 'G', 'Cv']


def string_to_float(element):
    '''because shit like 2.1997*^-6 happens'''
    return float(element.replace('*^', 'e'))

PROPERTIES_HANDLERS = [str, int] + [string_to_float] * (len(PROPERTIES_NAMES) - 2)

def parse_qm9_xyz(path):
    with open(path, 'r') as f:
        lines = list(f)
    #print(lines)

    #MODIFICATION TO ADD INCHI KEY
    inchi_ids = lines[-1].rstrip("\n").split("\t")

    assert len(inchi_ids) == 2

    n_atoms = int(lines[0])
    properties = {name:handler(value)
                  for handler, name, value in zip(PROPERTIES_HANDLERS,
                                            PROPERTIES_NAMES,
                                            lines[1].strip().split())}
    composition = ""
    positions = []
    for i in range(2, 2 + n_atoms):
        composition += lines[i].strip().split()[0]
        positions.append([string_to_float(value) 
                          for value in lines[i].strip().split()[1:4]])
        
    
    positions = np.array(positions)
    result = Atoms(composition, positions = np.array(positions))
    result.info.update(properties)
    result.info['inchi_key_0'] = inchi_ids[0]
    result.info['inchi_key_1'] = inchi_ids[1]

    return result

def parse_index(path):
    with open(path, "r") as f:
        lines = list(f)
    proper_lines = lines[9:-1]
    result = [int(line.strip().split()[0]) for line in proper_lines]
    return np.array(result, dtype = int)

def download_qm9(clean = True):
    #downloading from https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
    os.system("wget https://ndownloader.figshare.com/files/3195389 -O qm9_main.xyz.tar.bz2")
    os.system("wget https://ndownloader.figshare.com/files/3195404 -O problematic_index.txt")
    os.system("mkdir qm9_main_structures")
    os.system("tar xjf qm9_main.xyz.tar.bz2 -C qm9_main_structures")
    
    names = [name for name in os.listdir('qm9_main_structures/') if name.endswith('.xyz')]
    names = sorted(names)
    
    structures = [parse_qm9_xyz('qm9_main_structures/{}'.format(name))
              for name in tqdm.tqdm(names)]
    
    problematic_index = parse_index('problematic_index.txt')
    np.save('problematic_index.npy', problematic_index)
    ase.io.write('qm9_main.extxyz', structures)
    if (clean):
        os.system("rm -r qm9_main_structures")
        os.system("rm problematic_index.txt")
        os.system("rm qm9_main.xyz.tar.bz2")
    return structures, problematic_index
              
def get_qm9(clean = True):
    if ('qm9_main.extxyz' in os.listdir('.')) and \
              ('problematic_index.npy' in os.listdir('.')):
        structures = ase.io.read('qm9_main.extxyz', index = ':')
        problematic_index = np.load('problematic_index.npy')
        return structures, problematic_index
    else:
        return download_qm9(clean = clean)
    

def get_qm9_w_problematic(clean=True):
    
    structures, problematic_index = get_qm9(clean=clean)
    
    for structure in structures:
        if structure.info['index'] in problematic_index:
            structure.info['problematic'] = "PROBLEMATIC"
        else:
            structure.info['problematic'] = "OK"
        
    return structures

# ------------- our loading routine ------------

if __name__ == "__main__":

    SEED = 0
    random.seed(SEED)

    frames = get_qm9_w_problematic()
    ase.io.write("qm9.xyz", frames)

    frames = ase.io.read("qm9.xyz", index=":")

    frames_filtered = []
    frames_problematic = []

    for frame in frames:
        if frame.info['problematic'] == "OK":
            frames_filtered.append(frame)
        elif frame.info['problematic'] == "PROBLEMATIC":
            frames_problematic.append(frame)
        else:
            raise ValueError("problematic value not recognized")

    print("Number of frames: ", len(frames))
    print("Number of filtered frames: ", len(frames_filtered))
    print("Number of problematic frames: ", len(frames_problematic))

    random.shuffle(frames_filtered)

    frames_train = frames_filtered[:100000]
    frames_val = frames_filtered[100000:110000]
    frames_test = frames_filtered[120000:]

    print("Number of training frames: ", len(frames_train))
    print("Number of validation frames: ", len(frames_val))
    print("Number of test frames: ", len(frames_test))

    ase.io.write("qm9_train.xyz", frames_train)
    ase.io.write("qm9_val.xyz", frames_val)
    ase.io.write("qm9_test.xyz", frames_test)

    ase.io.write("qm9_problematic.xyz", frames_problematic)


