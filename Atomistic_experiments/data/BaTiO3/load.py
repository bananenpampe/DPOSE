import zipfile
from pathlib import Path

path = Path(__file__).resolve().parent

import os
import requests
import ase.io
import random


# Base directory for datasets
DATASET_PATH = path 

def extract_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_PATH)
        print(f"Extracted {zip_path} into {DATASET_PATH}")

def download_file(url, filename):
    response = requests.get(url, allow_redirects=True)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")


def download_BaTiO3():
    url = "https://archive.materialscloud.org/record/file?filename=BaTiO3_MaterialsCloud.zip&record_id=1265"
    download_file(url, DATASET_PATH / "BaTiO3_MaterialsCloud.zip")
    extract_zip(DATASET_PATH / "BaTiO3_MaterialsCloud.zip")




if __name__ == "__main__":
    import random
    
    random.seed(0)

    #download_BaTiO3()

    frames = ase.io.read(DATASET_PATH / "BaTiO3_MaterialsCloud/MLframework/GAP/Training_set.xyz",":")

    for frame in frames:
        frame.calculator = None
    
    for i, atoms in enumerate(frames):
        atoms.set_calculator(None)

        if hasattr(atoms, 'calculator') and atoms.calculator is not None:
            atoms.calculator.results.clear()

        if 'stress' in atoms.info:
            del atoms.info['stress']
    
    random.shuffle(frames)

    frames_train = frames[:1200]
    frames_val = frames[1200:1300]
    frames_test = frames[1300:]

    ase.io.write("train.xyz",frames_train)
    ase.io.write("val.xyz",frames_val)
    ase.io.write("test.xyz",frames_test)

