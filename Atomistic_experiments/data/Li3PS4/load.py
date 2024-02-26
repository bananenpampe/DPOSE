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


def download_LiPS():
    url = "https://archive.materialscloud.org/record/file?filename=MaterialsCloudArchive-LiPS.zip&record_id=2019"
    download_file(url, DATASET_PATH / "MaterialsCloudArchive-LiPS.zip")
    extract_zip(DATASET_PATH / "MaterialsCloudArchive-LiPS.zip")




if __name__ == "__main__":
    import random
    
    random.seed(0)

    download_LiPS()

    frames = ase.io.read(DATASET_PATH / "MaterialsCloudArchive/training-sets/PBEsol_final_dataset.extxyz",":2300")
    frames_test = ase.io.read(DATASET_PATH / "MaterialsCloudArchive/training-sets/PBEsol_final_dataset.extxyz","2300:")

    random.shuffle(frames)
    n_train = int(len(frames)*0.95 //1)
    frames_train = frames[:n_train]
    frames_val = frames[n_train:]

    ase.io.write(DATASET_PATH / "LiPS_train.xyz",frames_train)
    ase.io.write(DATASET_PATH / "LiPS_val.xyz",frames_val)
    ase.io.write(DATASET_PATH / "LiPS_test.xyz",frames_test)