import pandas as pd
import numpy as np

from pathlib import Path

path = Path(__file__).resolve()



import os
import requests
from pathlib import Path

# Base directory for datasets
DATASET_PATH = path.parent / "datasets/"
# Create the directory if it doesn't exist
DATASET_PATH.mkdir(parents=True, exist_ok=True)

import zipfile
from scipy.io import arff

from io import StringIO


# Define the base directory for datasets

# Function to download file from a URL
def download_file(url, filename):
    response = requests.get(url, allow_redirects=True)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

def extract_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_PATH)
        print(f"Extracted {zip_path} into {DATASET_PATH}")


def download_kin8m():
    """Download the kin8m dataset from a URL and return it as a pandas DataFrame."""
    url = "https://www.openml.org/data/download/3626/dataset_2175_kin8nm.arff"

    response = requests.get(url)
    if response.status_code == 200:
        # Load the ARFF file content
        data, meta = arff.loadarff(StringIO(response.text))
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # save the dataframe as: ./datasets/kinm8_new.data
        df.to_csv(DATASET_PATH / "kinm8_new.data", index=False)
    else:
        print("Failed to download the dataset")
        return None

# Functions to load datasets
def download_yacht():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
    download_file(url, DATASET_PATH / "yacht_hydrodynamics.data")

def download_housing():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    download_file(url, DATASET_PATH / "housing.csv")

def download_concrete():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    download_file(url, DATASET_PATH / "Concrete_Data.xls")

def download_power():
    #https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
    # For ZIP files, you might need additional logic to unzip after downloading
    download_file(url, DATASET_PATH / "PowerPlant.zip")

def download_naval():
    url = "https://archive.ics.uci.edu/static/public/316/condition+based+maintenance+of+naval+propulsion+plants.zip"
    # This URL is just an example, replace it with the actual URL for the naval dataset
    download_file(url, DATASET_PATH / "navalplantmaintenance.zip")

def download_energy():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    download_file(url, DATASET_PATH / "energy_efficiency.xlsx")

def download_proteins():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
    download_file(url, DATASET_PATH / "proteins.csv")

def download_wine():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    download_file(url, DATASET_PATH / "winequality-red.csv")

def download_yearsMSD():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
    # For ZIP files, you might need additional logic to unzip after downloading
    download_file(url, DATASET_PATH / "YearPredictionMSD.txt.zip")

def load_yacht():
    """loads the yacht dataset
    available from:  https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics
    """
    sheet = pd.read_csv(DATASET_PATH / "yacht_hydrodynamics.data", header=None, delim_whitespace=True)
    dat = sheet.to_numpy()
    size_dat = dat.shape[-1] -1 
    X = dat[:,:size_dat]
    Y = dat[:,-1].reshape(-1,1)

    return X,Y, sheet

def load_housing():
    """loads the boson housing dataset
    from: https://github.com/selva86/datasets/blob/master/BostonHousing.csv
    """
    sheet = pd.read_csv(DATASET_PATH / "housing.csv",delimiter=r",",header=0)
    dat = sheet.to_numpy()
    size_dat = dat.shape[-1] -1 
    X = dat[:,:size_dat]
    Y = dat[:,-1].reshape(-1,1)

    return X,Y, sheet

def load_concrete():
    """loads the concrete dataset
    from: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
    """
    sheet = pd.read_excel(DATASET_PATH / "Concrete_Data.xls")
    dat = sheet.to_numpy()
    size_dat = dat.shape[-1] -1 
    X = dat[:,:size_dat]
    Y = dat[:,-1].reshape(-1,1)

    return X,Y, sheet

def load_power():
    """loads the power plant dataset
    from: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
    """
    sheet = pd.read_excel(DATASET_PATH / "PowerPlant.xlsx")
    dat = sheet.to_numpy()
    size_dat = dat.shape[-1] -1 
    X = dat[:,:size_dat]
    Y = dat[:,-1].reshape(-1,1)

    return X,Y, sheet

def load_naval():
    """loads the naval dataset
    from: https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants
    """
    sheet = pd.read_csv(DATASET_PATH / "navalplantmaintenance.csv", header=None, delim_whitespace=True)
    dat = sheet.to_numpy()
    size_dat = dat.shape[-1] -2
    X = dat[:,:size_dat]
    Y = dat[:,-2:].reshape(-1,2)

    return X,Y, sheet

def load_kin8m():
    """loads the kin8m dataset
    from: https://www.cs.toronto.edu/~delve/data/kin/desc.html
    """

    # Due to the download from the U toronto ftp server beeing a bit complicated
    # we will use the file that we downloaded in the download_kin8m function
    
    # using the original file it might be necessary to change the delimiter to r"\s+"
    # sheet = pd.read_csv("./datasets/kinm8_new.data", delimiter=r"\s+",header=0)

    sheet = pd.read_csv(DATASET_PATH / "kinm8_new.data", delimiter=r",",header=0)
    dat = sheet.to_numpy()
    size_dat = dat.shape[-1] -1
    X = dat[:,:size_dat]
    Y = dat[:,-1:].reshape(-1,1)

    assert X.shape[1] == 8, "The kin8m dataset should have 8 features, check the delimiter"

    return X,Y, sheet

def load_energy():
    """loads the energy efficiency dataset
    from: https://archive.ics.uci.edu/dataset/242/energy+efficiency
    """
    sheet = pd.read_excel(DATASET_PATH / "energy_efficiency.xlsx")
    dat = sheet.to_numpy()
    size_dat = dat.shape[-1] -2
    X = dat[:,:size_dat]
    Y = dat[:,-2:].reshape(-1,2)

    return X,Y, sheet

def load_proteins():
    """
    loads the protein dataset
    from:  https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure
    """
    sheet = pd.read_csv(DATASET_PATH / "proteins.csv",delimiter=r",")
    dat = sheet.to_numpy()
    X = dat[:,1:]
    Y = dat[:,0].reshape(-1,1)

    return X,Y, sheet

def load_wine():
    """
    loads the wine dataset
    from: https://archive.ics.uci.edu/dataset/186/wine+quality
    """
    sheet = pd.read_csv(DATASET_PATH / "winequality-red.csv",delimiter=r";")
    dat = sheet.to_numpy()
    size_dat = dat.shape[-1] -1
    X = dat[:,:size_dat]
    Y = dat[:,-1:].reshape(-1,1)

    return X,Y, sheet

def load_yearsMSD():

    """

    loads the years MSD songs dataset
    from: https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
    Additional Information

    You should respect the following train / test split:
    train: first 463,715 examples
    test: last 51,630 examples
    It avoids the 'producer effect' by making sure no song
    from a given artist ends up in both the train and test set.

    """

    sheet = pd.read_csv(DATASET_PATH / "YearPredictionMSD.txt",header=None)
    dat = sheet.to_numpy()
    size_dat = dat.shape[-1] -1
    X = dat[:,1:]
    Y = dat[:,0].reshape(-1,1)

    return X,Y, sheet


if __name__ == "__main__":
    # Download all datasets
    download_naval()
    download_yacht()
    download_housing()
    download_concrete()
    download_power()
    
    download_kin8m()
    download_energy()
    download_proteins()
    download_wine()
    download_yearsMSD()

    # Extract the PowerPlant.zip file
    extract_zip(DATASET_PATH / "PowerPlant.zip")
    #power plant will be extracted to ./datasets/CCPP/Folds5x2_pp.xlsx
    #move it into ./datasets/PowerPlant.xlsx
    os.rename(DATASET_PATH / "CCPP/Folds5x2_pp.xlsx", DATASET_PATH / "PowerPlant.xlsx")

    # Extract the YearPredictionMSD.txt.zip file
    extract_zip(DATASET_PATH / "YearPredictionMSD.txt.zip")

    # Extract the navalplantmaintenance.zip file
    extract_zip(DATASET_PATH / "navalplantmaintenance.zip")

    # will extract to ./UCI CBM Dataset/data.txt move the file navalplantmaintenance.csv
    os.rename(DATASET_PATH / "UCI CBM Dataset/data.txt", DATASET_PATH / "navalplantmaintenance.csv")

    # Download the kin8m dataset
    download_kin8m()


    #make sure that it works:
    #------------------------------------------------

    # Load the kin8m dataset
    load_kin8m()
    # Load the yacht dataset
    load_yacht()
    # Load the housing dataset
    load_housing()
    # Load the concrete dataset
    load_concrete()
    # Load the power plant dataset
    load_power()
    # Load the naval dataset
    load_naval()
    # Load the energy efficiency dataset
    load_energy()
    # Load the protein dataset
    load_proteins()
    # Load the wine dataset
    load_wine()
    # Load the years MSD songs dataset
    load_yearsMSD()
    # Load the kin8m dataset