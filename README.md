# DPOSE
Contains supplementary code and data for the \
"Uncertainty quantification by direct propagation of shallow ensembles" - paper
The repository is organized as follows:
- `./UCI_experiments/` contains the regression experiments on UCI datasets
- `./Atomistic_experiments/` contains the regression experiments on atomistic data
- `./Data/` contains additional DFT calculations that were generated for this work
- `./Plots_and_analysis/` contains the figures and analysis notebooks for this work

## UCI_experiments
This part of the repository is structures as follows:
- `data`: contains the data used in the experiments, for setup see the README.md in the directory\
and run the `load.py`script to download the UCI data
- `models`: contains the source code for the models and training routines
- `experiments`: contains the individual experiments using different architectures on the UCI datasets

## Data
This part of the repository is structures as follows:
- `Surfaces_DFT`: contains the DFT calculations of liquid water surfaces
- `CP2K_inputs`: contains the input files for the CP2K calculations
- `H2O_size_extensive_data`: contains the data for the experiments on MD snapshots of liquid water of increasing size

## Atomistic_experiments
This part of the repository is structures as follows:
- `data`: contains the data used in the experiments, run the `load.py` scripts in each subdirectory, to:
    - download the data, from various source (ie MaterialsCloud or github)
    - obtain train test splits
    - preprocess the data (only needed for liquid water data)
- `materials_model_predictions`: contains the predictions of the different models
- `trajectories_H2O.zip`: Unzip this file to obtain the trajectory data for the liquid water experiments

## Plots_and_analysis
This part of the repository is structures as follows:
- `figure 2-8`: Each subdirectory contains the notebooks to generate the respective figures from the paper
- `si_plot 1-3`: Each subdirectory contains the notebooks to generate the respective supplementary figures from the paper
- `table_materials`: Contains the notebook to generate table II from the paper
- `heat_capacity`: Contains the notebook to generate the heat capacity data from the paper

## Convergence_toy
This part of the repository is structures as follows:
Contains experiments of the convergence of MVE and shallow ensembles on a challenging toy problem
