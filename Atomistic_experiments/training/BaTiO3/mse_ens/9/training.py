import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",  "..", "..", "utils"))

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from rascaline_trainer import BPNNRascalineModule
from load import load_PBE0_TS
from dataset.dataset import create_rascaline_dataloader
import rascaline
import rascaline.torch
import torch
import ase.io
import torch._dynamo
import traceback as tb
import random

from transformer.composition import CompositionTransformer
from pytorch_lightning.callbacks import LearningRateMonitor


#default type is float64


torch.set_default_dtype(torch.float64)

# --- load the data ---

# shuffle the frames
SEED = 9
random.seed(SEED)
torch.manual_seed(SEED)

frames_water_train = ase.io.read("../../train.xyz",":")
frames_water_val = ase.io.read("../../val.xyz", ":")
frames_water_test = ase.io.read("../../test.xyz", ":")

"""
{
    "structure_filename": "Training_set.xyz",
    "soap_hypers": {
        "soap_type": "PowerSpectrum",
        "interaction_cutoff": 5.5,
        "cutoff_smooth_width": 0.5,
        "cutoff_function_type": "RadialScaling",
        "cutoff_function_parameters": {
            "rate": 1,
            "scale": 3.5,
            "exponent": 4
        },
        "max_radial": 8,
        "max_angular": 6,
        "gaussian_sigma_type": "Constant",
        "gaussian_sigma_constant": 0.5,
        "radial_basis": "GTO",
        "optimization_args": {
            "type": "Spline",
            "accuracy": 1.0E-5
        },
        "normalize": true,
        "compute_gradients": true
    },


"""



# --- define the hypers ---
hypers_ps = {
    "cutoff": 5.5,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.50,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 4.0, "rate": 1.0, "scale": 3.5}}
}

hypers_rs = {
    "cutoff": 5.5,
    "max_radial": 16,
    "atomic_gaussian_width": 0.50,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 4.0, "rate": 1.0, "scale": 3.5}}
}


# --- define calculator ---
calc_rs = rascaline.torch.SoapRadialSpectrum(**hypers_rs)
calc_ps = rascaline.torch.SoapPowerSpectrum(**hypers_ps)

# --- create the dataloader ---
dataloader_init = create_rascaline_dataloader(frames_water_train,
                                         energy_key="energy",
                                         forces_key="forces",                                       
                                         calculators=[calc_rs],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_water_train), 
                                         shuffle=False)

dataloader = create_rascaline_dataloader(frames_water_train,
                                         energy_key="energy",
                                         forces_key="forces",                                       
                                         calculators=[calc_rs,calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=4, 
                                         shuffle=True)

dataloader_val = create_rascaline_dataloader(frames_water_val,
                                         energy_key="energy",
                                         forces_key="forces",                                       
                                         calculators=[calc_rs,calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_water_val), 
                                         shuffle=False)

dataloader_test = create_rascaline_dataloader(frames_water_test,
                                         energy_key="energy",
                                         forces_key="forces",                                       
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_water_test), 
                                         shuffle=False)

# --- create the trainer ---
# for now a batch of features is necessary 
# for correct weight init

#COPY YOUR WANDB API KEY HERE, or load it fromn a file

#read wandb api code from file
wandb_api_key = "YOUR_API"

wandb.login(key=wandb_api_key)
wandb_logger = WandbLogger(name=str(SEED), project="BaTiO3-mse-ens", log_model=True)
wandb_logger.experiment.config["key"] = wandb_api_key

# log the descriptor hyperparameters
wandb_logger.log_hyperparams({"hypers radial spectrum": hypers_rs})
wandb_logger.log_hyperparams({"hypers power spectrum": hypers_ps})

print("seed", SEED)

feat, prop, syst = next(iter(dataloader_init))

transformer_e = CompositionTransformer()
transformer_e.fit(syst, prop)

feat, prop, syst = next(iter(dataloader))

# define the trainer
module = BPNNRascalineModule(feat, transformer_e)

print(module)

#compiled_model = torch.compile(module,fullgraph=True )
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(max_epochs=2000,
                  precision=64,
                  accelerator="cpu",
                  logger=wandb_logger,
                  callbacks=[lr_monitor],
                  gradient_clip_val=0.5,
                  enable_progress_bar=False,
                  val_check_interval=1.0,
                  check_val_every_n_epoch=1,
                  inference_mode=False,
                  max_time="00:48:00:00")
                  #profiler="simple")

trainer.fit(module, dataloader, dataloader_val)
trainer.test(module, dataloaders=dataloader_test)
