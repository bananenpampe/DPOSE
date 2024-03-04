import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "BPNN_model", "H2O", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "BPNN_model", "H2O", "utils"))

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from rascaline_trainer import BPNNRascalineModule
 
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
SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)
#torch.use_deterministic_algorithms(True)

frames_water_train = ase.io.read("../../../../data/Li3PS4/LiPS_train.xyz",":")
frames_water_val = ase.io.read("../../../../data/Li3PS4/LiPS_val.xyz", ":")
frames_water_test = ase.io.read("../../../../data/Li3PS4/LiPS_test.xyz", ":")



# --- define the hypers ---
hypers_ps = {
    "cutoff": 5.,
    "max_radial": 5,
    "max_angular": 5,
    "atomic_gaussian_width": 0.30,
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
    "cutoff": 5.,
    "max_radial": 16,
    "atomic_gaussian_width": 0.30,
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






# log the descriptor hyperparameters



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

trainer = Trainer(max_epochs=500,
                  precision=64,
                  accelerator="cpu",
                  logger=None,
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
