import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "BPNN_model", "H2O", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "BPNN_model", "H2O", "utils"))

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
"""
frames_water = ase.io.read("../../data/water_converted.xyz", index=":")

for n, frame in enumerate(frames_water):
    frame.info["CONVERTED_ID"] = n

# shuffle the frames

random.shuffle(frames_water)

# select a subset of the frames
frames_water_train = frames_water[:1274]
frames_water_val = frames_water[1274:1433]
frames_water_test = frames_water[1433:]
"""

SEED = 0
random.seed(SEED)

# --- define the hypers ---
frames_water_train = ase.io.read("../../../data/BaTiO3/train.xyz",":")
frames_water_val = ase.io.read("../../../data/BaTiO3/val.xyz", ":")
frames_water_test = ase.io.read("../../../data/BaTiO3/test.xyz", ":")

   

for frame in frames_water_train:
    frame.calculator = None
    frame.info.pop("stress", None)

for frame in frames_water_val:
    frame.calculator = None
    frame.info.pop("stress", None)

for frame in frames_water_test:
    frame.calculator = None
    frame.info.pop("stress", None)

ase.io.write("all.xyz",frames_water_train + frames_water_val + frames_water_test) 

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
                                         batch_size=len(frames_water_train), 
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
#

#
#wandb_logger = WandbLogger(name="eval-run-1", project="LiPS", log_model=True)
#

# log the descriptor hyperparameters
#
#

"""
print("train split:",id_train)
print("val split:", id_val)
print("test split:",id_test)
print("seed", SEED)
"""

feat, prop, syst = next(iter(dataloader))

transformer_e = CompositionTransformer(multi_block=True)
transformer_e.fit(syst, prop)

# define the trainer
module = BPNNRascalineModule(feat, transformer_e)

print(module)

#compiled_model = torch.compile(module,fullgraph=True )
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(max_epochs=500,
                  precision=64,
                  accelerator="cpu",
                  callbacks=[lr_monitor],
                  gradient_clip_val=100,
                  enable_progress_bar=False,
                  val_check_interval=1.0,
                  check_val_every_n_epoch=1,
                  inference_mode=False,
                  max_time="00:160:00:00")
                  #profiler="simple")

#trainer.fit(module, dataloader, dataloader_val)
#trainer.test(module, dataloaders=dataloader_test)

checkpoint = torch.load("example.ckpt")['state_dict']
module.load_state_dict(checkpoint)
module.energy_transformer.is_fitted = True



dataloader = create_rascaline_dataloader(frames_water_train,
                                         energy_key="energy",
                                         forces_key="forces",                                       
                                         calculators=[calc_rs,calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_water_train), 
                                         shuffle=True)




feat, prop, syst = next(iter(dataloader))
torch.save(prop.block(0).values, "train_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "train_forces.pt")

out_train = module(feat, syst)
torch.save(out_train.block(1).values, "train_pred_energy_var.pt")
out_train = module.calculate(feat, syst)
torch.save(out_train.block(0).values, "train_pred_energy.pt")
torch.save(out_train.block(0).gradient("positions").values, "train_pred_forces.pt")


feat, prop, syst = next(iter(dataloader_val))
torch.save(prop.block(0).values, "val_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "val_forces.pt")

out_val = module(feat, syst)
torch.save(out_val.block(1).values, "val_pred_energy_var.pt")
out_val = module.calculate(feat, syst)
torch.save(out_val.block(0).values, "val_pred_energy.pt")
torch.save(out_val.block(0).gradient("positions").values, "val_pred_forces.pt")

feat, prop, syst = next(iter(dataloader_test))
torch.save(prop.block(0).values, "test_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "test_forces.pt")

out_test = module(feat, syst)
torch.save(out_test.block(1).values, "test_pred_energy_var.pt")
out_test = module.calculate(feat, syst)
torch.save(out_test.block(0).values, "test_pred_energy.pt")
torch.save(out_test.block(0).gradient("positions").values, "test_pred_forces.pt")

