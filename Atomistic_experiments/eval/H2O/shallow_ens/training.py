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

SEED = 0
random.seed(SEED)

frames_water_train = ase.io.read("../../../data/H2O/train_frames.xyz",":")
frames_water_val = ase.io.read("../../../data/H2O/validation_frames.xyz", ":")
frames_water_test = ase.io.read("../../../data/H2O/test_frames.xyz", ":")

#load extrapolation data
frames_surfaces = ase.io.read("../../../../Data/Surfaces_DFT/frames_surfaces.xyz",":")



# --- define the hypers ---
hypers_ps = {
    "cutoff": 5.,
    "max_radial": 5,
    "max_angular": 5,
    "atomic_gaussian_width": 0.25,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 6.0, "rate": 3.0, "scale": 2.0}}
}

hypers_rs = {
    "cutoff": 5.,
    "max_radial": 16,
    "atomic_gaussian_width": 0.25,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 6.0, "rate": 3.0, "scale": 2.0}}
}


# --- define calculator ---
calc_rs = rascaline.torch.SoapRadialSpectrum(**hypers_rs)
calc_ps = rascaline.torch.SoapPowerSpectrum(**hypers_ps)

# --- create the dataloader ---
dataloader = create_rascaline_dataloader(frames_water_train,
                                         energy_key="TotEnergy",
                                         forces_key="force",                                       
                                         calculators=[calc_rs,calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=4, 
                                         shuffle=True)

dataloader_val = create_rascaline_dataloader(frames_water_val,
                                         energy_key="TotEnergy",
                                         forces_key="force",                                       
                                         calculators=[calc_rs,calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_water_val), 
                                         shuffle=False)

dataloader_test = create_rascaline_dataloader(frames_water_test,
                                         energy_key="TotEnergy",
                                         forces_key="force",                                       
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_water_test), 
                                         shuffle=False)

dataloader_surfaces = create_rascaline_dataloader(frames_surfaces,
                                         energy_key="TotEnergy",
                                         forces_key="force",                                       
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_surfaces), 
                                         shuffle=False)

# --- create the trainer ---
# for now a batch of features is necessary 
# for correct weight init

#COPY YOUR WANDB API KEY HERE, or load it fromn a file

#read wandb api code from file

"""
print("train split:",id_train)
print("val split:", id_val)
print("test split:",id_test)
print("seed", SEED)
"""

feat, prop, syst = next(iter(dataloader))

transformer_e = CompositionTransformer()
transformer_e.fit(syst, prop)

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


tmp_frames = []
for i in range(1, 11):
    tmp_frames.append(frames_water_train[0] * (1,1,i))

ase.io.write("tmp_frames.xyz", tmp_frames)


dataloader_tmp = create_rascaline_dataloader(tmp_frames,
                                         energy_key="TotEnergy",
                                         forces_key="force",                                       
                                         calculators=[calc_rs,calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(tmp_frames), 
                                         shuffle=False)


dataloader = create_rascaline_dataloader(frames_water_train,
                                         energy_key="TotEnergy",
                                         forces_key="force",                                       
                                         calculators=[calc_rs,calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_water_train), 
                                         shuffle=False)


feat, prop, syst = next(iter(dataloader_tmp))
torch.save(prop.block(0).values, "tmp_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "tmp_forces.pt")

out_tmp = module(feat, syst)
torch.save(out_tmp.block(1).values, "tmp_pred_energy_var.pt")
out_tmp = module.calculate(feat, syst)

#out_tmp_M_energies = module.model.get_energy(feat, syst).block(0).values
#out_tmp_atomic_energies = module.model.get_atomic_energies(feat, syst).block(0).values
#out_tmp_atomic_forces  = module.model.get_committee_forces(feat, syst)

torch.save(out_tmp.block(0).values, "tmp_pred_energy.pt")
torch.save(out_tmp.block(0).gradient("positions").values, "tmp_pred_forces.pt")

#torch.save(out_tmp_M_energies, "tmp_M_pred_energy.pt")
#torch.save(out_tmp_atomic_energies,"tmp_M_pred_atomic_energies.pt" )
#torch.save(out_tmp_atomic_forces, "tmp_M_pred_atomic_forces.pt")

feat, prop, syst = next(iter(dataloader))
torch.save(prop.block(0).values, "train_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "train_forces.pt")

out_train = module(feat, syst)
torch.save(out_train.block(1).values, "train_pred_energy_var.pt")
out_train = module.calculate(feat, syst)
torch.save(out_train.block(0).values, "train_pred_energy.pt")
torch.save(out_train.block(0).gradient("positions").values, "train_pred_forces.pt")

#out_train_atomic_forces = module.model.get_committee_forces(feat, syst)
#out_train_atomic_energies = module.model.get_atomic_energies(feat, syst).block(0).values
#out_train_M_energies = module.model.get_energy(feat, syst).block(0).values

#torch.save(out_train_M_energies, "train_M_pred_energy.pt")
#torch.save(out_train_atomic_energies,"train_M_pred_atomic_energies.pt" )
#torch.save(out_train_atomic_forces, "train_M_pred_atomic_forces.pt")



feat, prop, syst = next(iter(dataloader_val))
torch.save(prop.block(0).values, "val_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "val_forces.pt")

out_val = module(feat, syst)
torch.save(out_val.block(1).values, "val_pred_energy_var.pt")
out_val = module.calculate(feat, syst)
torch.save(out_val.block(0).values, "val_pred_energy.pt")
torch.save(out_val.block(0).gradient("positions").values, "val_pred_forces.pt")

#out_val_atomic_forces = module.model.get_committee_forces(feat, syst)
#out_val_atomic_energies = module.model.get_atomic_energies(feat, syst).block(0).values
#out_val_M_energies = module.model.get_energy(feat, syst).block(0).values

#torch.save(out_val_M_energies, "val_M_pred_energy.pt")
#torch.save(out_val_atomic_energies,"val_M_pred_atomic_energies.pt" )
#torch.save(out_val_atomic_forces, "val_M_pred_atomic_forces.pt")

feat, prop, syst = next(iter(dataloader_test))
torch.save(prop.block(0).values, "test_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "test_forces.pt")

out_test = module(feat, syst)
torch.save(out_test.block(1).values, "test_pred_energy_var.pt")
out_test = module.calculate(feat, syst)
torch.save(out_test.block(0).values, "test_pred_energy.pt")
torch.save(out_test.block(0).gradient("positions").values, "test_pred_forces.pt")


#out_test_atomic_forces = module.model.get_committee_forces(feat, syst)
#out_test_atomic_energies = module.model.get_atomic_energies(feat, syst).block(0).values
#out_test_M_energies = module.model.get_energy(feat, syst).block(0).values

#torch.save(out_test_M_energies, "test_M_pred_energy.pt")
#torch.save(out_test_atomic_energies,"test_M_pred_atomic_energies.pt" )
#torch.save(out_test_atomic_forces, "test_M_pred_atomic_forces.pt")


feat, prop, syst = next(iter(dataloader_surfaces))
torch.save(prop.block(0).values, "surface_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "surface_forces.pt")

out_surface = module(feat, syst)
torch.save(out_surface.block(1).values, "surface_pred_energy_var.pt")
out_surface = module.calculate(feat, syst)
torch.save(out_surface.block(0).values, "surface_pred_energy.pt")
torch.save(out_surface.block(0).gradient("positions").values, "surface_pred_forces.pt")


#out_surface_atomic_forces = module.model.get_committee_forces(feat, syst)
#out_surface_atomic_energies = module.model.get_atomic_energies(feat, syst).block(0).values
#out_surface_M_energies = module.model.get_energy(feat, syst).block(0).values

#torch.save(out_surface_M_energies, "surface_M_pred_energy.pt")
#torch.save(out_surface_atomic_energies,"surface_M_pred_atomic_energies.pt" )
#torch.save(out_surface_atomic_forces, "surface_M_pred_atomic_forces.pt")
