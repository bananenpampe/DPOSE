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
import glob
import copy 

from transformer.composition import CompositionTransformer
from pytorch_lightning.callbacks import LearningRateMonitor
from nn.ensemble import DeepEnsemble

#default type is float64
torch.set_default_dtype(torch.float64)

# --- load the data ---

SEED = 0
random.seed(SEED)


ckpts = glob.glob("../../../training/BaTiO3/mse_ens/*/*/*/*/*.ckpt")

# --- define the hypers ---
frames_water_train = ase.io.read("../../../data/BaTiO3/train.xyz",":")
frames_water_val = ase.io.read("../../../data/BaTiO3/val.xyz", ":")
frames_water_test = ase.io.read("../../../data/BaTiO3/test.xyz", ":")


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

feat, prop, syst = next(iter(dataloader))



modules = []

for n, c in enumerate(ckpts):
    
    transformer_e = CompositionTransformer(multi_block=True)
    transformer_e.fit(syst, prop)

    checkpoint = torch.load(c)['state_dict']
    new_state_dict = {}
    
    for k, v in checkpoint.items():
        new_state_dict[k] = torch.tensor(v.clone().detach().numpy())  # Explicit new memory

    module = copy.deepcopy(BPNNRascalineModule(feat, None))
    module.energy_transformer = transformer_e
    module.energy_transformer.is_fitted = True

    module.load_state_dict(checkpoint)

    modules.append(module)


for module in modules:
    print(module.state_dict()["model.interaction.model.m_map.LabelsEntry(species_center=56).mean_out.weight"])


deep_ens = DeepEnsemble(modules, kind="mse-deep-ens")

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


#trainer.test(module, dataloaders=dataloader_test)

dataloader = create_rascaline_dataloader(frames_water_train,
                                         energy_key="energy",
                                         forces_key="forces",                                       
                                         calculators=[calc_rs,calc_ps],
                                         do_gradients=True,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_water_train), 
                                         shuffle=False)



feat, prop, syst = next(iter(dataloader))
torch.save(prop.block(0).values, "train_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "train_forces.pt")

E_pred, E_UQ, F_pred, F_UQ_epistemic = deep_ens.report_energy_forces(feat, syst)
torch.save(E_UQ, "train_pred_energy_var.pt")
torch.save(E_pred, "train_pred_energy.pt")
torch.save(F_pred, "train_pred_forces.pt")
torch.save(F_UQ_epistemic, "train_pred_forces_var.pt")


feat, prop, syst = next(iter(dataloader_val))
torch.save(prop.block(0).values, "val_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "val_forces.pt")

E_pred, E_UQ, F_pred, F_UQ_epistemic = deep_ens.report_energy_forces(feat, syst)
torch.save(E_UQ, "val_pred_energy_var.pt")
torch.save(E_pred, "val_pred_energy.pt")
torch.save(F_pred, "val_pred_forces.pt")
torch.save(F_UQ_epistemic, "val_pred_forces_var.pt")


feat, prop, syst = next(iter(dataloader_test))
torch.save(prop.block(0).values, "test_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "test_forces.pt")

E_pred, E_UQ, F_pred, F_UQ_epistemic = deep_ens.report_energy_forces(feat, syst)
torch.save(E_UQ, "test_pred_energy_var.pt")
torch.save(E_pred, "test_pred_energy.pt")
torch.save(F_pred, "test_pred_forces.pt")
torch.save(F_UQ_epistemic, "test_pred_forces_var.pt")
