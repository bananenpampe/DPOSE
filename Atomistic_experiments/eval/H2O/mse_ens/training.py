import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "BPNN_model", "H2O", "utils"))

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
import glob
import copy 

from transformer.composition import CompositionTransformer
from pytorch_lightning.callbacks import LearningRateMonitor
from nn.ensemble import DeepEnsemble

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

frames_water_train = ase.io.read("../64_rs_ps_4_batch_correct_w_UQ/train_frames.xyz",":")
frames_water_val = ase.io.read("../64_rs_ps_4_batch_correct_w_UQ/validation_frames.xyz", ":")
frames_water_test = ase.io.read("../64_rs_ps_4_batch_correct_w_UQ/test_frames.xyz", ":")

frames_surf_train = ase.io.read("/home/kellner/BPNN_packages/H2O/run_surface/run_1/frames_surface_train.xyz",":")
frames_surf_val = ase.io.read("/home/kellner/BPNN_packages/H2O/run_surface/run_1/frames_surface_val.xyz",":")
frames_surf_test = ase.io.read("/home/kellner/BPNN_packages/H2O/run_surface/run_1/frames_surface_test.xyz",":")

frames_surfaces = frames_surf_train + frames_surf_val + frames_surf_test

ckpts = glob.glob("/home/kellner/BPNN_packages/H2O/example/shallow_ens_commitee_CPRS/*/*/*/*/*.ckpt")

"""
id_train = []
id_val = []
id_test = []

for frame in frames_water_train:
    id_train.append(frame.info["CONVERTED_ID"])

for frame in frames_water_val:
    id_val.append(frame.info["CONVERTED_ID"])

for frame in frames_water_test:
    id_test.append(frame.info["CONVERTED_ID"])
"""

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
wandb_api_key = "YOUR_API"

wandb.login(key=wandb_api_key)
wandb_logger = WandbLogger(project="H2O-sr",log_model=True)
wandb_logger.experiment.config["key"] = wandb_api_key

# log the descriptor hyperparameters
wandb_logger.log_hyperparams({"hypers radial spectrum": hypers_rs})
wandb_logger.log_hyperparams({"hypers power spectrum": hypers_ps})

"""
print("train split:",id_train)
print("val split:", id_val)
print("test split:",id_test)
print("seed", SEED)
"""

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
    print(module.state_dict()["model.interaction.model.m_map.LabelsEntry(species_center=8).mean_out.weight"])


deep_ens = DeepEnsemble(modules, kind="deep-ens")

#compiled_model = torch.compile(module,fullgraph=True )
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(max_epochs=500,
                  precision=64,
                  accelerator="cpu",
                  logger=wandb_logger,
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

E_pred, E_UQ,  E_UQ_aleatoric, E_UQ_epistemic, F_pred, F_UQ_epistemic = deep_ens.report_energy_forces(feat, syst, report_aleatoric=True)
torch.save(E_UQ, "tmp_pred_energy_var.pt")
out_tmp = module.calculate(feat, syst)
torch.save(E_pred, "tmp_pred_energy.pt")
torch.save(E_UQ_aleatoric, "tmp_pred_energ_var_aleatoric.pt")
torch.save(E_UQ_epistemic, "tmp_pred_energ_var_epistemic.pt")
torch.save(F_pred, "tmp_pred_forces.pt")
torch.save(F_UQ_epistemic, "tmp_pred_forces_var.pt")


feat, prop, syst = next(iter(dataloader))
torch.save(prop.block(0).values, "train_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "train_forces.pt")

E_pred, E_UQ,  E_UQ_aleatoric, E_UQ_epistemic, F_pred, F_UQ_epistemic = deep_ens.report_energy_forces(feat, syst, report_aleatoric=True)
torch.save(E_UQ, "train_pred_energy_var.pt")
torch.save(E_pred, "train_pred_energy.pt")
torch.save(F_pred, "train_pred_forces.pt")
torch.save(E_UQ_aleatoric, "train_pred_energ_var_aleatoric.pt")
torch.save(E_UQ_epistemic, "train_pred_energ_var_epistemic.pt")
torch.save(F_UQ_epistemic, "train_pred_forces_var.pt")


feat, prop, syst = next(iter(dataloader_val))
torch.save(prop.block(0).values, "val_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "val_forces.pt")

E_pred, E_UQ,  E_UQ_aleatoric, E_UQ_epistemic, F_pred, F_UQ_epistemic = deep_ens.report_energy_forces(feat, syst, report_aleatoric=True)
torch.save(E_UQ, "val_pred_energy_var.pt")
torch.save(E_pred, "val_pred_energy.pt")
torch.save(F_pred, "val_pred_forces.pt")
torch.save(E_UQ_aleatoric, "val_pred_energ_var_aleatoric.pt")
torch.save(E_UQ_epistemic, "val_pred_energ_var_epistemic.pt")
torch.save(F_UQ_epistemic, "val_pred_forces_var.pt")


feat, prop, syst = next(iter(dataloader_test))
torch.save(prop.block(0).values, "test_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "test_forces.pt")

E_pred, E_UQ,  E_UQ_aleatoric, E_UQ_epistemic, F_pred, F_UQ_epistemic = deep_ens.report_energy_forces(feat, syst, report_aleatoric=True)
torch.save(E_UQ, "test_pred_energy_var.pt")
torch.save(E_pred, "test_pred_energy.pt")
torch.save(F_pred, "test_pred_forces.pt")
torch.save(E_UQ_aleatoric, "test_pred_energ_var_aleatoric.pt")
torch.save(E_UQ_epistemic, "test_pred_energ_var_epistemic.pt")
torch.save(F_UQ_epistemic, "test_pred_forces_var.pt")

feat, prop, syst = next(iter(dataloader_surfaces))
torch.save(prop.block(0).values, "surfaces_energy.pt")
torch.save(prop.block(0).gradient("positions").values, "surfaces_forces.pt")

E_pred, E_UQ,  E_UQ_aleatoric, E_UQ_epistemic, F_pred, F_UQ_epistemic = deep_ens.report_energy_forces(feat, syst, report_aleatoric=True)
torch.save(E_UQ, "surfaces_pred_energy_var.pt")
torch.save(E_pred, "surfaces_pred_energy.pt")
torch.save(F_pred, "surfaces_pred_forces.pt")
torch.save(E_UQ_aleatoric, "surfaces_pred_energ_var_aleatoric.pt")
torch.save(E_UQ_epistemic, "surfaces_pred_energ_var_epistemic.pt")
torch.save(F_UQ_epistemic, "surfaces_pred_forces_var.pt")