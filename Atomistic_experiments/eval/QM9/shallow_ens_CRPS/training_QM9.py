import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..",  "utils"))

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from rascaline_trainer_uncertainty_QM9 import BPNNRascalineModule
from load import load_PBE0_TS
from dataset.dataset import create_rascaline_dataloader
from dataset.dataset_helpers import get_global_unique_species
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
SEED = 0
random.seed(SEED)


# select a subset of the frames
frames_train = ase.io.read("/home/kellner/BPNN_packages/H2O/data/qm9_train.xyz",":")[:2000]
frames_val = ase.io.read("/home/kellner/BPNN_packages/H2O/data/qm9_val.xyz",":")[:2000]
frames_test = ase.io.read("/home/kellner/BPNN_packages/H2O/data/qm9_test.xyz",":")



# --- define the hypers ---
hypers_ps = {
    "cutoff": 5.,
    "max_radial": 9,
    "max_angular": 9,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 7.0, "rate": 1.0, "scale": 2.0}}
}

hypers_rs = {
    "cutoff": 5.,
    "max_radial": 16,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width":0.5},
    },
    "radial_scaling":{"Willatt2018": {"exponent": 7.0, "rate": 1.0, "scale": 2.0}}
}


# --- define calculator ---
calc_rs = rascaline.torch.SoapRadialSpectrum(**hypers_rs)
calc_ps = rascaline.torch.SoapPowerSpectrum(**hypers_ps)

# --- create the dataloader ---
dataloader_init = create_rascaline_dataloader(frames_train,
                                         energy_key="U0",                                       
                                         calculators=[calc_rs],
                                         do_gradients=False,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_train), 
                                         shuffle=True)

dataloader = create_rascaline_dataloader(frames_train,
                                         energy_key="U0",                                       
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients=False,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_train), 
                                         shuffle=True)

dataloader_val = create_rascaline_dataloader(frames_val,
                                         energy_key="U0",                                       
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients=False,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_val), 
                                         shuffle=False)

dataloader_test = create_rascaline_dataloader(frames_test,
                                         energy_key="U0",                                       
                                         calculators=[calc_rs, calc_ps],
                                         do_gradients=False,
                                         precompute = True,
                                         lazy_fill_up = False,
                                         batch_size=len(frames_test), 
                                         shuffle=False)

# --- create the trainer ---
# for now a batch of features is necessary 
# for correct weight init

#COPY YOUR WANDB API KEY HERE, or load it fromn a file

#read wandb api code from file
wandb_api_key = "YOUR_API"

wandb.login(key=wandb_api_key)
wandb_logger = WandbLogger(project="learn-QM9-eval",log_model=True)
wandb_logger.experiment.config["key"] = wandb_api_key

# log the descriptor hyperparameters
wandb_logger.log_hyperparams({"hypers radial spectrum": hypers_rs})
wandb_logger.log_hyperparams({"hypers power spectrum": hypers_ps})

print("seed", SEED)

feat, prop, syst = next(iter(dataloader_init))

transformer_e = CompositionTransformer(multi_block=True)
transformer_e.fit(syst, prop)





all_spec = dataloader_init.dataset.all_species

expected_keys = []
for spec in all_spec:
    expected_keys.append("LabelsEntry(species_center={})".format(spec))

expected_keys = set(expected_keys)

for n, (feat, _, _) in enumerate(dataloader):
    tmp_keys = []
    for key, t_block in feat.items():
        tmp_keys.append(str(key))    
    if set(tmp_keys) == expected_keys:
        NO_FULL = False
        break 

# define the trainer
module = BPNNRascalineModule(feat, transformer_e)

#compiled_model = torch.compile(module,fullgraph=True )
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(max_epochs=500,
                  precision=64,
                  accelerator="cpu",
                  logger=wandb_logger,
                  callbacks=[lr_monitor],
                  gradient_clip_val=0.5,
                  enable_progress_bar=False,
                  val_check_interval=1.0,
                  check_val_every_n_epoch=1,
                  inference_mode=False)
                  #profiler="simple")

checkpoint = torch.load("example.ckpt")['state_dict']
module.load_state_dict(checkpoint)
module.energy_transformer.is_fitted = True


feat, prop, syst = next(iter(dataloader))
out = module.calculate(feat, syst)
Ytrain_pred = out.block(0).values.detach()
Ytrain_var_pred = out.block(1).values.detach()
Ytrain_true = prop.block(0).values.detach()

torch.save(Ytrain_pred, "Ytrain_pred.pt")
torch.save(Ytrain_var_pred, "Ytrain_var_pred.pt")
torch.save(Ytrain_true, "Ytrain_true.pt")


feat, prop, syst = next(iter(dataloader_val))
out = module.calculate(feat, syst)
Yval_pred = out.block(0).values.detach()
Yval_var_pred = out.block(1).values.detach()
Yval_true = prop.block(0).values.detach()

torch.save(Yval_pred, "Yval_pred.pt")
torch.save(Yval_var_pred, "Yval_var_pred.pt")
torch.save(Yval_true, "Yval_true.pt")

feat, prop, syst = next(iter(dataloader_test))
out = module.calculate(feat, syst)
Ytest_pred = out.block(0).values.detach()
Ytest_var_pred = out.block(1).values.detach()
Ytest_true = prop.block(0).values.detach()

torch.save(Ytest_pred, "Ytest_pred.pt")
torch.save(Ytest_var_pred, "Ytest_var_pred.pt")
torch.save(Ytest_true, "Ytest_true.pt")
