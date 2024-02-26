import sys
sys.path.append("../../")

from model.utils_modified import get_toy, fit_model_NLL, fit_model_mse
from data import load
from model.mlp import MLP_mean, MLP_mean_var, MLP_var_by_ensemble
import torch
import numpy as np
from model.loss import CPRS

import matplotlib.pyplot as plt
import argparse
# Create the parser
parser = argparse.ArgumentParser(description='parse')
# Add the '-n' argument
parser.add_argument('-n', type=int, help='number of epochs')
# Parse the arguments
args = parser.parse_args()
# Retrieve the value of '-n'
EPOCHS = args.n



SEED = 0
torch.manual_seed(SEED)
rng = np.random.RandomState(SEED)

x, y = get_toy()


NN_KWARGS_SHALLOW = dict(n_in=x.shape[-1], 
                         n_out=y.shape[-1],
                         n_hidden=128,
                         n_hidden_layers=2,
                         activation=torch.nn.Tanh)

TRAINING_KWARGS = dict(epochs=EPOCHS,
                       batch_size=100,
                       l2_reg=0.,
                       old=False,)

SOLVER_KWARGS = dict(lr=5e-4)

model_mse, loss_curve = fit_model_mse(x,y,nn=MLP_mean,\
                    model_kwargs=NN_KWARGS_SHALLOW,solver=torch.optim.Adam,\
                    solver_kwargs=SOLVER_KWARGS,training_kwargs=TRAINING_KWARGS)

ypred = model_mse(x)
ypred = ypred.detach().numpy()

np.save("ypred.npy", ypred)
np.save("loss_curve", np.array(loss_curve))

