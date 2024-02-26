import sys
sys.path.append("../../")
from data import load
from model.fitting import fit_deepensemble_mse_sequential,fit_model_NLL, fit_deepensemble_NLL_sequential
from model.mlp import MLP_mean, MLP_mean_var
from model.ensembles import Ensemble_MLP_mean
import torch
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from model.metrics import NLL, get_coeff
import numpy as np
from model.rescaling import MultiRescaler

SEED = 0
torch.manual_seed(SEED)
rng = np.random.RandomState(SEED)
torch.use_deterministic_algorithms(True)

rmses = []
nlls = []

all_data = {"concrete":load.load_concrete,"kin8m":load.load_kin8m,
            "energy":load.load_energy,"housing":load.load_housing,
            "naval":load.load_naval, "power":load.load_power,
            "wine":load.load_wine, "yacht": load.load_yacht,
           "years":load.load_yearsMSD, "protein":load.load_proteins}

#all_data = {"years":load.load_yearsMSD} #"kin8m":load.load_kin8m} #"concrete":load.load_concrete}#"years":load.load_yearsMSD,}#"naval":load.load_naval,"yacht": load.load_yacht}

with open("results_NLL.txt","w") as fg:
    fg.write("Experiment: all NLL \n")

for name, load_f in all_data.items():
    special_split = False
    n_split = 20
    n_hidden = 50
    lr=0.01
    
    if name == "protein":
        n_split = 20
        n_hidden = 100

    elif name in ["years"]:
        special_split = True
        n_hidden = 100
        n_split = 5
        lr=0.001

    nlls = []
    nlls_rescaled = []
    nlls_w_alpha = []
    coeffs = []
    coeffs_w_alpha = []
    rmses = []

    print("doing {}n splits".format(n_split))

    for i in range(n_split):
        #print("fitting: {} ... ".format(i))
        X,Y, sheet = load_f()
        size_dat = X.shape[1]

        if special_split:
            #following the producer problem in the years dataset 
            Xtrain_u = np.copy(X[:463715])
            Xtest_u = np.copy(X[-51630:])
            Ytrain = np.copy(Y[:463715])
            Ytest = np.copy(Y[-51630:])
            Xval_u, Xtest_u, Yval, Ytest = model_selection.train_test_split(Xtest_u,Ytest,test_size=0.5,shuffle=False,random_state=rng)

        else:
            Xtrain_u, Xtest_u, Ytrain, Ytest = model_selection.train_test_split(X,Y,test_size=0.2,random_state=rng)
            Xval_u, Xtest_u, Yval, Ytest = model_selection.train_test_split(Xtest_u,Ytest,test_size=0.5,random_state=rng)
        
        scaler = StandardScaler()
        scaler.fit(Xtrain_u)
        Xtrain = scaler.transform(Xtrain_u)
        Xtrain = torch.Tensor(Xtrain)
        Xval = torch.Tensor(scaler.transform(Xval_u))
        Xtest = torch.Tensor(scaler.transform(Xtest_u))

        scaler_Y = StandardScaler()
        Ytrain = scaler_Y.fit_transform(Ytrain)
        Ytrain = torch.Tensor(Ytrain)
        Yval = torch.Tensor(Yval)
        Ytest = torch.Tensor(Ytest)

        print(Ytrain.shape, Yval.shape, Ytest.shape)

        NN_KWARGS = dict(n_in=size_dat, n_out=Ytrain.shape[-1], n_hidden=n_hidden,n_hidden_layers=1,activation=torch.nn.ReLU, n_committee=5)
        SOLVER_KWARGS = dict(lr=lr)
        TRAINING_KWARGS = dict(epochs=40,batch_size=100,l2_reg=0.,old=False,eps=0.)

        # fit_model_NLL(x,y,nn,model_kwargs,solver,solver_kwargs,training_kwargs):
        model_mse = fit_model_NLL(Xtrain,Ytrain,nn=Ensemble_MLP_mean ,\
                    model_kwargs=NN_KWARGS,solver=torch.optim.Adam,\
                    solver_kwargs=SOLVER_KWARGS,training_kwargs=TRAINING_KWARGS)

        model_mse.eval()
        Ypred_train, Ypred_train_var = model_mse.report_w_scaler(Xtrain,scaler_Y)
        Ypred_val, Ypred_val_var = model_mse.report_w_scaler(Xval,scaler_Y)
        Ypred_test, Ypred_test_var = model_mse.report_w_scaler(Xtest,scaler_Y)

        Ytrain_np = Ytrain.detach().numpy()
        Ypred_train_np = Ypred_train.detach().numpy()
        Ypred_train_var_np = Ypred_train_var.detach().numpy()

        Yval_np = Yval.detach().numpy()
        Ypred_val_np = Ypred_val.detach().numpy()
        Ypred_val_var_np = Ypred_val_var.detach().numpy()

        z_val_split = torch.abs(Ypred_val-Yval)
        alpha = torch.mean(z_val_split**2/Ypred_val_var, dim=0)

        scaler_multi = MultiRescaler(n_properties=Ytrain.shape[-1])
        scaler_multi.fit(Ypred_val_np,Ypred_val_var_np,Yval_np)

        Ypred_test_var_iso_rescaled = torch.tensor(scaler_multi.predict(Ypred_test_var.detach().numpy()))

        nlls.append(NLL(Ypred_test,Ytest,Ypred_test_var,eps=0.,full=True).detach())
        nlls_w_alpha.append(NLL(Ypred_test,Ytest,Ypred_test_var*alpha,eps=0.,full=True).detach())
        nlls_rescaled.append(NLL(Ypred_test,Ytest,Ypred_test_var_iso_rescaled,eps=0.,full=True).detach()) #full = True, 
        coeffs.append(get_coeff(Ypred_test,Ytest,Ypred_test_var).detach())
        coeffs_w_alpha.append(get_coeff(Ypred_test,Ytest,Ypred_test_var*alpha).detach())
        rmses.append(torch.sqrt(torch.nn.functional.mse_loss(Ypred_test.flatten(),Ytest.flatten())))

    rmses = torch.stack(rmses)
    nlls = torch.stack(nlls)
    nlls_w_alpha = torch.stack(nlls_w_alpha)
    nlls_rescaled = torch.stack(nlls_rescaled)
    coeffs = torch.stack(coeffs)
    coeffs_w_alpha = torch.stack(coeffs_w_alpha)

    with open("results_NLL.txt","a") as fg:
        fg.write("{}  rmse: {:.2f} +- {:.2f} \n".format(name, float(torch.mean(rmses)), float(torch.std(rmses))))
        fg.write("{}  nlls: {:.2f} +- {:.2f} \n".format(name, float(torch.mean(nlls)), float(torch.std(nlls))))
        fg.write("{}  nlls rescaled w. alpha: {:.2f} +- {:.2f} \n".format(name, float(torch.mean(nlls_w_alpha)), float(torch.std(nlls_w_alpha))))
        fg.write("{}  nlls rescaled w. iso: {:.2f} +- {:.2f} \n".format(name, float(torch.mean(nlls_rescaled)), float(torch.std(nlls_rescaled))))
        fg.write("{}  coeffs: {:.2f} +- {:.2f} \n".format(name, float(torch.mean(coeffs)), float(torch.std(coeffs))))
        fg.write("{}  coeffs rescaled w. alpha: {:.2f} +- {:.2f} \n".format(name, float(torch.mean(coeffs_w_alpha)), float(torch.std(coeffs_w_alpha))))
