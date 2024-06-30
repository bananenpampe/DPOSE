import torch
import copy
import sys

from . import utils
from . import ensembles


NLL_loss = lambda mu, y, var: torch.mean(torch.log(var)/2 + ((y-mu)**2)/(2*var))

def fit_model_mse(x,y,nn,model_kwargs,solver,solver_kwargs,training_kwargs,loss=torch.nn.functional.mse_loss):
    
    """"
    Fits a model using MSE loss
    
    Parameters
    ----------
    x : torch.Tensor
        Input data
    
    y : torch.Tensor
        Target data
    
    nn : torch.nn.Module
        Neural network to be fitted, or prefit model object
    
    model_kwargs : dict
        Keyword arguments to be passed to the neural network
        can be empty if nn is a prefit model

        contains:
        n_in : int
            Number of input features
        n_out : int
            Number of output predictions
        n_hidden : int
            Number of hidden units
        n_hidden_layers : int
            Number of hidden layers
        activation : torch.nn.functional
            Activation function to be used

    solver : torch.optim
        Optimizer to be used for training

    solver_kwargs : dict
        Keyword arguments to be passed to the optimizer

        contains:
        lr : float
            Learning rate
        
    training_kwargs : dict
        Keyword arguments to be passed to this training function

        contains:
        epochs : int
            Number of epochs to train for
        batch_size : int
            Batch size to be used for training
        l2_reg : float
            L2 regularization strength
        old : bool
            Whether to use the old initialization scheme
        reinit : bool
            Whether to reinitialize the model before training
        pretrain_mse : bool
            Whether to pretrain the model using MSE loss
            # not implemented yet
        eps : float
            Epsilon to be used for NLL loss
        gain : float
            Gain to for initialization
            # not implemented yet

        reset_last : bool
            Whether to reinitialize the last layer before training
        freeze_n : int
            Number of layers to freeze before training
            use to freeze n layers during training of a prefit MLP


    loss : torch.nn.functional
        Loss function to be used for training
    
    Returns
    -------
    torch.nn.Module
        Fitted model
    """

    dataset = torch.utils.data.TensorDataset(x,y)
    epochs = training_kwargs["epochs"]
    batch_size = training_kwargs["batch_size"]
    lambda_ = training_kwargs.get("l2_reg",0.)
    #gain = training_kwargs.get("gain",1.)
    reinit_old = training_kwargs.get("old",False)
    reinit = training_kwargs.get("reinit",True)

    loss_curve = []
    
    full_x = torch.clone(x)#.copy()
    full_y = torch.clone(y)#.copy()

    
    shuffle = True
    
    if reinit is True:
        model = nn(**model_kwargs)

        if reinit_old:
            #print("reinitializing old way")
            utils.initialize_parameters_old(model)
        
        
    else:
        model = copy.deepcopy(nn)

    if batch_size == "Full":
        batch_size = len(x)
        shuffle = False

    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    optimizer = solver(model.parameters(), **solver_kwargs)



    if isinstance(optimizer,torch.optim.LBFGS):
        raise ValueError()

    else:
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                model.train()
                x,y = batch
                mean = model(x)
                l = loss(mean.flatten(), y.flatten())
                #reg = sum([(param**2).sum() for param in model.parameters()]) #loss(means.flatten(), y_wo_noise.flatten(), vars.flatten()) 
                #l += lambda_ * reg
                l.backward()
                optimizer.step()
            
            if epoch % 25 == 0:
                loss_curve.append(float(torch.sqrt(torch.nn.functional.mse_loss(full_y, model(full_x))).detach()))

    return model, loss_curve

def fit_model_NLL(x,y,nn,model_kwargs,solver,solver_kwargs,training_kwargs,loss=NLL_loss):
    """
    Fits a model using NLL loss

    Parameters
    ----------
    x : torch.Tensor
        Input data
    y : torch.Tensor
        Target data
    nn : torch.nn.Module
        Neural network to be fitted, or prefit model object
    model_kwargs : dict
        Keyword arguments to be passed to the neural network
        can be empty if nn is a prefit model
        contains:
        n_in : int
            Number of input features
        n_out : int
            Number of output predictions
        n_hidden : int
            Number of hidden units
        n_hidden_layers : int
            Number of hidden layers
        activation : torch.nn.functional
            Activation function to be used

    solver : torch.optim
        Optimizer to be used for training
    solver_kwargs : dict
        Keyword arguments to be passed to the optimizer
        contains: 
        lr : float
            Learning rate
    training_kwargs : dict
        Keyword arguments to be passed to this training function
        contains:
        epochs : int
            Number of epochs to train for
        batch_size : int
            Batch size to be used for training
        l2_reg : float
            L2 regularization strength
        old : bool
            Whether to use the old initialization scheme
        reinit : bool
            Whether to reinitialize the model before training
        pretrain_mse : bool
            Whether to pretrain the model using MSE loss
            # not implemented yet
        eps : float
            Epsilon to be used for NLL loss
        gain : float
            Gain to for initialization
            # not implemented yet

    loss : torch.nn.functional
        Loss function to be used for training
        must be a loss that takes, mean and predicted var/uncertainty as input
    
    Returns
    -------
    torch.nn.Module
        Fitted model

    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.utils.data.TensorDataset(x,y)
    #loss = torch.nn.functional.gaussian_nll_loss
    loss_2 = torch.nn.functional.mse_loss

    epochs = training_kwargs["epochs"]
    batch_size = training_kwargs["batch_size"]
    lambda_ = training_kwargs.get("l2_reg",0.)
    shuffle = True
    reinit = training_kwargs.get("reinit",True)
    reinit_old = training_kwargs.get("old",False)
    pretrain_mse = training_kwargs.get("pretrain_mse",False)
    eps = training_kwargs.get("eps",0.)
    gain = training_kwargs.get("gain",1.)

    loss_curve = []

    #print("fitting a model using ")
    #print(loss)

    if batch_size == "Full":
        batch_size = len(x)
        shuffle = False
    
    full_x = torch.clone(x)#.copy()
    full_y = torch.clone(y)#.copy()

    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

    if reinit is True:
        model = nn(**model_kwargs)

        if reinit_old:
            #print("reinitializing old way")
            utils.initialize_parameters_old(model)
        
    else:
        print("not reinitializing model")
        model = copy.deepcopy(nn)
    
    #model.to(device)

    #print(model)
    #print(type(model))
    optimizer = solver(model.parameters(), **solver_kwargs)

    if isinstance(optimizer,torch.optim.LBFGS):
        raise ValueError()

    else:
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                model.train()
                x,y = batch
                #x, y = x.to(device), y.to(device)
                mean, var = model(x)

                l = loss(mean.flatten(), y.flatten(), var.flatten())#,eps=eps)
                #reg = sum([(param**2).sum() for param in model.parameters()]) #loss(means.flatten(), y_wo_noise.flatten(), vars.flatten()) 
                #l += lambda_ * reg
                l.backward()
                optimizer.step()
            
            if epoch % 25 == 0:
                loss_curve.append(float(torch.sqrt(torch.nn.functional.mse_loss(full_y, model(full_x)[0])).detach()))


    return model, loss_curve


def get_toy():
    x = torch.linspace(0, 12, 1000)
    y = 0.4*torch.sin(2*torch.pi*x) + torch.normal(0, 0.01, (1000,))

    return x.reshape(-1,1), y.reshape(-1,1)

def get_toy_linear():
    x = torch.linspace(0, 12, 1000)
    y = x + torch.normal(0, 1, (1000,))

    return x.reshape(-1,1), y.reshape(-1,1)
