import torch
import copy
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

        for epoch in range(epochs):
            for batch in dataloader:
                x,y = batch

                def closure():
                    optimizer.zero_grad()
                    model.train()

                    mean = model(x)
                    l = loss(mean.flatten(),y.flatten())

                    reg = sum([(param**2).sum() for param in model.parameters()]) #loss(means.flatten(), y_wo_noise.flatten(), vars.flatten()) 
                    l += lambda_ * reg

                    # adding l2 reg
                    #reg = sum([(param**2).sum() for param in model_i.parameters()]) #loss(means.flatten(), y_wo_noise.flatten(), vars.flatten()) 
                    #l += lambda_ * reg

                    l.backward()
                    
                    return l
                
                optimizer.step(closure)

    else:
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                model.train()
                x,y = batch
                mean = model(x)
                l = loss(mean.flatten(), y.flatten())
                reg = sum([(param**2).sum() for param in model.parameters()]) #loss(means.flatten(), y_wo_noise.flatten(), vars.flatten()) 
                l += lambda_ * reg
                l.backward()
                optimizer.step()


    return model

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

    #print("fitting a model using ")
    #print(loss)

    if batch_size == "Full":
        batch_size = len(x)
        shuffle = False
    
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

    if reinit is True:
        model = nn(**model_kwargs)

        if reinit_old:
            #print("reinitializing old way")
            utils.initialize_parameters_old(model)
        
    else:
        print("not reinitializing model")
        model = copy.deepcopy(nn)
    


    #print(model)
    #print(type(model))
    optimizer = solver(model.parameters(), **solver_kwargs)

    if isinstance(optimizer,torch.optim.LBFGS):

        batch_size = len(x)
        shuffle = False
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

        for epoch in range(epochs):
            for batch in dataloader:
                x,y = batch

                def closure():
                    optimizer.zero_grad()
                    model.train()

                    mean, var = model(x)


                    l = loss(mean.flatten(),y.flatten(), var.flatten())#,eps=eps)

                    """
                    if pretrain_mse:
                        #print("training pretrain")
                        lambda_dec = (epoch-1)/(epochs/2)
                        l = (1-lambda_dec) * loss_2(mean.flatten(),y.flatten())
                        l += lambda_dec * loss(mean.flatten(),y.flatten(), var.flatten(),eps=eps)
                    """

                    # adding l2 reg
                    reg = sum([(param**2).sum() for param in model.parameters()]) #loss(means.flatten(), y_wo_noise.flatten(), vars.flatten()) 
                    l += lambda_ * reg

                    l.backward()
                    
                    return l
                
                optimizer.step(closure)

    else:
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                model.train()
                x,y = batch
                mean, var = model(x)

                l = loss(mean.flatten(), y.flatten(), var.flatten())#,eps=eps)
                reg = sum([(param**2).sum() for param in model.parameters()]) #loss(means.flatten(), y_wo_noise.flatten(), vars.flatten()) 
                l += lambda_ * reg
                l.backward()
                optimizer.step()


    return model



def fit_deepensemble_NLL_sequential(x,y,nn,model_kwargs,solver,solver_kwargs,training_kwargs,p_subsample,n_ensemble,loss=NLL_loss):
    """fitiing of a deep ensemble using a mean, var loss, sequentially
    """
    
    mask = utils.get_subsample_mask(len(x),n_ensemble,p_subsample)
    models = []

    for i in range(n_ensemble):
        mask_i = mask[:,i]
        x_ens = torch.clone(x[mask_i,:])
        y_ens = torch.clone(y[mask_i])

        model = fit_model_NLL(x_ens,y_ens,nn,model_kwargs,solver,solver_kwargs,training_kwargs,loss)
        models.append(model)

    return ensembles.DeepEnsemble(ensembles=models), mask

def fit_deepensemble_mse_sequential(x,y,nn,model_kwargs,solver,solver_kwargs,training_kwargs,p_subsample,n_ensemble):
    """
    Fits a deep ensemble using MSE loss
    
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
        must be a loss that takes, mean and target as input




        """
    mask = utils.get_subsample_mask(len(x),n_ensemble,p_subsample)
    models = []

    for i in range(n_ensemble):
        mask_i = mask[:,i]
        x_ens = torch.clone(x[mask_i,:])
        y_ens = torch.clone(y[mask_i])

        model = fit_model_mse(x_ens,y_ens,nn,model_kwargs,solver,solver_kwargs,training_kwargs)
        models.append(model)

    return ensembles.DeepEnsemble_mean(ensembles=models), mask

def fit_shallowensemble_mse_sequential(x,y,nn,model_kwargs,solver,solver_kwargs,training_kwargs,p_subsample,n_ensemble,loss=torch.nn.functional.mse_loss):
    mask = utils.get_subsample_mask(len(x),n_ensemble,p_subsample)
    models = []

    
    model = fit_model_mse(x,y,nn,model_kwargs,solver,solver_kwargs,training_kwargs,loss)


    training_kwargs = copy.deepcopy(training_kwargs)
    training_kwargs["reinit"] = False

    reset_last = training_kwargs.get("reset_last",True)
    freeze_n = training_kwargs.get("freeze_n",0)



    for i in range(n_ensemble):
        mask_i = mask[:,i]
        
        model_i = copy.deepcopy(model)

        for n, param in enumerate(model_i.nn.parameters()):
            if n < freeze_n:
                #print("freezing a parameter")
                param.requires_grad = False

        model_i.reinitialize_n_last(len(list(model_i.nn.parameters()))-freeze_n)

        if reset_last:
            #print("reinit mean")
            model_i.reinitialize_mean()
            #todo: rewrite this with has_attribute
    
        x_ens = torch.clone(x[mask_i,:])
        y_ens = torch.clone(y[mask_i])

        model_i = fit_model_mse(x_ens,y_ens,model_i,model_kwargs,solver,solver_kwargs,training_kwargs,loss)
        models.append(model_i)

    return ensembles.DeepEnsemble_mean(ensembles=models), mask

def fit_shallowensemble_NLL_sequential(x,y,nn,model_kwargs,solver,solver_kwargs,training_kwargs,p_subsample,n_ensemble,loss=NLL_loss):
    mask = utils.get_subsample_mask(len(x),n_ensemble,p_subsample)
    models = []

    
    model = fit_model_NLL(x,y,nn,model_kwargs,solver,solver_kwargs,training_kwargs,loss)


    training_kwargs = copy.deepcopy(training_kwargs)
    training_kwargs["reinit"] = False

    reset_last = training_kwargs.get("reset_last",True)
    freeze_n = training_kwargs.get("freeze_n",0)



    for i in range(n_ensemble):
        mask_i = mask[:,i]
        
        model_i = copy.deepcopy(model)

        for n, param in enumerate(model_i.nn.parameters()):
            if n < freeze_n:
                #print("freezing a parameter")
                param.requires_grad = False

        model_i.reinitialize_n_last(len(list(model_i.nn.parameters()))-freeze_n)

        if reset_last:
            #print("reinit mean")
            model_i.reinitialize_mean()
            #todo: rewrite this with has_attribute
    
        x_ens = torch.clone(x[mask_i,:])
        y_ens = torch.clone(y[mask_i])

        model_i = fit_model_NLL(x_ens,y_ens,model_i,model_kwargs,solver,solver_kwargs,training_kwargs,loss)
        models.append(model_i)

    return ensembles.DeepEnsemble(ensembles=models), mask