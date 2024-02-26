import torch
from . import mlp

class DeepEnsemble(torch.nn.Module):


    """
    Constructs a Deep Ensemble from N*Mean_Var MLPs
    """
    
    def __init__(self, ensembles) -> None:
        super().__init__()

        #TODO: changed from older notebook
        self.ensembles = torch.nn.ModuleList(ensembles)

    def forward(self, x: torch.Tensor):

        
        out_pred = []
        out_var = []

        for ens in self.ensembles:
            mean, var = ens(x)
            out_pred.append(mean)
            out_var.append(var)

        out_pred = torch.cat(out_pred,dim=1)
        out_var = torch.cat(out_var,dim=1)

        mean_pred = torch.mean(out_pred,dim=1)

        var_epi = torch.var(out_pred,dim=1)
        var_alo = torch.mean(out_var,dim=1)
        
        var_tot = var_epi + var_alo

        return mean_pred, var_tot
    
    def report_w_scaler(self, x: torch.Tensor, scaler):
        """returns rescaled mean and variance with an sklearn like scaler
        """

        out_pred = []
        out_var = []

        for ens in self.ensembles:
            mean, var = ens(x)
            mean = scaler.inverse_transform(mean.detach())
            var = var.detach() * scaler.scale_ ** 2
            out_pred.append(torch.Tensor(mean))
            out_var.append(torch.Tensor(var))

        out_pred = torch.stack(out_pred)
        out_var = torch.stack(out_var)

        mean_pred = torch.mean(out_pred,dim=0)

        var_epi = torch.var(out_pred,dim=0)
        var_alo = torch.mean(out_var,dim=0)
        
        var_tot = var_epi + var_alo

        return mean_pred, var_tot
    
    def report_wo_cross_w_scaler(self, x: torch.Tensor, scaler):
        """depreceated
        """

        out_pred = []
        out_var = []

        for ens in self.ensembles:
            mean, var = ens(x)
            mean = scaler.inverse_transform(mean.detach())
            var = var.detach() * scaler.scale_ ** 2
            out_pred.append(torch.Tensor(mean))
            out_var.append(torch.Tensor(var))

        out_pred = torch.stack(out_pred)
        out_var = torch.stack(out_var)

        mean_pred = torch.mean(out_pred,dim=0)

        var_tot = torch.mean((out_var + out_pred**2),dim=0) - mean_pred**2

        return mean_pred, var_tot

    
    def report_epi_alo(self, x: torch.Tensor):
        """returns epistemic and aleatoric variance
        """

        out_pred = []
        out_var = []

        for ens in self.ensembles:
            mean, var = ens(x)
            out_pred.append(mean)
            out_var.append(var)

        out_pred = torch.cat(out_pred,dim=1)
        out_var = torch.cat(out_var,dim=1)

        mean_pred = torch.mean(out_pred,dim=1)

        var_epi = torch.var(out_pred,dim=1)
        var_alo = torch.mean(out_var,dim=1)
        
        var_tot = var_epi + var_alo

        return mean_pred, var_tot, var_epi, var_alo
    
    
    def report_epi_alo_wo_cross(self, x: torch.Tensor):
        """deprecated
        """

        out_pred = []
        out_var = []

        for ens in self.ensembles:
            mean, var = ens(x)
            out_pred.append(mean)
            out_var.append(var)

        out_pred = torch.cat(out_pred,dim=1)
        out_var = torch.cat(out_var,dim=1)

        mean_pred = torch.mean(out_pred,dim=1)
        var_alo = torch.mean(out_var,dim=1)
        var_epi = torch.mean(out_pred**2,dim=1) -  mean_pred**2

        var_tot = torch.mean((out_var + out_pred**2),dim=1) - mean_pred**2

        return mean_pred, var_tot, var_epi, var_alo
    
    def return_ensemble_predictions(self, x: torch.Tensor):
        """returns the individual ensemble predictions
        """

        out_pred = []
        out_var = []

        for ens in self.ensembles:
            mean, var = ens(x)
            out_pred.append(mean)
            out_var.append(var)
        
        out_pred = torch.cat(out_pred,dim=1)
        out_var = torch.cat(out_var,dim=1)

        return out_pred, out_var

class DeepEnsemble_mean(torch.nn.Module):
    """Constructs a Deep Ensemble from N*Mean MLPs
    """
    
    def __init__(self, ensembles) -> None:
        super().__init__()

        #TODO: changed from older notebook
        self.ensembles = torch.nn.ModuleList(ensembles)

    def forward(self, x: torch.Tensor):
        
        out_pred = []

        for ens in self.ensembles:
            mean = ens(x)
            out_pred.append(mean)

        out_pred = torch.stack(out_pred,dim=1)
        mean_pred = torch.mean(out_pred,dim=1)
        var_epi = torch.var(out_pred,dim=1)

        var_tot = var_epi

        return mean_pred, var_tot
    
    def return_ensemble_predictions(self, x: torch.Tensor):
        """returns the individual ensemble predictions
        """

        out_pred = []

        for ens in self.ensembles:
            mean = ens(x)
            out_pred.append(mean)
        
        out_pred = torch.cat(out_pred,dim=1)

        return out_pred
    
    def report_w_scaler(self, x: torch.Tensor, scaler):
        """returns rescaled mean and variance with an sklearn like scaler
        """

        out_pred = []

        for ens in self.ensembles:
            mean = ens(x)
            mean = scaler.inverse_transform(mean.detach())
            out_pred.append(torch.Tensor(mean))

        out_pred = torch.stack(out_pred)

        mean_pred = torch.mean(out_pred,dim=0)
        var_tot = torch.var(out_pred,dim=0)

        return mean_pred, var_tot
    


class Ensemble_MLP_mean(DeepEnsemble_mean):
    """ 
    Initializes a mean MLP with an arbitrary number of hidden layers and activation functions
    """

    def __init__(self, n_in, n_out, n_hidden, n_hidden_layers, activation, n_committee) -> None:
        super().__init__([])
        self.ensembles = torch.nn.ModuleList(
            [mlp.MLP_mean(n_in, n_out, n_hidden, n_hidden_layers, activation) for _ in range(n_committee)]
        )