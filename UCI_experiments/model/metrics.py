import torch

def NLL(input: torch.Tensor, target: torch.Tensor, var: torch.Tensor, full=True, eps=1e-06):
    return torch.nn.functional.gaussian_nll_loss(input.flatten(), target.flatten(), var.flatten(), full=full,eps=eps)

def MAE(input, target):
    return torch.mean(torch.abs((input.flatten() - target.flatten())))


def RMSE(input, target):
    return torch.sqrt(torch.mean((input.flatten() - target.flatten())**2))

def MSE(input, target):
    return torch.mean((input.flatten() - target.flatten())**2)

def get_coeff(input: torch.Tensor, target: torch.Tensor, var: torch.Tensor, eps=1e-08) -> torch.tensor:
    """ Returns dimensionless NLL coefficient
    """
    
    mse = MSE(input,target)
    uncertainty_estimate = (input.flatten() - target.flatten())**2
    
    LL_best = torch.nn.functional.gaussian_nll_loss(input.flatten(), target.flatten(), uncertainty_estimate.flatten(), full=False, eps=eps)
    
    LL_worst_case_best_RMSE = torch.nn.functional.gaussian_nll_loss\
        (input.flatten(), target.flatten(), torch.ones_like(var.flatten())*mse, full=False, eps=eps)
    
    LL_actual = torch.nn.functional.gaussian_nll_loss(input.flatten(), target.flatten(), var.flatten(), full=False, eps=eps)
    
    coeff = 1/( LL_best - LL_worst_case_best_RMSE) * (LL_actual - LL_worst_case_best_RMSE) * 100

    return coeff



def report_NLL_model(model, Xtest, target):
    """ Returns NLL, dimless_coeff, RMSE
    """
    model.eval()
    mean, var = model(Xtest)

    nll = NLL(mean,target,var)
    dimless_coeff = get_coeff(mean,target,var)
    rmse = RMSE(mean,target)
    
    return nll, dimless_coeff, rmse