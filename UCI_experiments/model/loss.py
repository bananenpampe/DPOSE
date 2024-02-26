import torch

def CPRS(means: torch.Tensor, targets: torch.Tensor, vars: torch.Tensor) -> torch.Tensor:
    """ Computes the CRPS of a gaussian distribution and
      a mean and uncertainty estimate

    means: torch.Tensor
        shape (N_samples, N_outputs)
    targets: torch.Tensor
        shape (N_samples, N_outputs)
    vars: torch.Tensor
        shape (N_samples, N_outputs)

    """

    sigma = torch.sqrt(vars)
    norm_x = ( targets - means)/sigma
    # torch.tensor(ndtr(norm_x.numpy())) 
    cdf =   0.5 * (1 + torch.erf(norm_x / torch.sqrt(torch.tensor(2))))

    normalization = 1 / (torch.sqrt(torch.tensor(2.0*torch.pi)))

    pdf = normalization * torch.exp(-(norm_x ** 2)/2.0)
    
    crps = sigma * (norm_x * (2*cdf-1) + 2 * pdf - 1/(torch.sqrt(torch.tensor(torch.pi))))

    #since it is a loss we return a negative value

    return torch.mean(crps)
