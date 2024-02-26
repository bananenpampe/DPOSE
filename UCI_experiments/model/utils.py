import torch
import math

def get_subsample_mask(n_samples: int, n_ensemble: int, p_subsample:float) -> torch.Tensor:
    """returns a mask of shape (n_samples, n_ens) where a sample is chosen 
    with a probability of p_subsample
    """

    mask = torch.rand((n_samples,n_ensemble)) > (1-p_subsample)
    
    return mask

def gain_init(tensor,gain):
    """ multiplies initial tensor weights with a gain
    """
    #initializes the weights of a tensor with a gain
    k = math.sqrt(1/tensor.shape[-1])
    k *= gain
    torch.nn.init.uniform_(tensor, -k,k)

def initialize_parameters_old(model):
    """for reproducing papers before PyTorch 0.4.0
    loops over each module in a simple MLP and reinitializes the weights using xavier init
    """

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            # Initialize linear layer weights using the "old" initialization (Xavier initialization)
            torch.nn.init.xavier_normal_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
            #torch.nn.init.xavier_uniform_(module.bias)
            if module.bias is not None:
                # Initialize bias with zeros
                # ?? questionable
                torch.nn.init.zeros_(module.bias)