import torch
import math

#class MLPsimple()

class MLP_var_by_ensemble(torch.nn.Module):
    """ Initializes a mean and var MLP with an arbitrary number of hidden layers and activation functions
    """

    def __init__(self, n_in, n_out, n_hidden, n_hidden_layers, activation, n_linear_out) -> None:
        super().__init__()

        modules = []

        self.n_hidden = n_hidden
        self.n_linear_out = n_linear_out
        self.n_out = n_out

        if n_hidden_layers > 0:
            
            modules.append(torch.nn.Linear(n_in,n_hidden))
            modules.append(activation())
        else:
            modules.append(torch.nn.Linear(n_in,n_hidden))
        
        
        for n in range(n_hidden_layers-1):
            modules.append(torch.nn.Linear(n_hidden,n_hidden))
            modules.append(activation())

        self.nn = torch.nn.Sequential(
                *modules
                )
        
        self.mean_var_out = torch.nn.Linear(n_hidden,n_out*n_linear_out)

        

        for module in self.nn:
            if isinstance(module, torch.nn.Linear):
                # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                #torch.nn.init.xavier_uniform_(module.bias)
                if module.bias is not None:
                    # Initialize bias with zeros
                    torch.nn.init.zeros_(module.bias)
        
  
        
        torch.nn.init.xavier_uniform_(self.mean_var_out.weight)
        torch.nn.init.zeros_(self.mean_var_out.bias)



    #TRY THIS:
    def reinitialize_n_last(self, n):
        """ 
        The function reinitializes the last n layers of the MLP with Xavier initialization.
        """
        for n_h, module in enumerate(reversed(list(self.nn))):
            if n_h < n:
                if isinstance(module, torch.nn.Linear):
                    #print("reinit weight")
                    # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                    torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                    #torch.nn.init.xavier_uniform_(module.bias)
                    if module.bias is not None:
                        #print("reinit bias")
                        # Initialize bias with zeros
                        # ?? questionable:
                        torch.nn.init.zeros_(module.bias)
    

    def reinitialize_mean(self):
        """
        The function reinitializes the mean layer of the MLP with Xavier initialization."""
        torch.nn.init.xavier_uniform_(self.mean_var_out.weight)
        torch.nn.init.zeros_(self.mean_var_out.bias)

    
    def reinitialize_n_first(self, n):
        """
        The function reinitializes the first n layers of the MLP with Xavier initialization.
        """

        for n_h, module in enumerate(list(self.nn)):
            if n_h < n:
                if isinstance(module, torch.nn.Linear):
                    # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                    torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                    #torch.nn.init.xavier_uniform_(module.bias)
                    if module.bias is not None:
                        # Initialize bias with zeros
                        # ?? questionable
                        torch.nn.init.zeros_(module.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hidden = self.nn(x)
        out = self.mean_var_out(x_hidden)

        mean = torch.mean(out.reshape(-1,self.n_out,self.n_linear_out),dim=2)
        var = torch.var(out.reshape(-1,self.n_out,self.n_linear_out),dim=2)

        return mean, var
    
    def report_w_scaler(self, x: torch.Tensor, scaler):
        """ Reports the mean and variance of the model with a given scaler
        """

        self.eval()
        mean, var = self.forward(x)
        mean = scaler.inverse_transform(mean.detach())
        var = var.detach() * scaler.scale_ ** 2

        return torch.tensor(mean), torch.tensor(var)
    
    def report_w_mean_var(self, x: torch.Tensor, mean, std):
        """ Reports the mean and variance of the model with a given scaler
        """

        self.eval()
        mean, var = self.forward(x)
        mean = mean.detach().numpy() + mean
        var = var.detach().numpy() * std ** 2

        return torch.tensor(mean), torch.tensor(var)

class MLP_mean_var_pred_err(torch.nn.Module):
    """ Initializes a mean and var MLP with an arbitrary number of hidden layers and activation functions
    """

    def __init__(self, n_in, n_out, n_hidden, n_hidden_layers, activation) -> None:
        super().__init__()

        modules = []

        self.n_hidden = n_hidden
        self.n_out = n_out

        if n_hidden_layers > 0:
            
            modules.append(torch.nn.Linear(n_in,n_hidden))
            modules.append(activation())
        else:
            modules.append(torch.nn.Linear(n_in,n_hidden))
        
        
        for n in range(n_hidden_layers-1):
            modules.append(torch.nn.Linear(n_hidden,n_hidden))
            modules.append(activation())

        self.nn = torch.nn.Sequential(
                *modules
                )
        
        self.mean_var_out = torch.nn.Linear(n_hidden,n_out*2)
        

        for module in self.nn:
            if isinstance(module, torch.nn.Linear):
                # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                #torch.nn.init.xavier_uniform_(module.bias)
                if module.bias is not None:
                    # Initialize bias with zeros
                    torch.nn.init.zeros_(module.bias)
        
  
        
        torch.nn.init.xavier_uniform_(self.mean_var_out.weight)
        torch.nn.init.zeros_(self.mean_var_out.bias)



    #TRY THIS:
    def reinitialize_n_last(self, n):
        """ 
        The function reinitializes the last n layers of the MLP with Xavier initialization.
        """
        for n_h, module in enumerate(reversed(list(self.nn))):
            if n_h < n:
                if isinstance(module, torch.nn.Linear):
                    #print("reinit weight")
                    # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                    torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                    #torch.nn.init.xavier_uniform_(module.bias)
                    if module.bias is not None:
                        #print("reinit bias")
                        # Initialize bias with zeros
                        # ?? questionable:
                        torch.nn.init.zeros_(module.bias)
    

    def reinitialize_mean(self):
        """
        The function reinitializes the mean layer of the MLP with Xavier initialization."""
        torch.nn.init.xavier_uniform_(self.mean_var_out.weight)
        torch.nn.init.zeros_(self.mean_var_out.bias)

    
    def reinitialize_n_first(self, n):
        """
        The function reinitializes the first n layers of the MLP with Xavier initialization.
        """

        for n_h, module in enumerate(list(self.nn)):
            if n_h < n:
                if isinstance(module, torch.nn.Linear):
                    # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                    torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                    #torch.nn.init.xavier_uniform_(module.bias)
                    if module.bias is not None:
                        # Initialize bias with zeros
                        # ?? questionable
                        torch.nn.init.zeros_(module.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hidden = self.nn(x)
        #print(x_hidden.shape)
        #mean = self.mean_out(x_hidden)
        #var = self.var_out(x_hidden)

        out = self.mean_var_out(x_hidden)
        mean = out[:,:(self.n_out*2)//2]
        var = out[:,(self.n_out*2)//2:]

        var = torch.nn.functional.softplus(var) ** 2

        return mean, var
    
    def report_w_scaler(self, x: torch.Tensor, scaler):
        """ Reports the mean and variance of the model with a given scaler
        """

        self.eval()
        mean, var = self.forward(x)
        mean = scaler.inverse_transform(mean.detach())
        var = var.detach() * scaler.scale_ ** 2

        return torch.tensor(mean), torch.tensor(var)
    
    def report_w_mean_var(self, x: torch.Tensor, mean, std):
        """ Reports the mean and variance of the model with a given scaler
        """

        self.eval()
        mean, var = self.forward(x)
        mean = mean.detach().numpy() + mean
        var = var.detach().numpy() * std ** 2

        return torch.tensor(mean), torch.tensor(var)

class MLP_mean_var(torch.nn.Module):
    """ Initializes a mean and var MLP with an arbitrary number of hidden layers and activation functions
    """

    def __init__(self, n_in, n_out, n_hidden, n_hidden_layers, activation) -> None:
        super().__init__()

        modules = []

        self.n_hidden = n_hidden
        self.n_out = n_out

        if n_hidden_layers > 0:
            
            modules.append(torch.nn.Linear(n_in,n_hidden))
            modules.append(activation())
        else:
            modules.append(torch.nn.Linear(n_in,n_hidden))
        
        
        for n in range(n_hidden_layers-1):
            modules.append(torch.nn.Linear(n_hidden,n_hidden))
            modules.append(activation())

        self.nn = torch.nn.Sequential(
                *modules
                )
        
        self.mean_var_out = torch.nn.Linear(n_hidden,n_out*2)
        

        for module in self.nn:
            if isinstance(module, torch.nn.Linear):
                # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                #torch.nn.init.xavier_uniform_(module.bias)
                if module.bias is not None:
                    # Initialize bias with zeros
                    torch.nn.init.zeros_(module.bias)
        
  
        
        torch.nn.init.xavier_uniform_(self.mean_var_out.weight)
        torch.nn.init.zeros_(self.mean_var_out.bias)



    #TRY THIS:
    def reinitialize_n_last(self, n):
        """ 
        The function reinitializes the last n layers of the MLP with Xavier initialization.
        """
        for n_h, module in enumerate(reversed(list(self.nn))):
            if n_h < n:
                if isinstance(module, torch.nn.Linear):
                    #print("reinit weight")
                    # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                    torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                    #torch.nn.init.xavier_uniform_(module.bias)
                    if module.bias is not None:
                        #print("reinit bias")
                        # Initialize bias with zeros
                        # ?? questionable:
                        torch.nn.init.zeros_(module.bias)
    

    def reinitialize_mean(self):
        """
        The function reinitializes the mean layer of the MLP with Xavier initialization."""
        torch.nn.init.xavier_uniform_(self.mean_var_out.weight)
        torch.nn.init.zeros_(self.mean_var_out.bias)

    
    def reinitialize_n_first(self, n):
        """
        The function reinitializes the first n layers of the MLP with Xavier initialization.
        """

        for n_h, module in enumerate(list(self.nn)):
            if n_h < n:
                if isinstance(module, torch.nn.Linear):
                    # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                    torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                    #torch.nn.init.xavier_uniform_(module.bias)
                    if module.bias is not None:
                        # Initialize bias with zeros
                        # ?? questionable
                        torch.nn.init.zeros_(module.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hidden = self.nn(x)
        #print(x_hidden.shape)
        #mean = self.mean_out(x_hidden)
        #var = self.var_out(x_hidden)

        out = self.mean_var_out(x_hidden)
        mean = out[:,:(self.n_out*2)//2]
        var = out[:,(self.n_out*2)//2:]

        var = torch.nn.functional.softplus(var)

        return mean, var
    
    def report_w_scaler(self, x: torch.Tensor, scaler):
        """ Reports the mean and variance of the model with a given scaler
        """

        self.eval()
        mean, var = self.forward(x)
        mean = scaler.inverse_transform(mean.detach())
        var = var.detach() * scaler.scale_ ** 2

        return torch.tensor(mean), torch.tensor(var)
    
    def report_w_mean_var(self, x: torch.Tensor, mean, std):
        """ Reports the mean and variance of the model with a given scaler
        """

        self.eval()
        mean, var = self.forward(x)
        mean = mean.detach().numpy() + mean
        var = var.detach().numpy() * std ** 2

        return torch.tensor(mean), torch.tensor(var)

class MLP_mean(torch.nn.Module):
    """ Initializes a mean MLP with an arbitrary number of hidden layers and activation functions
    """
    def __init__(self, n_in, n_out, n_hidden, n_hidden_layers, activation) -> None:
        super().__init__()

        modules = []

        if n_hidden_layers > 0:
            modules.append(torch.nn.Linear(n_in,n_hidden))
            modules.append(activation())
        else:
            modules.append(torch.nn.Linear(n_in,n_hidden))
        
        
        for n in range(n_hidden_layers-1):
            modules.append(torch.nn.Linear(n_hidden,n_hidden))
            modules.append(activation())
        


        self.nn = torch.nn.Sequential(
                *modules
                )
        
        self.mean_out = torch.nn.Linear(n_hidden,n_out)

        for module in self.nn:
            if isinstance(module, torch.nn.Linear):
                # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                #torch.nn.init.xavier_uniform_(module.bias)
                if module.bias is not None:
                    # Initialize bias with zeros
                    # ?? questionable
                    torch.nn.init.zeros_(module.bias)


        torch.nn.init.xavier_uniform_(self.mean_out.weight)
        torch.nn.init.zeros_(self.mean_out.bias)

    def reinitialize_n_last(self, n):
        """ 
        The function reinitializes the last n layers of the MLP with Xavier initialization.
        """
        for n_h, module in enumerate(reversed(list(self.nn))):
            if n_h < n:
                if isinstance(module, torch.nn.Linear):
                    #print("reinit weight")
                    # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                    torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                    #torch.nn.init.xavier_uniform_(module.bias)
                    if module.bias is not None:
                        #print("reinit bias")
                        # Initialize bias with zeros
                        # ?? questionable:
                        torch.nn.init.zeros_(module.bias)

    
    def reinitialize_mean(self):
        """
        The function reinitializes the mean layer of the MLP with Xavier initialization.
        """
        torch.nn.init.xavier_uniform_(self.mean_out.weight)
        torch.nn.init.zeros_(self.mean_out.bias)

    
    def reinitialize_n_first(self, n):
        """
        The function reinitializes the first n layers of the MLP with Xavier initialization.
        """
        for n_h, module in enumerate(list(self.nn)):
            if n_h < n:
                if isinstance(module, torch.nn.Linear):
                    # Initialize linear layer weights using the "old" initialization (Xavier initialization)
                    torch.nn.init.xavier_uniform_(module.weight,gain=torch.nn.init.calculate_gain('relu'))
                    #torch.nn.init.xavier_uniform_(module.bias)
                    if module.bias is not None:
                        # Initialize bias with zeros
                        # ?? questionable
                        torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hidden = self.nn(x)
        mean = self.mean_out(x_hidden)
        return mean