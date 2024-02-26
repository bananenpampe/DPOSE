from sklearn.isotonic import IsotonicRegression
import numpy as np


class MultiRescaler:
    """
    Rescales the variance of the predictions to the variance of the true values
    using isotonic regression.
    
    Parameters
    ----------
    n_properties : int
        number of properties to rescale
        
    """
    def __init__(self, n_properties) -> None:
        self.n_properties = n_properties
        self.rescalers = [IsotonicRegression(out_of_bounds="clip") for _ in range(n_properties)]
    
    def fit(self, Ypred, Yvar_pred, Ytrue):
        assert Ypred.shape[1] == self.n_properties
        assert Yvar_pred.shape[1] == self.n_properties
        assert Ytrue.shape[1] == self.n_properties

        for i in range(self.n_properties):
            Z = np.abs(Ypred[:,i] - Ytrue[:,i])**2
            var = Yvar_pred[:,i].reshape(-1,1)
            self.rescalers[i].fit(var,Z)

    def predict(self, Yvar_pred):
        assert Yvar_pred.shape[1] == self.n_properties
        tmp_var = []
        for i in range(self.n_properties):
            var = Yvar_pred[:,i].reshape(-1,1)
            tmp_var.append(self.rescalers[i].predict(var).reshape(-1,1))

        return np.hstack(tmp_var)