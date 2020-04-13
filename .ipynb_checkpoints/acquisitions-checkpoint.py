import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF, Matern
import math


# EI takes the measured x-values and the gaussian process object as well as the current evaluated loss
# Boolean for maximization/minimization
def EI(X, gaussian_process, current_loss, n_params, find_min = True, **args):
    
    X_pred = X.reshape(-1, n_params)
    mu, std = gaussian_process.predict(X_pred, return_cov = True)
    
    if find_min:
        best_loss = np.min(current_loss)
    else:
        best_loss = np.max(current_loss)
    
    # Normalize based on the GP posterior and account for max/min condition
    sign_X = (-1) ** find_min
    with np.errstate(divide = 'ignore'):
        norm_X = sign_X * (mu - best_loss)/std
        ei = mu * sign_X * (mu - best_loss) * norm.cdf(norm_X) + std * norm.pdf(norm_X)  
        
        # to exclude points with no standard deviation (likely alredy been tested)
        ei[std == 0] = 0
        return (-1) * ei

    
# UCB takes the same input as EI for simplicity, even though it does not need all of them
def UCB(X, gaussian_process, current_loss, n_params, find_min = True, kappa = 4):
    
    X_pred = X.reshape(-1, n_params)
    mu, std = gaussian_process.predict(X_pred, return_cov = True)
    
    sign_X = (-1) ** find_min
    ucb = sign_X * (-mu + std * kappa)
    
    return ucb

