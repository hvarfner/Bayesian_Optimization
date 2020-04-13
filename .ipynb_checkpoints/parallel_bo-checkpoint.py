import numpy as np
from acquisitions import EI, UCB
from penalizers import LP, HLP
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize


# same arguments as before, and bounds to limit the optimization as well as n_restarts to allow multiple optimization attempts
def sample_next_parallel(acq_parallel, gaussian_process, X_eval, current_loss, bounds, find_min = True, n_restarts = 10,
                     acq_func = UCB, penalizer = HLP, local_L = True):
    
    X_best = None
    best_acq_val = 999
    n_params = bounds.shape[0]
    
    starting_points = np.random.uniform(bounds[:,0], bounds[:,1], size = (n_restarts, n_params))
    
    for point in starting_points:
        result = minimize(acq_parallel, point.reshape(1, -1), method = 'L-BFGS-B', bounds = bounds, args = 
                          (acq_func, penalizer, gaussian_process, X_eval, current_loss, 
                        find_min, n_params, bounds, local_L))
        
        if result.fun < best_acq_val:
            X_best = result.x
            best_acq_val = result.fun
            
    return X_best


def acq_parallel(X, acq_func, penalizer, gaussian_process, X_eval, current_loss, find_min, n_params, bounds, local_L):
    
    acq_value = acq_func(X, gaussian_process, current_loss, n_params, find_min)
    penalty = penalizer(X, gaussian_process, X_eval, current_loss, n_params, find_min, bounds, local_L)
    return acq_value * penalty

