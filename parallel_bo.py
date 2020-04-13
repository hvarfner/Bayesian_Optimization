import numpy as np
from acquisitions import EI, UCB
from penalizers import LP, HLP
import time
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


def sample_callback(X_eval, output_X, output_y, lock, process_nbr, function, acq_parallel, gaussian_process,
                    X_eval_np, output_y_np, bounds, find_min, acq_func, penalizer, local_L):
    
    # First, find the next point to evaluate based on the info retrieved when calling sample_callback (snapshot)
    X_to_eval = sample_next_parallel(acq_parallel, gaussian_process, X_eval_np, output_y_np, bounds, 
                         find_min = find_min, acq_func = acq_func, penalizer = penalizer, local_L = local_L)
    
    # When point to evaluate is found, add it to list of points currently evaluated 
    lock.acquire()
    X_eval[process_nbr] = X_to_eval
    lock.release()
    
    # Then, start evaluating the function for said value - currently takes approx. 3 times longer than the sample_next_function
    time.sleep(np.random.randint(40, 120))
    y_to_eval = function(X_to_eval)
    
    # When finished, add to the lists of evaluated points and remove the currently evaluated point from the list
    lock.acquire()
    X_eval[process_nbr] = None
    output_X.append(X_to_eval)
    output_y.append(y_to_eval)
    lock.release()