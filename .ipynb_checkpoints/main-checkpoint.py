import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF, Matern
import math
import time
from copy import deepcopy
from multiprocessing import Process, Manager
from acquisitions import EI, UCB
from penalizers import LP, HLP
from objectives import branin
from parallel_BO import sample_next_parallel, acq_parallel, sample_callback

def bayesian_optimization(n_iters, function, bounds, acq_func = UCB, penalizer = HLP, local_L = True,
                          X_init = None, n_init = 10, gp_params = None, find_min = True, alpha = 1e-5, epsilon = 1e-7):
    
    
    X_tested = []
    y_tested = []
    n_params = bounds.shape[0]
    
    if X_init is None:
        X_init = np.random.uniform(bounds[:,0], bounds[:,1], (n_init, bounds.shape[0]))
    
    for X in X_init:
        X_tested.append(X)
        y_tested.append(function(X))
    
    # creating the gaussian process
    if gp_params is not None:
        gaussian_process = gp.GaussianProcessRegressor(**args) 
    
    else:
        kernel = Matern()
        gaussian_process = gp.GaussianProcessRegressor(
            kernel = kernel, alpha = alpha, n_restarts_optimizer = 10, normalize_y = True)
        gaussian_process.fit(X_tested, y_tested)
        
    if __name__ == '__main__':

        manager = Manager()
        lock = manager.Lock()
        process_list = [None] * n_processes
        # tracks where the current processes are being evaluated
        X_eval = manager.list([[0] * n_params] * n_processes)
        output_X = manager.list(X_tested)
        output_Y = manager.list(y_tested)
       
        # Initializing, iteration counts the iteration number
        iteration = 0
        for i in range(n_processes):
            process_list[i] = Process(target=sample_callback, args = (X_eval, output_X, output_Y, lock, i, function,
                acq_parallel, deepcopy(gaussian_process), np.array(X_eval), np.array(output_y), bounds, find_min,
                acq_func, penalizer, local_L))
            process_list[i].start()
            print(f'Process {i +1} started')
            iteration += 1
        
        while iteration < n_iters:
            
            # if the process is done
            for i in range(n_processes):
        
                if not process_list[i].is_alive():

                    # fit an updated gaussian process with copies of the output lists
                    gaussian_process.fit(list(output_X), list(output_y))

                    # then start a new process
                    process_list[i] = Process(target=sample_callback, args = (X_eval, output_X, output_Y, lock, i, function,
                        acq_parallel, deepcopy(gaussian_process), np.array(X_eval), np.array(output_y), bounds, find_min,
                        acq_func, penalizer, local_L))
                    process_list[i].start()
                    print(f'Process {i +1} restarted')
                    iteration += 1


    return list(output_X), list(output_y)

n_iters, function, bounds, acq_func = UCB, penalizer = HLP, local_L = True,
                          X_init = None, n_init = 10, gp_params = None, find_min = True, alpha = 1e-5, epsilon = 1e-7

X_tested, y_tested = bayesian_optimization(200, branin, bounds = np.array([[-5, 10], [0, 15]]), 
                                           n_init = 16, acq_func = UCB, penalizer = HLP, local_L = True)


best_value = np.min(y_tested)
best_iter = np.argmin(y_tested)
best_X = X_tested[best_iter]
print(best_X, best_value, best_iter)
