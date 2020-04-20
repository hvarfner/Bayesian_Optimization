import numpy as np
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF, Matern
from copy import deepcopy
import time
from multiprocessing import Process, Queue
from acquisitions import EI, UCB
from penalizers import LP, HLP
from objectives import branin
from parallel_bo import sample_next_parallel, acq_parallel

def bayesian_optimization(n_iters, function, bounds, acq_func = UCB, penalizer = HLP, local_L = True, n_processes = 4,
                          X_init = None, set_init = [], n_init = 10, gp_params = None, find_min = True, alpha = 1e-5, verbose = True):
    X_tested = []
    y_tested = []
    n_params = bounds.shape[0]
    if len(set_init) > 0:
        X_init = set_init

    if X_init is None:
        X_init = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_init, bounds.shape[0]))

    for X in X_init:
        X_tested.append(X)
        y_tested.append(function(X))

    y_np = np.array(y_tested)

    # creating the gaussian process
    if gp_params is not None:
        gaussian_process = gp.GaussianProcessRegressor(**args) 
    
    else:
        kernel = Matern()
        gaussian_process = gp.GaussianProcessRegressor(
            kernel = kernel, alpha = alpha, n_restarts_optimizer = 10, normalize_y = True)
        gaussian_process.fit(X_tested, y_tested)


    print(f'Parallel BO ititiated.')
    # the queue should hold process number, X- and y-values for the point to evaluate
    q = Queue()
    X_eval = np.array([[None] * n_params] * n_processes)
    process_list = [None] * n_processes
    for i in range(n_processes):
        print(f'Searching for X_eval {i}...')

        X_eval[i] = sample_next_parallel(acq_parallel, gaussian_process, X_eval, y_np, bounds, find_min=find_min,
                                                acq_func=acq_func, penalizer=penalizer, local_L=local_L)


    for i in range(n_processes):
        print(f'Initiating evaluation of {X_eval[i]}')
        # initiating the processes
        process_list[i] = Process(target=function, args = (X_eval[i], i, q))
        process_list[i].start()


        ctr = 0
        print(f'{i} started')

    for i in range(n_iters):

        res = q.get()
        finished = res['process']
        y = res['y']

        # append the results
        X_tested.append(X_eval[finished])
        y_tested.append(y)

        # fit an updated gaussian process
        gaussian_process.fit(X_tested, y_tested)

        # then start a new process
        print(f'Worker {finished} is evaluating its next point.')
        X_eval[finished] = sample_next_parallel(acq_parallel, gaussian_process, X_eval, y_np, bounds, find_min=find_min,
                                                acq_func=acq_func, penalizer=penalizer, local_L=local_L)
        process_list[finished] = Process(target=function, args=(X_eval[finished], finished, q))
        process_list[finished].start()

        if verbose:
            print(f'{X_eval[finished]} is being evaluated by worker {finished}.')

    return X_tested, y_tested

if __name__ == '__main__':
    X_tested, y_tested = bayesian_optimization(20, branin, bounds = np.array([[-5, 10], [0, 15]]),
                                               n_init = 16, acq_func = UCB, penalizer = HLP, local_L = True)


    best_value = np.min(y_tested)
    best_iter = np.argmin(y_tested)
    best_X = X_tested[best_iter]
    print(best_X, best_value, best_iter)
