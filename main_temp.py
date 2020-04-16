import pandas as pd
import numpy as np
from numpy.linalg import norm as euclidian
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF, Matern
import time
from acquisitions import EI, UCB
from penalizers import LP, HLP
from objectives import branin
from parallel_bo import sample_next_parallel, acq_parallel


def bayesian_optimization(n_iters, function, bounds, acq_func=UCB, penalizer=HLP, local_L=True, n_processes=4,
                          X_init=None, n_init=10, gp_params=None, find_min=True, alpha=1e-5, epsilon=1e-7,
                          verbose=True, set_init = []):
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

    # numpy array of y needed for processes
    y_np = np.array(y_tested)

    # creating the gaussian process
    if gp_params is not None:
        gaussian_process = gp.GaussianProcessRegressor(**args)

    else:
        kernel = Matern()
        gaussian_process = gp.GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, n_restarts_optimizer=10, normalize_y=True)
        gaussian_process.fit(X_tested, y_tested)

    # Initializing, iteration counts the iteration number
    iteration = 0
    X_eval = X_eval = np.array([[None] * n_params] * n_processes)
    for i in range(n_processes):
        X_eval[i] = sample_next_parallel(acq_parallel, gaussian_process, np.array(X_eval), y_np, bounds,
                                         find_min=find_min,
                                         acq_func=acq_func, penalizer=penalizer, local_L=local_L)
        if verbose:
            print(f'{X_eval[i]} is being evaluated.')

    for i in range(n_iters):

        if verbose:
            print(f'{X_eval[finished]} just finished by worker {finished}.')

        finished = np.random.randint(4)
        X_tested.append([val for val in X_eval[finished]])
        y_tested.append(function(X_eval[finished]))
        y_np = np.array(y_tested)

        # fit an updated gaussian process
        gaussian_process.fit(X_tested, y_tested)

        # then start a new process
        X_eval[finished] = sample_next_parallel(acq_parallel, gaussian_process, X_eval, y_np, bounds, find_min=find_min,
                                                acq_func=acq_func, penalizer=penalizer, local_L=local_L)

        if verbose:
            print(f'{X_eval[finished]} is being evaluated.')

    return X_tested, y_tested


names = ['Local HLP', 'Global LP']
results = {}
no_tests = 10
n_init = 20
bounds=np.array([[-5, 10], [0, 15]])
set_init = [np.random.uniform(bounds[:, 0], bounds[:, 1], (n_init, bounds.shape[0])) for i in range(no_tests)]
ctr = 0

for version in [[True, HLP], [False, LP]]:

    for run in range(no_tests):


        X_tested, y_tested = bayesian_optimization(40, branin, bounds=bounds,
                                                       n_init=n_init, acq_func=UCB, penalizer=version[1], local_L=version[0],
                                                       verbose=False, set_init=set_init[run])

        # de-normalize the output
        results[names[ctr] + str(run)] = (np.array(y_tested) * 400 + 400).tolist()
        print(f'{names[ctr]} run {run} finished.')

    ctr += 1

df = pd.DataFrame(results)
df.to_csv('results_same_start.csv')