import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF, Matern
import math


def LP(X, gaussian_process, X_eval, current_loss, n_params, find_min, bounds, local_L):
    
    other_X = X_eval[X_eval != None].reshape(-1, n_params)
    
    if other_X.shape[0] == 0:
        return 1
 
    mu_eval, std_eval = gaussian_process.predict(other_X, return_std = True)
    
    if local_L:
        L = compute_L(gaussian_process, n_params, bounds, X)
    
    else: 
        L = compute_L(gaussian_process, n_params, bounds)
        
    M = np.min(current_loss)
    n_under_eval = other_X.shape[0]
    # X, X_eval must be row vectors. Are they? Shape (n_under_eval, 2) otherwise reshape
    
    dists = np.array([np.linalg.norm(X - x) for x in other_X])
    Z = [(L * dists[i] - M + mu_eval[i]) / (std_eval[i] * np.sqrt(2))  for i in range(n_under_eval)]
    penalties = [0.5 * math.erfc((-1) * Z[i]) for i in range(n_under_eval)]
    
    # If LP needs to be investigated further (Normalize inputs?)
    #print('std_eval',std_eval, '\nmu', mu_eval, '\ndists',dists, '\nL', L, '\nM', M, '\nZ', Z,'\nPenalties', penalties)
    
    return np.prod(penalties)


# needs to compute L both globally and locally
def HLP(X, gaussian_process, X_eval, current_loss, n_params, find_min, bounds, local_L, gamma = 0.1):
    
    other_X = X_eval[X_eval != None].reshape(-1, n_params)
    
    if other_X.shape[0] == 0:
        return 1
 
    mu_eval, std_eval = gaussian_process.predict(other_X, return_std = True)
    
    if local_L:
        L = compute_L(gaussian_process, n_params, bounds, X)
    
    else: 
        L = compute_L(gaussian_process, n_params, bounds)
        
    M = np.min(current_loss)
    n_under_eval = other_X.shape[0]
    # X, X_eval must be row vectors. Are they? Shape (n_under_eval, 2) otherwise reshape
    
    dists = np.array([np.linalg.norm(X - x) for x in other_X])
    penalties = [dists[i] * L / (np.abs(mu_eval[i] - M) + gamma * std_eval[i]) for i in range(n_under_eval)]
    penalties = np.minimum(penalties, 1).tolist()

    return np.prod(penalties)


def compute_L(gaussian_process, n_params, bounds, X_local = None, n_samples = 101):
    
    if X_local is not None:
        
        X_range = np.array([bounds[i, 1] - bounds[i, 0] for i in range(n_params)])
        low_bounds, high_bounds = np.maximum(X_local - X_range / 20, bounds[:,0]), np.minimum(X_local + X_range / 20, bounds[:,1])
        loc_bounds = np.array([low_bounds, high_bounds]).T

    else:
        loc_bounds = bounds
    
    # dists - distance between samples in each input dimension (used to approximate derivative)
    dists = [(loc_bounds[i,1] - loc_bounds[i,0]) / (n_samples - 1) for i in range(n_params)]
    
    # mesh_X - matrix for every single point in input space
    mesh_X = [np.linspace(loc_bounds[i,0], loc_bounds[i,1], n_samples) for i in range(n_params)]
    
    all_X = np.array(np.meshgrid(*[mesh_X[i] for i in range(n_params)])).reshape(n_params,-1).T
    mu = gaussian_process.predict(all_X).reshape(n_samples, n_samples)
    return np.abs(np.gradient(mu, *dists)).max()
    