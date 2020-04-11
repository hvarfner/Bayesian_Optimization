import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF, Matern
import math

def branin(X):

    result = np.square(X[1] - (5.1/(4*np.square(math.pi)))*np.square(X[0]) + 
         (5/math.pi)*X[0] - 6) + 10*(1-(1./(8*math.pi)))*np.cos(X[0]) + 10
    
    result = float(result)
    noise = np.random.normal() * 0.
    
    time_sleep = np.random.randint(1, 3)
    #print ('result:', result, '\nobserved:', noise + result, '\nSleeping for', time_sleep, 'seconds.')
    #time.sleep(time_sleep)
    return result + noise

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
    ucb = mu + std * sign_X * kappa
    
    return ucb


# UCB takes the same input as EI for simplicity, even though it does not need all of them
# need to compute L both globally and locally

def LP(X, gaussian_process, X_eval, current_loss, n_params, find_min, bounds, local_L):
    
    mu_eval, std_eval = gaussian_process.predict(X_eval, return_std = True)
    
    if local_L:
        L = compute_L(gaussian_process, n_params, bounds, X)
    
    else: 
        L = compute_L(gaussian_process, n_params, bounds)
        
    M = np.min(current_loss)
    n_under_eval = X_eval.shape[0]
    # X, X_eval must be row vectors. Are they? Shape (n_under_eval, 2) otherwise reshape
    
    dists = np.array([np.linalg.norm(X - x) for x in X_eval])
    Z = [(L * dists[i] - M + mu_eval[i]) / (std_eval[i] * np.sqrt(2))  for i in range(n_under_eval)]
    penalties = [0.5 * math.erfc((-1) * Z[i]) for i in range(n_under_eval)]
    print('std_eval',std_eval, '\nmu', mu_eval, '\ndists',dists, '\nL', L, '\nM', M, '\nZ', Z,'\nPenalties', penalties)
    
    return np.prod(penalties)


# needs to compute L both globally and locally
def HLP(X, gaussian_process, X_eval, current_loss, n_params, find_min, bounds, local_L, gamma = 1):
    
    mu_eval, std_eval = gaussian_process.predict(X_eval, return_std = True)
    
    if local_L:
        L = compute_L(gaussian_process, n_params, bounds, X)
    
    else: 
        L = compute_L(gaussian_process, n_params, bounds)
        
    M = np.min(y_np)
    n_under_eval = X_eval.shape[0]
    # X, X_eval must be row vectors. Are they? Shape (n_under_eval, 2) otherwise reshape
    
    dists = np.array([np.linalg.norm(X - x) for x in X_eval])
    penalties = [dists[i] * L / (np.abs(mu_eval[i] - M) + gamma * std_eval[i]) for i in range(n_under_eval)]
    penalties = np.minimum(penalties, 1).tolist()
    print(penalties)

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
    