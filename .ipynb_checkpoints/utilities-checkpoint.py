import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.stats

from numpy.linalg import inv
import matplotlib.pyplot as plt
from acq_functions import expected_imp, UCB, prob_imp

def plot_gp(mu, cov, X, X_train = None, y_train = None, samples = []):
    X = X.ravel()
    mu = mu.ravel()
    conf_interval = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + conf_interval, mu - conf_interval, alpha = 0.3)
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw = 1, ls = ':', label = f'Sample {i + 1}')

    if X_train is not None:
        plt.plot(X_train, y_train, 'rx')
        plt.plot(X, mu, lw = 1, label = 'Mean function', color = 'black')

    plt.legend()
    plt.show()

def plot_acq_function(X, mu, stdevs, y_max, acq_func, **params):
    acq_values = acq_func(mu, stdevs, y_max, **params)
    plt.plot(X, acq_values[-1], lw = 1.5, label = 'Acquisition function value')
    plt.show()

# EXAMPLE FUNCTIONS
def example_func(X_val):
    return np.exp(np.sin(np.power(X_val - 4 , 2)) + 0.5).reshape(-1, 1)


# ACQUISITION FUNCTIONS
# takes one point and evaluates the acq.function at said point
def expected_imp(mu_n, std_n, y_max, epsilon = 1):
    norm_y = (mu_n - y_max - epsilon) / std_n
    return (-1) * norm_y * std_n * scipy.stats.norm.pdf(norm_y) + std_n * scipy.stats.norm.cdf(norm_y)

def prob_imp(mu_n, std_n, y_max, epsilon = 1):
    norm_y = (mu_n - y_max - epsilon)/std_n
    return scipy.stats.norm.cdf(norm_y)

def UCB(mu_n, std_n, y_max, beta = 1):
    return mu_n + std_n * beta


# KERNELS
# Write kernel class and access the separate functions
def squared_exp(X1, X2, l = 1.0, sigma_f = 1.0):

    # X1 - Point in n-dim room for which cov. is to be computed, m * d
    # X2 - points already computed, n * d
    # computing covariance matrix between X and X_other
    print(X1, X2)
    sqdist = np.sum(np.power(X1, 2), 1).reshape(-1, 1) + np.sum(np.power(X2, 2), 1) - 2 * np.dot(X1, X2.T)
    return np.power(sigma_f, 2) * np.exp((-0.5)/np.power(l, 2) * sqdist)

def matern52(X1, X2, sigma_f = 1.0, sigma_l = 1.0):
    sqdist = np.sum(np.power(X1, 2), 1).reshape(-1, 1) + np.sum(np.power(X2, 2), 1) - 2 * np.dot(X1, X2.T)
    return np.power(sigma_f, 2) * (1 + np.sqrt(5) * sqdist/ sigma_l + np.power((np.sqrt(5) * sqdist/ sigma_l), 2)
                                   ) * np.exp((-1)*np.sqrt(5) * sqdist/sigma_l)

