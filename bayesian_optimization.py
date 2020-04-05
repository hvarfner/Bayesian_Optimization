import numpy as np


def compute_posterior(X_n, X_train, y_train, kernel):

    K = kernel(X_train, X_train) + 1e-10 *np.eye(len(X_train))
    k_n = kernel(X_n, X_train)

    # sample length * sample length in size
    k_nn = kernel(X_n, X_n) + 1e-10 * np.eye(len(X_n))
    K_inv = inv(K)
    mu_n = k_n.dot(K_inv).dot(y_train)
    cov_n = k_nn - k_n.dot(K_inv).dot(k_n.T)

    return mu_n, cov_n

def query_random_points(no_points, X, objective):
    X_train = np.random.randint(X.shape[1], size = (10, X.shape[0]))
    y_train =  objective(X_train)
    return X_train, y_train

def find_and_query(X_train, y_train, objective):
    acq_values = acq_func(mu, stdevs, y_max, **params)[-1]
    X_next = X[np.argmax(acq_values)]
    X_train = np.append(X_train, X_next)
    y_train = np.append(y_train, objective(X_next))


def bayesian_optimization(objective, no_initial, no_iters, X_range):
# Creating function and parameters to be optimized
    # objective - known or unknown function to be optimized
    # params - list of parameter names and their associated ranges in dict form

    # setting kernel and acq.function with relevant parameters - WAIT
    kernel = squared_exp
    acq_function = expected_imp

    # Function for initiating range, mu and lambda
    X = np.array([np.linspace(X_range[0], X_range[1], 1001) for item in params.keys()])
    print(X)
    mu = np.zeros(X.shape)
    cov = squared_exp(X, X)

    # query_next_point FOR a number of random points
    X_train, y_train = query_random_points(no_initial, X, objective)
    # compute_posterior when done
    mu, cov = compute_posterior(X, X_train, y_train, kernel)
    for iter in range(no_iters):
        X_train, y_train = find_and_query(y_train, mu, cov, acq_function, objective)
        mu, cov = compute_posterior(X, X_train, y_train, kernel)
        plot_gp(mu, cov, X_train, y_train)



bayesian_optimization(example_func, 10, 20, [-1, 1])