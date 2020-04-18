import numpy as np
import math


def branin(X):

    result = np.square(X[1] - (5.1/(4*np.square(math.pi)))*np.square(X[0]) + 
         (5/math.pi)*X[0] - 6) + 10*(1-(1./(8*math.pi)))*np.cos(X[0]) + 10
    
    result = float(result)
    noise = np.random.normal() * 0

    #time.sleep = np.random.randint(20, 60)
    return (result + noise - 400)/400