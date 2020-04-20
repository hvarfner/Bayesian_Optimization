import numpy as np
import math
import time

def branin(X, process = None, queue = None):
    time.sleep = np.random.randint(10, 30)

    result = np.square(X[1] - (5.1/(4*np.square(math.pi)))*np.square(X[0]) + 
         (5/math.pi)*X[0] - 6) + 10*(1-(1./(8*math.pi)))*np.cos(X[0]) + 10
    
    result = float(result)
    noise = np.random.normal() * 0
    if queue is not None:
        res = {'process': process, 'y': (result + noise - 400)/400}
        queue.put(res)
    return (result + noise - 400)/400