# Contains exact solutions for convergence analysis
import numpy as np

def exact_henrick(x, t):
    return np.sin(np.pi*(x-t) - np.sin(np.pi*(x-t))/np.pi)

