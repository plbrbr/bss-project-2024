import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

def prox_oblique(A):
    for ii in range((A).shape[1]):
        normeA = np.sqrt(np.sum(A[:,ii]**2))
        if normeA > 0 and normeA > 1.:
            A[:,ii] /= normeA
        
    return A

def prox_l1(x,lamb=0): #Proximal operator of l1 norm
        mask1 = x > lamb
        mask2 = x < -lamb
        return (
            mask1 * (-lamb)
            + mask1 * x
            + mask2 * lamb
            + mask2 * x
        )


def ISTA_S(A,X,S=None,lamb = 0, itmax = 10000):
    # set_loader : a data loader object
    # lamb : the threshold value used in the l1 proximal operator
    # itmax : maximum number of iterations. You can keep this value and not modify it.
    
    eps = 0.01
    S_est_list = [] # Each element of S_est_list should be a mini-batch of estimated S matrices.
    itmax_list = []# Each element of itmax_list should be a mini-batch of the maximal number of iterations reached by ISTA (only used for display).
    
    
    it = 0


    S_est = np.zeros([A.shape[1], X.shape[1]], dtype=float)
    S_est_prev = deepcopy(S_est)

    L = np.linalg.norm(A.squeeze()@A.squeeze().T,'nuc')
    it = 0
    while (np.linalg.norm(S_est - S_est_prev, ord='fro') > 1e-6 or it < 2) and it < itmax:
        if it > 0:
            S_est_prev = deepcopy(S_est)

        S_est = prox_l1(S_est - (0.9 / L) * (A.T @ (A @ S_est - X)), lamb=0.9/L*lamb)

        it += 1


    return S_est, it
    
