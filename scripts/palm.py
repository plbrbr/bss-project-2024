#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:56:48 2022

@author: ckervazo
"""

import numpy as np
import copy as cp
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
#%%
def prox_l1(x,lamb=0): #Proximal operator of l1 norm
        mask1 = x > lamb
        mask2 = x < -lamb
        return (
            mask1 * (-lamb)
            + mask1 * x
            + mask2 * lamb
            + mask2 * x
        )

def prox_oblique(A):
    for ii in range((A).shape[1]):
        normeA = np.sqrt(np.sum(A[:,ii]**2))
        if normeA > 0 and normeA > 1.:
            A[:,ii] /= normeA
        
    return A

#%%
def PALM(X, A, S, lamb = 0, itmax = 100, Ainit=None):    
    eps = 0.01
  
    it=0
    S_est = np.zeros([S.shape[0],S.shape[1]], dtype=float) # S_est_mb contains all the different samples of the estimated sources in a mini-batch
    A_est = np.zeros([A.shape[0],A.shape[1]], dtype=float) # A_est_mb contains all the different samples of A in a mini-batch
    if Ainit:
        A_est = Ainit
    else :
        A_est = np.random.rand(*A.shape)
    # A_est = Ainit#[0]# Initialization of A

    S_est_prev = deepcopy(S_est) # For the stopping criterion
    A_est_prev = deepcopy(A_est) # For the stopping criterion
    
    while(np.linalg.norm(S_est-S_est_prev, ord = 'fro') > 1e-6 or np.linalg.norm(A_est-A_est_prev, ord = 'fro') > 1e-6 or it < 2) and it < itmax: # PALM iterations
        if it>0:
            S_est_prev = deepcopy(S_est)
            A_est_prev = deepcopy(A_est)
        
        L = np.linalg.norm(A_est.squeeze()@A_est.squeeze().T,'nuc') # Lipschitz constant for S update
        gamma = (0.9 / L)
        S_est = prox_l1(S_est-  gamma* A_est.T@(A_est @ S_est - X), lamb=0.9/L*lamb)  # Proximal gradient step for S
        
        eta = 0.9/np.linalg.norm(S_est.squeeze()@S_est.squeeze().T,'nuc') # Lipschitz constant for A update
        A_est = prox_oblique(A_est+eta*(X-A_est@S_est)@S_est.T) #Proximal gradient step for A
        it += 1
    
    print('itmax %s'%it)

        
    return A_est,S_est,it
