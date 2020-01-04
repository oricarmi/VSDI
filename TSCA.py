# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:52:48 2020

@author: user
"""

# TSCA module
import numpy as np
import scipy as sp

def FrobNorm(A,B):
    return np.trace(np.transpose(B)*A)

def Func(Z,X,*Y,*gamma,reduceComp):
    if X.shape[0] == X.size: # if it is a vector
        T = X.size
        Cx = outer(X,X)/T # estimate autocorrelation matrix
        else: # it is already a square autocorrelation matrix
            Cx = X
            T = X.shape[0]
    for i in Y:
        if Y.shape[0] == Y.size # if it is a vector
            Cy[i] = outer(Y[i],Y[i])/T
            else:
                Cy[i] = Y
                