import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import signal
import numpy.matlib as npml
from numpy import linalg as la

def centerData(x): # function to center the data
    return x - np.mean(x, axis=1, keepdims=True) # subtract mean of each component

def whitenData(x): # function to whiten the data
    U, S, V = np.linalg.svd(np.cov(x))# Single value decoposition of covariance of X
    d = np.diag(1.0 / np.sqrt(S))# get inverse diagonal matrix of eigenvalues
    whiteM = U@d@U.T # whitening matrix
    return whiteM@x # whiten X by projecting it on whitening matrix (rotation, not reduction)

def g(x):
    return np.tanh(x)
def gTag(x):
    return 1/np.power(np.cosh(x),2) #1 - g(x) * g(x)
def fastIca(X,  alpha = 1, thresh=1e-8, maxIters=5000):
    m, n = X.shape
    # Initialize random weights
    W = np.random.rand(m, m) # initial random weights
    for c in range(m): # iterate the components
            w = W[c, :].copy().reshape(m, 1) # reshape to column vector
            w = w / np.sqrt((w ** 2).sum()) # normalize weights
            iters = 0
            change = 1e10
            while ((change > thresh) and (iters < maxIters)): # while change in w is still siginificant and haven't reached max iterations, do another loop 
                ws = w.T@X # project signal onto weights
                # <----- calculate negentropy
                wg = g(ws*alpha).T 
                wgTag = gTag(ws)*alpha
                # ----
                # <-----  Update weights 
                wNew = np.mean(X*wg.T) - np.mean(wgTag)*np.squeeze(w) 
                wNew = wNew/np.sqrt((wNew ** 2).sum()) # normalize weights
                wNew = wNew - wNew@W[:c].T@W[:c] # Decorrelate weights    
                wNew = wNew/np.sqrt(wNew@wNew.T) # renormalize weights
                # ---->
                change = np.abs(np.abs((wNew * w).sum()) - 1) # Calculate change in w
                w = wNew # Update old weights to new weights 
                iters += 1 # iterate counter
            W[c, :] = w.T
    return W

def cICA(X,reference,W,thresh=0.05,learningRate=0.001, mu=0.1,lamda=0.1,gamma=0.1,maxIters=5000,minChange=1e-7):
    T = X.shape[1]
    oldw = W
    iters = 1
    Rxx = np.cov(X)
    change = 1
    while change>minChange and iters<maxIters:
        y = W.T@X
        v_gauss = np.random.normal(0,np.std(y),T)# gaussian random variable
        G = np.mean(np.log(np.cosh(y)) - np.log(np.cosh(v_gauss))) # calculate G (negentropy)
        gg = np.mean(np.power((y-reference),2)) - thresh # inequality constraint (MSE between this signal and reference signal)
        h = np.mean(np.power(y,2)) - 1 # equality constraints
        Gamma1 = np.sign(G)*(X@g(y).T)/T - mu*(X@(y-reference).T)/T - lamda*(X@y.T)/T # Gamma1 calculation
        Gamma2 = np.sign(G)*np.mean(gTag(y)) - mu - lamda # Gamma2 calculation
        W = W - learningRate*(la.inv(Rxx)@Gamma1)/Gamma2 # calculate new weights
        W = W/np.sqrt(np.sum(np.abs(W))) # normalize new weights
        mu = np.maximum(0,mu+gamma*gg) # update mu      
        lamda = lamda + gamma*h # update lamda
        change = 1-np.abs(W.T@oldw) # calculate change
        oldw = W # update weights
        iters += 1 # iterate counter
    return     W,W.T@X,iters                       


