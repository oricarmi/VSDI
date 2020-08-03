# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 19:26:44 2020

@author: Ori
"""
import numpy as np

imgSize = () # image size
T1 = () # length of time signal
T = () 
fs = 100
t = np.linspace(0,(T1-1)/fs,T1)
alphas = np.asarray([[50,200],[10,100],[10,50],[50,150]])/1000
randomAlphas = np.concatenate((alphas[0,0]+np.random.rand(15000,1)*(alphas[0,1]-alphas[0,0]),alphas[1,0]+np.random.rand(15000,1)*(alphas[1,1]-alphas[1,0]),alphas[2,0]+np.random.rand(15000,1)*(alphas[2,1]-alphas[2,0]),alphas[3,0]+np.random.rand(15000,1)*(alphas[3,1]-alphas[3,0])),axis=1)
r = np.zeros((randomAlphas.shape[0],t.size))
for i in np.arange(r.shape[0]): # iterate random signals
    for j in np.arange(r.shape[1]): # iterate time
        if t[j]<=randomAlphas[i,0] or t[j]>=(randomAlphas[i,0]+randomAlphas[i,1]+randomAlphas[i,2]+randomAlphas[i,3]): # if before latency or after decay, it is zero
            continue
        elif randomAlphas[i,0]<=t[j] and t[j]<=(randomAlphas[i,0]+randomAlphas[i,1]): # if in rise time
            r[i,j] = 0.5*np.deg2rad((1-np.cos(np.pi*(t[j]-randomAlphas[i,0])/randomAlphas[i,1])))
        elif (randomAlphas[i,0]+randomAlphas[i,1])<=t[j] and t[j]<=(randomAlphas[i,0]+randomAlphas[i,1]+randomAlphas[i,2]): # if in plateau time
            r[i,j] = 1
        else: # if (randomAlphas(i,1)+randomAlphas(i,2)+randomAlphas(i,3))<=t(j) && t(j)<=(randomAlphas(i,1)+randomAlphas(i,2)+randomAlphas(i,3)+randomAlphas(i,4)) # if in decaying time
            r[i,j] = 0.5*np.deg2rad((1+np.cos(np.pi*(t[j]-randomAlphas[i,0]-randomAlphas[i,1]-randomAlphas[i,2])/randomAlphas[i,3])))
U,S,V = np.linalg.svd(r,full_matrices = False)
basis = V[0:2,:]

    