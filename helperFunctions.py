# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:07:29 2020

@author: orica
"""
# VSDI helper functions
import math
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import numpy.matlib
#import globalVals
import colorsys
def rshp(Z):
    return Z.reshape((globalVals.imgSize[0],globalVals.imgSize[1],-1))

def pltMaps(maps,ind2plot = 0):
    fig1 = plt.figure()
    if len(maps) == 5: # simulation
        k = 0
        for i in maps:
            ax = fig1.add_subplot(3, 6, ind2plot[k])
            ax.imshow(i)
            ax.set_title('signal %s' %k)
            k += 1
    elif len(maps) == 8: # 8 locs
        k = 1
        for i in maps:
            ax = fig1.add_subplot(3, 5, ind2plot(k))
            ax.imshow(i)
          #  ax.set_title('loc ' + k)
            k += 1
    elif len(maps) == 9 and ind2plot : # 9 locs
        k = 1
        for i in maps:
            ax = fig1.add_subplot(3, 3, ind2plot(k))
            ax.imshow(i)
           # ax.set_title('loc ' + k)
            k += 1
    else: # moving bars 2 Hz
        k = 1
        for i in maps:
            ax = fig1.add_subplot(3, 3, k)
            ax.imshow(i)
          #  ax.set_title('loc ' + k)
            k += 1
    
def retMap(maps,prctile = 95,nrow=270,ncol=327):
    fig3 = plt.figure()
    tmp = np.dstack(maps)
    final = np.amax(tmp,2)
    final_index = np.argmax(tmp,2)+1
    thresh = np.percentile(maps,95);
    retMap = np.zeros((nrow,ncol,3))
    for i in range(nrow):
        for j in range(ncol):
            if final[i,j]>thresh:
                retMap[i,j,:] = colorsys.hsv_to_rgb(final_index[i,j]/8,1,final[i,j])
    plt.imshow(retMap)
    index2plot = [11,3,15,7,5,13,1,9]
    fig4 = plt.figure()
    for i,j in enumerate(np.unique(final_index)):
        ax = fig4.add_subplot(3,5,index2plot[i])
        ax.imshow([[colorsys.hsv_to_rgb(j/8,1,1)]])
        ax.set_title('loc %d' %(i+1))
def MinMaxNorm(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def centerData(x): # function to center the data
    return x - np.mean(x, axis=1, keepdims=True) # subtract mean of each component

def whitenData(x): # function to whiten the data
    U, S, V = np.linalg.svd(np.cov(x))# Single value decoposition of covariance of X
    d = np.diag(1.0 / np.sqrt(S))# get inverse diagonal matrix of eigenvalues
    whiteM = U@d@U.T # whitening matrix
    return whiteM@x # whiten X by projecting it on whitening matrix (rotation, not reduction)
