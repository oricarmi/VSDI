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
import globalVals

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
    
