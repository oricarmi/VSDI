# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 19:56:40 2020

@author: Ori
"""
import math
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import TSCA
import numpy.matlib
import globalVals

fs = 100
T = 1000
p = 1600
m = int(math.sqrt(p))
imgSize = (40,40)
globalVals.imgSize = imgSize

signal = {
        "time" : np.hstack((np.zeros(20),np.exp((-1/20)*np.arange(T-20)))),
        "space" : np.zeros((m,m)) 
        }
signal["space"][5:15,5:15] = 1
noise = {
        "time" : np.random.normal(0,1,T),
        "space" : np.matlib.repmat(np.sin(np.arange(m)),m,1)
        }
Z = []
for i in range(signal["time"].size):
    thisFrame  =signal["time"][i]*signal["space"]+noise["time"][i]*noise["space"]
    Z.append(thisFrame.reshape((thisFrame.size)))
ZZ = np.array(Z).transpose()
output = TSCA.Func(ZZ,signal["time"],[noise["time"]],(1,-0.2),1)
TSCA.Analyze(output,3,[],1,T)
