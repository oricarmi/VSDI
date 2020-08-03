# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:14:31 2020

@author: orica
"""

import helperFunctions as hf
import numpy as np
import globalVals

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
    thisFrame  = signal["time"][i]*signal["space"]+noise["time"][i]*noise["space"]
    Z.append(thisFrame.reshape((thisFrame.size)))
ZZ = np.array(Z).transpose()
imgSize = (40,40)
globalVals.imgSize = imgSize

ZZZ = hf.rshp(ZZ)