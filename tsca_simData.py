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
import ICA
from ICA import centerData
from ICA import whitenData
import skimage
def createCircle(imgSize,center,r,smoothen = 0):
    circle = np.zeros(imgSize)
    for x in range(imgSize[0]):
        for y in range(imgSize[1]):
            circle[x,y] = (x-center[0])**2+(y-center[1])**2 < r**2
    if smoothen:
        circle = skimage.filters.gaussian(circle,sigma=smoothen,mode='nearest',truncate=2.0)
    return circle
fs = 100
T = 1000
p = 1600
m = int(math.sqrt(p))
TotalTime = 10
tAll = np.linspace(0,TotalTime,int(TotalTime*fs)) # time vector of entire recording
imgSize = (40,40)
globalVals.imgSize = imgSize
# <---- Create almost periodic 1.5Hz (noise) source
freqs = 2*np.pi*np.arange(1.2,1.7,0.05)
phases = 2*np.pi*np.random.rand(len(freqs))
sumSine = np.zeros(len(tAll))
for (freq,phase) in zip(freqs,phases):
    sumSine += np.sin(freq*tAll + phase)
sumSine = sumSine/3
# ---->
centers = [(10,10),(10,30),(20,20),(30,10),(30,30)]
signals = []
sigTime = np.hstack((np.zeros(50),np.exp((-1/20)*np.arange(T-50))))
for i in range(5):
    circle = createCircle(imgSize,centers[i],4,1)
    signal = {
            "time" : np.roll(sigTime,i*200),
            "space" : circle
            }
    signals.append(signal)
noise1 = {
        "time" : np.random.normal(0,1,T),
        "space" : np.random.random((imgSize[0],imgSize[0]))
        }
noise2 = {
        "time" : sumSine,
        "space" : np.matlib.repmat(np.sin(np.arange(m)),m,1)
        }
Z = []
for i in range(signal["time"].size):
    thisFrame  = signals[0]["time"][i]*signals[0]["space"]+\
    signals[1]["time"][i]*signals[1]["space"]+\
    signals[2]["time"][i]*signals[2]["space"]+\
    signals[3]["time"][i]*signals[3]["space"]+\
    signals[4]["time"][i]*signals[4]["space"]+\
    noise1["time"][i]*noise1["space"]+noise2["time"][i]*noise2["space"]
    Z.append(thisFrame.reshape((thisFrame.size)))
ZZ = np.array(Z).transpose()
X = ZZ
Xcentered = centerData(X)# Center mxied signals
Xw = whitenData(Xcentered) # Whiten mixed signals
output = TSCA.Func(Xw,signals[0]["time"],[noise2["time"]],(1,0),1)
TSCA.Analyze(output,3,[],1,T)
#WcICA,SIGcICA,iters = ICA.cICA(ZZ,signal["time"],np.random.rand(ZZ.shape[0],))
