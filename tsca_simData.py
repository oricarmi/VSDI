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

fs = 100
T = 1000
p = 1600
m = int(math.sqrt(p))
imgSize = [m,m]

signal = {
        "time" : np.hstack((np.zeros(20),np.exp((-1/20)*np.arange(T-20)))),
        "space" : np.zeros((m,m)) 
        }
noise = {
        "time" : np.random.normal(0,1,T)
        "space" : }
