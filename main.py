# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:43:40 2019

@author: Ori
"""
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

class experiment:
    def __init__(self,Z,signal,noise,gamma,reduceComp):
        self.Z = Z
        self.signal = signal
        self.noise = noise
        self.gamma = gamma
        self.reduceComp = reduceComp

 