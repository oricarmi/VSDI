# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:21:35 2020

@author: Ori
"""

from scipy.io import loadmat
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import TSCA
import helperFunctions as hp
import MapMethods as mm
import skimage
x = loadmat(r"D:\dataForComparison\181218\zz_2.mat")
responseSig = loadmat(r"C:\Users\orica\OneDrive\Desktop\2nd degree\matlab codez\matlab - vsdi\VSDI-MATLAB\responseSig.mat")
responseSig = np.concatenate((responseSig["responseSig"][0],np.zeros(700)))

zz = x['zz']
[nrow,ncol] = [270,327]
T = zz.shape[1]
zz3D = zz.reshape((ncol,nrow,T))
#imgs = [[plt.imshow(i, animated=True)] for i in np.swapaxes(zz3D,0,2)]
#fig = plt.figure()
#ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True,
#                                repeat_delay=1000)
zz3D = zz3D.swapaxes(0,1)
zzz = np.reshape(zz3D,(nrow*ncol,T))
#fig1 = plt.figure()
#ax = fig1.add_subplot(1, 2, 1)
#ax.plot(zz3D[150,150,:])
#ax.set_title(r'pixel [150,150]')
#ax = fig1.add_subplot(1, 2, 2)
#ax.plot(zz3D[150,180,:])
#ax.set_title(r'pixel [150,180]')

#fig2 = plt.figure()
#AOF = []
#for i in range(8):
#    ax = fig2.add_subplot(1, 8, i+1)
#    temp = np.mean(zz3D[:,:,i*100+10:i*100+15],2)
#    temp = (temp-np.min(temp))/(np.max(temp)-np.min(temp))
#    AOF.append(temp)
#    plt.imshow(temp) 
#fi3 = plt.figure()
#tmp = np.dstack(AOF)
#AOF_final = np.amax(tmp,2)
#AOF_final_index = np.argmax(tmp,2)+1
#thresh = np.percentile(AOF_final,95);
#retMap = np.zeros((nrow,ncol,3))
#for i in range(nrow):
#    for j in range(ncol):
#        if AOF_final[i,j]>thresh:
#            retMap[i,j,:] = colorsys.hsv_to_rgb(AOF_final_index[i,j]/8,1,AOF_final[i,j])
#plt.imshow(retMap)
#index2plot = [11,3,15,7,5,13,1,9]
#fig4 = plt.figure()
#for i,j in enumerate(np.unique(AOF_final_index)):
#    ax = fig4.add_subplot(3,5,index2plot[i])
#    ax.imshow([[colorsys.hsv_to_rgb(j/8,1,1)]])
#    ax.set_title('loc %d' %(i+1))
###
fig5 = plt.figure()
mapsTSCA = []
theoreticalSigs = [np.roll(responseSig,100*i) for i in range(8)]
zzzCentered = hp.centerData(zzz)# Center mxied signals
zzzWhitened = hp.whitenData(zzzCentered) # Whiten mixed signals
for i,thisSig in enumerate(theoreticalSigs):
    ax = fig5.add_subplot(1, 8, i+1)
#    Xc = hp.centerData(zzz)
#    Xw = hp.MinMaxNorm(hp.whitenData(Xc.T)) 
    temp = TSCA.Func(zzzWhitened,thisSig,[np.eye(len(responseSig))],(1,-0.25))
    I = np.argmax(np.dot(temp["projected"][0:8,:],thisSig))
    tmp = np.reshape(np.abs(temp["components"][:,I]),(nrow,ncol))
    tmp = hp.MinMaxNorm(skimage.filters.gaussian(sp.signal.medfilt2d(tmp,[3,5]),0.2))
    mapsTSCA.append(tmp)
    plt.imshow(tmp)
hp.retMap(mapsTSCA)
fig6 = plt.figure()
for i,map in enumerate(mapsTSCA): 
    ax = fig6.add_subplot(1,8,i+1)
    plt.imshow(map)