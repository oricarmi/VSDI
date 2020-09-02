# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:55:49 2020

@author: orica
"""
import MapMethods as mm
import helperFunctions as hp
#theoreticalSigs = [np.roll(responseSig,100*i) for i in range(8)]
#temp = mm.GLM(zzz,[0.78,2.7,3.3,4.8],theoreticalSigs)
#temp = np.reshape(temp.T,(nrow,ncol,8))
#fig2 = plt.figure()
#mapsGLM = []
#for i in range(8):
#    ax = fig2.add_subplot(1, 8, i+1)
#    tmp = (temp[:,:,i])
#    ax.imshow(tmp)
#    tmp = hp.MinMaxNorm(tmp)
#    mapsGLM.append(tmp)
#hp.retMap(mapsGLM)
hp.retMap(mapsTSCA)