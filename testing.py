# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:55:49 2020

@author: orica
"""
import MapMethods as mm

theoreticalSigs = [np.roll(responseSig,100*i) for i in range(8)]
temp = mm.GLM(zzz,[0.78,3.3],theoreticalSigs)
temp = np.reshape(temp.T,(nrow,ncol,8))
fig2 = plt.figure()
mapsGLM = []
for i in range(8):
    ax = fig2.add_subplot(1, 8, i+1)
    tmp = (temp[:,:,i])
    ax.imshow(tmp)
    tmp = (tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))
    mapsGLM.append(tmp)
fig3 = plt.figure()
tmp = np.dstack(mapsGLM)
TSCA_final = np.amax(tmp,2)
TSCA_final_index = np.argmax(tmp,2)+1
thresh = np.percentile(TSCA_final,95);
retMap = np.zeros((nrow,ncol,3))
for i in range(nrow):
    for j in range(ncol):
        if TSCA_final[i,j]>thresh:
            retMap[i,j,:] = colorsys.hsv_to_rgb(TSCA_final_index[i,j]/8,1,TSCA_final[i,j])
plt.imshow(retMap)
index2plot = [11,3,15,7,5,13,1,9]
fig4 = plt.figure()
for i,j in enumerate(np.unique(TSCA_final_index)):
    ax = fig4.add_subplot(3,5,index2plot[i])
    ax.imshow([[colorsys.hsv_to_rgb(j/8,1,1)]])
    ax.set_title('loc %d' %(i+1))
