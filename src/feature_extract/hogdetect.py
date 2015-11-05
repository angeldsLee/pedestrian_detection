'''
Created on 2014-5-18

@author: angelds
'''

import numpy as np
import numpy.linalg as ng
import cv2
import cv2.cv as cv
import matplotlib.pyplot as mp
import matplotlib.pyplot as plt
import scipy.ndimage as se
#from scipy.ndimage import filters
from skimage import data, color

#np.set_printoptions(threshold=np.nan)

im1=cv2.imread("test.png")
im = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
#i=np.maximum(im,1)

#print i
#im=cv2.imread("test.png")
nwin_x=3
nwin_y=3
bin=9               # number of histogram bins
l=im.shape[0]
c=im.shape[1]
h=np.zeros((nwin_x*nwin_y*bin,1),np.float64)
m=np.sqrt(l/2)
step_x=np.floor(c/(nwin_x+1))
step_y=np.floor(l/(nwin_y+1))
cou=0
height, width= im.shape

#hx=np.tile(np.array([-1,0,1]).reshape(1,3), (height, 1))
#hy=np.tile(np.array([1,0,-1]).reshape(3,1), (1, width))

hx = np.array([-1,0,1]).reshape(1,3)
hy = np.array([1,0,-1]).reshape(3,1)

#gra_xr=se.filters.convolve1d(np.double(im),hx)
#gra_yu=se.filters.convolve1d(np.double(im),hy)
#gra_xr=cv2.filter2D(im,cv2.CV_32F,hx)
#gra_yu=cv2.filter2D(im,cv2.CV_32F,hy)
#gra_xr=se.convolve(im,hx,mode='constant',cval=0.0)
#gra_yu=se.convolve(im,hy,mode='constant',cval=0.0)
gra_xr=np.zeros(im.shape)
gra_yu=np.zeros(im.shape)
for i in xrange(l):
    for j in xrange(c):         # row 
        if (j == 0):
            gra_xr[i,j] = -1 * 0 + 0 * im[i,j] + 1 * im[i,j+1]
        elif (j == c-1):
            gra_xr[i,j] = -1 * im[i, j-1] + 0 * im[i,j] + 1 * 0
        else:
            gra_xr[i,j] = -1 * im[i, j-1] + 0 * im[i,j] + 1 * im[i,j+1]
            
for i in xrange(l):
    for j in xrange(c):         # row 
        if (i == 0):
            gra_yu[i,j] = 1 * 0 + 0 * im[i,j] - 1 * im[i+1,j]
        elif (i == l-1):
            gra_yu[i,j] = 1 * im[i-1,j] + 0 * im[i,j] - 1 * 0
        else:
            gra_yu[i,j] = 1 * im[i-1,j] + 0 * im[i,j] - 1 * im[i+1,j]
        
            
        
#        if (j==0 | j==c-1):
#            gra_xr[i,j]=im[i,j]    
#        elif(i==0 | i==l-1):
#            gra_yu[i,j]=im[i,j]
#        else:    
#            gra_xr[i,j]=im[i,j+1].astype(float)-im[i,j-1].astype(float)
#            gra_yu[i,j]=im[i+1,j].astype(float)-im[i-1,j].astype(float)
            

#np.savetxt("im.txt", im, fmt="%d")        
#np.savetxt("gra_xr.txt", gra_xr, fmt="%d")
#np.savetxt("gra_yu.txt", gra_yu, fmt="%d")
angles=np.arctan2(gra_yu,gra_xr)
magnit=np.sqrt(np.square(gra_yu)+np.square(gra_xr));
K=0
for n in xrange(nwin_y):
    for m in xrange(nwin_x):
        cou=cou+1
        angles2=angles[n*step_y+1:(n+2)*step_y,m*step_x+1:(m+2)*step_x]
        magnit2=magnit[n*step_y+1:(n+2)*step_y,m*step_x+1:(m+2)*step_x]
        v_angles=angles2.flatten()
        v_magnit=magnit2.flatten()
#        K=np.maximum(np.size(v_angles))
        K=np.size(v_angles)
        #assembling the histogram with 9 bins (range of 20 degrees per bin)
        bin1=0
        H2=np.zeros((bin,1),np.float64) 
        for ang_lim in np.arange(-np.pi+2*np.pi/bin, np.pi, 2*np.pi/bin):
            bin1=bin1+1
            for k in xrange(K):
                if (v_angles[k]<ang_lim):
                    v_angles[k]=100
                    H2[bin1]=H2[bin1]+v_magnit[k]
              
                
        H2=H2/(ng.norm(H2)+0.01);        
        h[(cou-1)*bin1:cou*bin1+1,0]=H2[:,0]

#i1=np.min(v_angles)
#print i1
#print h
np.savetxt("hogd.txt", h)

#a=h.shape
#print a

#, cmap=plt.cm.gray
plt.imshow(h, cmap=plt.cm.gray)
plt.title('Histogram of Oriented Gradients')
plt.show()
