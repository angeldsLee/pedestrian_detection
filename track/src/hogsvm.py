'''
Created on 2014-5-17

@author: angelds
'''
import numpy as np
import cv2
import cv2.cv as cv
import matplotlib.pyplot as mp
from skimage.feature import hog
from skimage import data, color, exposure
import copy

CV_CAP_PROP_FRAME_COUNT = 7

cap = cv2.VideoCapture("SampleVideo.avi")
nFrames = cap.get(CV_CAP_PROP_FRAME_COUNT)
print nFrames
if cap.isOpened():
    a=0  
    hog_mat=np.zeros((240,360,nFrames))         # hog vector matrix
    hog_matcompare=np.zeros((240,360,nFrames))
#    pos=np.zeros((2,nFrames),np.float64)
    while True:  
        ret, prev = cap.read()  
        b=np.shape(prev)
        if ret==True:
#            cv2.namedWindow('hog_image of 90th nframes')
            cv2.imshow('video', prev)
            a=a+1
            img = prev
            im=cv.fromarray(img)
            cv.SaveImage('testsvm.png',im)   
#           save an image for being tested
            img=cv2.imread("testsvm.png")
            image = color.rgb2gray(img)
            fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True)
            l,c=hog_image.shape
            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
            hog_mat[:,:,a-1]=hog_image_rescaled[:,:]
            if (a==1):
                hog_matcompare[:,:,a-1]=hog_image_rescaled[:,:]
            else:
                hog_matcompare[:,:,a-1]=hog_image_rescaled[:,:]-hog_mat[:,:,0]
                
        
#                c = cv.WaitKey(0)
#                if c == ord('q'):
#                    break
#                cv2.setMouseCallback('image',draw_circle)
            
#                while(1):
#                    cv2.imshow('image',img)
#                    k = cv2.waitKey(1) & 0xFF
#                    if k == ord('m'):
#                        mode = not mode
#                    elif k == 27:
#                        break
                    
        else:  
            break  
        if cv2.waitKey(20)==27:  
            break  
        

cap.release()
cv2.destroyAllWindows()
#np.savetxt("hog_mat.txt", hog_mat)
hmat1=hog_mat[:,:,56]-hog_mat[:,:,0]            #  hmat1 is the same as hmat2
hmat2=hog_matcompare[:,:,56]
#cv2.namedWindow('hog_image of 90th nframes')
#cv2.imshow(hmat)
mp.figure(1)
mp.imshow(hog_mat[:,:,56], cmap=mp.cm.gray)
mp.title('hog_image of 56th nframes')
mp.figure(2)
mp.imshow(hmat2,cmap=mp.cm.gray)
mp.title('hog_image of 56thth nframes PLUS the first frame')
mp.show()

ho0=np.zeros((l,c,nFrames))
copi=np.zeros_like((l,c,nFrames))
copi=copy.copy(hog_matcompare)          # save data
iidx = np.where(hog_matcompare<=0.75)
hog_matcompare[iidx] = 0
ho0=copy.copy(hog_matcompare)
ho1=ho0[:,:,56]
mp.figure(3)
mp.imshow(ho1,cmap=mp.cm.gray)
mp.title('56th frame after threshold process')
mp.show()

pos=np.zeros((2,int(nFrames)))           # central position of tar
#find the position of moving target
#for k in xrange(128):
#    s=50
#    b=200
#    d=b-s
#    buff=ho0[:,:,k]
#    for j in xrange(c):
#        for i in xrange(s,b,1):
#            nu1=np.nonzero(buff[i:b,j])
#            nu2=np.nonzero(buff[s:i,j])
#            if ((buff[i,j]>0) & (nu1>=5) & (i<s+d/2)):
#                ymin=i
#            elif((buff[i,j]>0) & (nu2>=5) & (i>s+d/2)):
#                ymax=i
#            else:
#                ymin=s
#                ymax=b
#     
#           
#    ymid=(ymin+ymax)/2
#    pos[1,k]=ymid
#    
#    
#for k in xrange(128):
#    buff=ho0[:,:,k]
#    for i in xrange(ymin,ymax,1):
#        for j in xrange(c):
#            nu1=np.nonzero(buff[i,j:c])
#            nu2=np.nonzero(buff[i,0:j])
#            if ((buff[i,j]>0) & (nu1>=3)):
#                xmin=j
#            elif((buff[i,j]>0) & (nu2>=3)):
#                xmax=j
#            else:
#                xmin=0
#                xmax=c
#           
#           
#    xmid=(xmin+xmax)/2
#    pos[0,k]=xmid
T1=0.08
T2=0.08
for k in xrange(128):
    buff=ho0[:,:,k]
    buf=np.zeros((1,c))
    bu=np.zeros((1,c))
    for j in xrange(c):
        buf[0,j]=np.sum(buff[:,j])  
         
#    buf=buf/np.max(buf)         # histogram normalization could divide by 0 !!!! to inf
    indx = np.where(buf<=T1*np.max(buf)) # so in where existing inf > or < this invalid
    buf[indx] = 0               
    bu=copy.copy(buf)
    bu1=np.where(bu>0)          # it is possible to find nothing, i.e. bu1 is None so pop this indexError !!! consider by yourself
    nu=np.size(bu1,1)
    if (nu>=1):
        xmin=bu1[1][0]              # tuple
        
        xmax=bu1[1][nu-1]
        xmid=(xmin+xmax)/2          # find the x position of target
        pos[0,k]=xmid
    
    
for k in xrange(128):
    buff=ho0[:,:,k]
    buf=np.zeros((l,1))
    bu=np.zeros((l,1))
    s=50
    b=170
    for i in xrange(s,b,1):
        buf[i,0]=np.sum(buff[i,:])
            
#    buf=buf/np.max(buf)         # histogram normalization
    indx1=np.where(buf<=T2*np.max(buf))
    buf[indx1] = 0
    bu=copy.copy(buf)
    bu2=np.where(bu>0)
    nu=np.size(bu2,1)
    if (nu>=1):
        ymin=bu2[0][0]+s
        ymax=bu2[0][nu-1]+s
        ymid=(ymin+ymax)/2          # find the y position of target
        pos[1,k]=ymid    



mp.figure(4)  
mp.scatter(pos[0,:],pos[1,:])  
mp.title('position of target')
mp.xlim([0,360])
mp.ylim([0,240])
mp.show()
#print pos
