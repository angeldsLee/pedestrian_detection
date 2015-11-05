'''
Created on 2014-5-10

@author: angelds
'''
#from cv import *
import numpy as np
import cv2
import cv2.cv as cv
import scipy.misc
import matplotlib.pyplot as mp

CV_CAP_PROP_FRAME_COUNT = 7

cap = cv2.VideoCapture("SampleVideo.avi")
nFrames = cap.get(CV_CAP_PROP_FRAME_COUNT)

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy=x,y


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,255),2)

        else:
            cv2.circle(img,(x,y),5,(0,0,255),1)


if cap.isOpened():
    a=0  
    
    while True:  
        ret, prev = cap.read()  
        b=np.shape(prev)
        if ret==True:
            cv2.namedWindow('image')
            cv2.imshow('video', prev)
            a=a+1
            if (a==20):
                cv2.setMouseCallback('image',draw_circle)
                storage1 = cv.CreateMemStorage(0)
                img = prev
#                img_1=scipy.misc(.img,'find_edges') 

                img_1=cv.fromarray(img)  
                found0 =  list(cv.HOGDetectMultiScale(img_1, storage1, win_stride=(8,8),
                                padding=(32,32), scale=1.05, group_threshold=2))
                
                print found0
                ix_1=found0[0][0][0]
                iy_1=found0[0][0][1]
                x_1=found0[0][1][0]
                y_1=found0[0][1][1]
                print ix_1,iy_1,x_1,y_1
                

            
            if (a==np.ceil(nFrames/2)):
                storage = cv.CreateMemStorage(0)
                img1=prev
                img11=cv.fromarray(img1)  
                found =  list(cv.HOGDetectMultiScale(img11, storage, win_stride=(8,8),
                                padding=(32,32), scale=1.05, group_threshold=2))
                
            
                while(1):
                    cv2.imshow('image',img)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('m'):
                        mode = not mode
                    elif k == 27:
                        break
                    
                print found
                ix1=found[0][0][0]
                iy1=found[0][0][1]
                x1=found[0][1][0]
                y1=found[0][1][1]
                print ix1,iy1,x1,y1
#                drawing=True
                cv.NamedWindow("image1", 1)
                cv2.imshow('image', img1)
                cv2.rectangle(img1,(ix1,iy1),(x1,y1),(0,255,255),2)
                    
        else:  
            break  
        if cv2.waitKey(20)==27:  
            break  

cap.release()
cv2.destroyAllWindows()







