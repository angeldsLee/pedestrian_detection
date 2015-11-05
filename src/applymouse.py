'''
Created on 2014-5-10

@author: angelds
'''

import numpy as np
import cv2
import cv2.cv as cv

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
        ix,iy = x,y

#    elif event == cv2.EVENT_MOUSEMOVE:
#        if drawing == True:
#            if mode == True:
#                cv2.rectangle(img,(ix,iy),(x,y),(255,255,0),2)
#            else:
#                cv2.circle(img,(x,y),5,(0,0,255),1)

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
            if (a==np.ceil(nFrames/2)):
                img = prev
                im=cv.fromarray(img)
                cv.SaveImage('test1.png',im)             # save an image for being tested
                cv2.setMouseCallback('image',draw_circle)
            
                while(1):
                    cv2.imshow('image',img)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('m'):
                        mode = not mode
                    elif k == 27:
                        break
                    
        else:  
            break  
        if cv2.waitKey(20)==27:  
            break  

cap.release()
cv2.destroyAllWindows()