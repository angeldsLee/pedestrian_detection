'''
Created on 2014-5-10

@author: angelds
'''

import numpy as np
import cv2

cap = cv2.VideoCapture("SampleVideo.avi")

if cap.isOpened():  
    while True:  
        ret, prev = cap.read()  
        if ret==True:  
            cv2.imshow('video', prev)  
        else:  
            break  
        if cv2.waitKey(20)==27:  
            break  


cap.release()
cv2.destroyAllWindows()
