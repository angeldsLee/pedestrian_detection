'''
Created on 2014-5-20

@author: angelds
'''
# -*- coding: utf-8 -*-
"""
Created on Sat May 03 18:16:39 2014

@author: Administrator
"""

import cv2
import cv2.cv as cv
import numpy as np
import time
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter

#cap = cv2.VideoCapture(0) #read from camera
cap = cv2.VideoCapture('test.flv') #read from video
time.sleep(1)
boxes = {'start':(),'end':()}
#cv2.rectangle(img, pt1, pt2, color)
#cv2.line(img, pt1, pt2, color) 
drawing_box = False
def on_mouse(event, x, y, flags, params):
    if event == cv.CV_EVENT_LBUTTONDOWN:
        print 'Start Mouse Position: '+str(x)+', '+str(y)
        sbox = (x, y)
        boxes['start'] = sbox
        boxes['end'] = ()
    elif event == cv.CV_EVENT_LBUTTONUP:
        print 'End Mouse Position: '+str(x)+', '+str(y)
        ebox = (x, y)
        boxes['end'] = ebox

count = 0

while(1):
    cv2.namedWindow('real image')
    cv2.namedWindow('select image')
    cv2.namedWindow('gray image')
    #cv2.namedWindow('HOG image')
    cv.SetMouseCallback('real image', on_mouse, 0)
    count += 1
    flag,img = cap.read()
    #img = cv2.blur(img, (3,3))
    #vis = img.copy()
    
    

    if flag:
        if (boxes['start'] != ()) and (boxes['end'] != ()) :
            print boxes['start'] ,boxes['end']
            poS = boxes['start']
            poE = boxes['end']
            cv2.rectangle(img,boxes['start'],boxes['end'],(0, 255, 0), 2)
            select_img = img[poS[1]:poE[1],poS[0]:poE[0]]
            gray_img = cv2.cvtColor(select_img, cv2.COLOR_BGR2GRAY)
            #f_img,hog_img = hog(gray_img,visualise=True,normalise=True)
            cv2.imshow('select image', select_img)
            cv2.imshow('gray image', gray_img)
            #cv2.imshow('HOG image', hog_img)
        cv2.imshow('real image', img)
        print boxes
        if cv2.waitKey(4) == 27:
            cv2.destroyAllWindows()
            break
    else:
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break
#    elif count >= 150:
#        if cv2.waitKey(0) == 27:
#            cv2.destroyAllWindows()
#            break
#        count = 0