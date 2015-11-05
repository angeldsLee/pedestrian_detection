'''
Created on 2014-5-17

@author: angelds
'''
'''
Created on 2014-5-10

@author: angelds
'''
import sys
import matplotlib.pyplot as mp
import numpy as np
import cv2
import cv2.cv as cv

CV_CAP_PROP_FRAME_COUNT = 7

cap = cv2.VideoCapture("SampleVideo.avi")
nFrames = cap.get(CV_CAP_PROP_FRAME_COUNT)
print nFrames
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

def inside(r, q):
    (rx, ry), (rw, rh) = r
    (qx, qy), (qw, qh) = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


if cap.isOpened():
    a=0  
    pos=np.zeros((2,nFrames),np.float64)
    while True:  
        ret, prev = cap.read()  
        b=np.shape(prev)
        if ret==True:
            cv2.namedWindow('image')
            cv2.imshow('video', prev)
            a=a+1
            img = prev
            im=cv.fromarray(img)
            cv.SaveImage('test.png',im)             # save an image for being tested
            imglist = ["test.png"]

            cv.NamedWindow("people detection demo", 1)
            storage = cv.CreateMemStorage(0)
            
            for name in imglist:
                n = name.strip()
                print n
                try:
                    img = cv.LoadImage(n)
                except:
                    continue
                
                #ClearMemStorage(storage)
                found = list(cv.HOGDetectMultiScale(img, storage, win_stride=(8,8),
                    padding=(32,32), scale=1.05, group_threshold=2))
                found_filtered = []
                for r in found:
                    insidef = False
                    for q in found:
                        if inside(r, q):
                            insidef = True
                            break
                    if not insidef:
                        found_filtered.append(r)
                for r in found_filtered:
                    (rx, ry), (rw, rh) = r
                    rx1=rx + int(rw*0.1)
                    ry1=ry + int(rh*0.07)
                    tl = (rx1, ry1)
                    rx2=rx + int(rw*0.9)
                    ry2= ry + int(rh*0.87)
                    br = (rx2, ry2)
#                    cv.Rectangle(img, tl, br, (0, 255, 0), 3)

                    posx=np.ceil((rx1+rx2)/2)            # center of target
                    posy=np.ceil((ry1+ry2)/2)
                    pos[0][a-1]=posx
                    pos[1][a-1]=posy
                    

                    
#                cv.ShowImage("people detection demo", img)
                c = cv.WaitKey(0)
                if c == ord('q'):
                    break
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

mp.scatter(pos[0,:],pos[1,:])  
mp.xlim([0,360])
mp.ylim([0,240])
mp.show()
print pos     
c=pos.shape[1]
print c
np.savetxt("test.txt",pos,fmt='%.4e') 
