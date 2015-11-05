'''
Created on 2014-5-17

@author: angelds
'''

import sys
import cv2
import cv2.cv as cv



def inside(r, q):
    (rx, ry), (rw, rh) = r
    (qx, qy), (qw, qh) = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


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
        tl = (rx + int(rw*0.1), ry + int(rh*0.07))
        br = (rx + int(rw*0.9), ry + int(rh*0.87))
        cv.Rectangle(img, tl, br, (0, 255, 0), 3)
        
    cv.ShowImage("people detection demo", img)
    c = cv.WaitKey(0)
    if c == ord('q'):
        break
cv.DestroyAllWindows()
