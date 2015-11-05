'''
Created on 2014-5-27

@author: angelds
'''
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:17:24 2014

@author: jiangmaofei
"""
import numpy.linalg as ng
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as mp
import time
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
from numpy import dot
from numpy.linalg import inv
import biaozhi
#cap = cv2.VideoCapture(0) #read from camera
cap = cv2.VideoCapture('SampleVideo.avi') #read from video
time.sleep(1)
boxes = {'start':(),'end':()}
#cv2.rectangle(img, pt1, pt2, color)
#cv2.line(img, pt1, pt2, color) 
drawing_box = False
def hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), visualise=False, normalise=False):

    image = np.atleast_2d(image)
    if image.ndim > 3:
        raise ValueError("Currently only supports grey-level images")

    if normalise:
        image = sqrt(image)

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)
    gy[:-1, :] = np.diff(image, n=1, axis=0)

    magnitude = sqrt(gx ** 2 + gy ** 2)
    orientation = arctan2(gy, (gx + 1e-15)) * (180 / pi) + 90

    sy, sx = image.shape
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    for i in range(orientations):

        temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                            orientation, 0)
        temp_ori = np.where(orientation >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, magnitude, 0)

        #orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cy, cx))[cy/2::cy,cx/2::cx]
        orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cy, cx))[cy/2::cy,cx/2::cx]

    radius = min(cx, cy) // 2 - 1
    hog_image = None
    if visualise:
        hog_image = np.zeros((sy, sx), dtype=float)

    if visualise:
        from skimage import draw
        
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o in range(orientations):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    dx = radius * cos(float(o) / orientations * np.pi)
                    dy = radius * sin(float(o) / orientations * np.pi)
                    rr, cc = draw.bresenham(centre[0] - dx, centre[1] - dy,
                                            centre[0] + dx, centre[1] + dy)
                    hog_image[rr, cc] += orientation_histogram[y, x, o]


    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx,by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y + by, x:x + bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / sqrt(block.sum() ** 2 + eps)

    if visualise:
        return normalised_blocks.ravel(), hog_image
    else:
        return normalised_blocks.ravel()
    
count = 0
number = 0
biaozhi.num = 0

#cv2.namedWindow('HOG image')
def on_mouse(event, x, y, flags, params):
    if event == cv.CV_EVENT_LBUTTONDOWN:
        print 'Start Mouse Position: '+str(x)+', '+str(y)
        sbox = (x, y)
        boxes['start'] = sbox
        boxes['end'] = ()
        
        biaozhi.f_qishi = 1
    elif event == cv.CV_EVENT_LBUTTONUP:
        print 'End Mouse Position: '+str(x)+', '+str(y)
        ebox = (x, y)
        boxes['end'] = ebox
        biaozhi.f_qishi = 0
        biaozhi.f_biaozhi = 1
        
    

length = 0
capture=cv.CaptureFromFile("SampleVideo.avi")
frame = cv.QueryFrame(capture)
tx = frame.width
ty = frame.height

stx0 = 0
sty0 = 0
etx0 = 0
ety0 = 0
T = 14
N = T/2

xuan_shuzu = np.zeros((ty, tx, 130))
weizhi = np.zeros((2, 2, 130),np.int)
weizhi1 = np.zeros((2, 2, T*T),np.int)
zuobiao = np.zeros((2,130),np.int)

while True:
    cv2.namedWindow('real image')
    cv2.namedWindow('select image')
    cv2.namedWindow('gray image')
    cv.SetMouseCallback('real image', on_mouse, 0)
    if biaozhi.f_qishi == 0:
        count += 1
        flag,img = cap.read() 
        
    #gray_imgg = cv.CreateMat(frame.height, frame.width, cv.CV_8U) #Gray frame at t-1
    #cv.CvtColor(frame, gray_imgg, cv.CV_RGB2GRAY)
    #gray_imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    k = 0
    
    #flag,img = cap.read()
    #img = cv2.blur(img, (3,3))
    #vis = img.copy()
    
    if biaozhi.f_qishi == 1:
        flag = False
        k = 1
    
    if flag == True:    
               
       if biaozhi.f_biaozhi == 1:
           
           if (boxes['start'] != ()) and (boxes['end'] != ()) :
               print boxes['start'] ,boxes['end']
               poS = boxes['start']
               poE = boxes['end']
               #stx0 = poS[0]
               #sty0 = poS[1]
               #etx0 = poE[0]
               #ety0 = poE[1]
               
               cv2.rectangle(img,boxes['start'],boxes['end'],(0, 255, 0), 2)
               select_img = img[poS[1]:poE[1],poS[0]:poE[0]]
               xuan_tuxian = cv2.cvtColor(select_img, cv2.COLOR_BGR2GRAY)
               cv2.imshow('select image', select_img)
               cv2.imshow('gray image', xuan_tuxian)
               chushi_tezheng = hog(xuan_tuxian)
               length = np.size(chushi_tezheng)
               
               #num = count
               number = count
               biaozhi.f_yihua = 1
               biaozhi.f_biaozhi = 0
               print weizhi[:,:,0]
               print poS[1]
               print poE[1]
               print poS[0]
               print poE[0]
               
               #print xuan_shuzu[poS[1]:poE[1],poS[0]:poE[0],count]
       if biaozhi.f_yihua == 1:
           
           print biaozhi.num
           print count
           
           poS = boxes['start']
           poE = boxes['end']
           
           stx1 = poS[0]
           sty1 = poS[1]
           etx1 = poE[0]
           ety1 = poE[1]
           
           weizhi[0,0,0] = poS[1]
           weizhi[1,0,0] = poE[1]
           weizhi[0,1,0] = poS[0]
           weizhi[1,1,0] = poE[0]
           print weizhi
           juli_tezheng = 0
           
                              
           tezheng = np.zeros((length, T*T))
           H2 = np.zeros((length, 1))
           
           sty0 = weizhi[0,0,biaozhi.num] + N
           ety0 = weizhi[1,0,biaozhi.num] + N
           stx0 = weizhi[0,1,biaozhi.num] - N
           etx0 = weizhi[1,1,biaozhi.num] - N
           weizhi1[0,0,0] = sty0
           weizhi1[1,0,0] = ety0
           weizhi1[0,1,0] = stx0
           weizhi1[1,1,0] = etx0
           
           select_img1 = img[sty0:ety0,stx0:etx0]
           xuan_tuxian1 = cv2.cvtColor(select_img1, cv2.COLOR_BGR2GRAY)
            
           
           tezheng[:, 0] = hog(xuan_tuxian1)
           H2 = tezheng[:, 0] - chushi_tezheng
           juli_tezheng = ng.norm(H2)
           for i in xrange(0,T):
               for j in xrange(0,T):
                   weizhi1[0,0,T*i+j] = weizhi1[0,0,0] - i 
                   weizhi1[1,0,T*i+j] = weizhi1[1,0,0] - i
                   weizhi1[0,1,T*i+j] = weizhi1[0,1,0] + j
                   weizhi1[1,1,T*i+j] = weizhi1[1,1,0] + j
                   stx2 = weizhi1[0,1,T*i+j]
                   sty2 = weizhi1[0,0,T*i+j]
                   etx2 = weizhi1[1,1,T*i+j]
                   ety2 = weizhi1[1,0,T*i+j]
                   
                   select_img2 = img[sty2:ety2,stx2:etx2]
                   xuan_tuxian2 = cv2.cvtColor(select_img2, cv2.COLOR_BGR2GRAY)
                   tezheng[:,T*i+j] = hog(xuan_tuxian2)
                   H2 = tezheng[:,T*i+j] - chushi_tezheng
                   if ng.norm(H2) < juli_tezheng:
                        juli_tezheng = ng.norm(H2)
           for i in xrange(0,T):
                 for j in xrange(0,T):
                     H3 = tezheng[:,T*i+j] - chushi_tezheng
                     if juli_tezheng == ng.norm(H3):
                         weizhi[0,0,(biaozhi.num+1)] = weizhi1[0,0,T*i+j]
                         weizhi[1,0,(biaozhi.num+1)] = weizhi1[1,0,T*i+j]
                         weizhi[0,1,(biaozhi.num+1)] = weizhi1[0,1,T*i+j]
                         weizhi[1,1,(biaozhi.num+1)] = weizhi1[1,1,T*i+j]
              # h1 = hog(gray_img)
               #print h1 
            #cv2.imshow('HOG image', hog_img)
           
           #print boxes
           biaozhi.num += 1
       cv2.imshow('real image', img)
       
           
    else:
       if k == 0:
           biaozhi.f_yihua = 0
           break
   
    if cv2.waitKey(80) == 100:
       
        break
cap.release()

print count
print number
print tx, ty
print stx0, sty0, etx0, ety0
print weizhi[:,:,30]
print biaozhi.num
cap = cv2.VideoCapture('SampleVideo.avi')
boxes = {'start':(),'end':()}
count = 0
biaozhi.num = 0


#replay vedio
while True:
    cv2.namedWindow('real image')
    #cv2.namedWindow('select image')
    #cv2.namedWindow('gray image')
    #cv.SetMouseCallback('real image', on_mouse, 0)
    
    flag,img = cap.read() 
    count += 1
       
    if flag == True:
        if count >= number:
            sty0 = weizhi[0,0,biaozhi.num]
            ety0 = weizhi[1,0,biaozhi.num]
            stx0 = weizhi[0,1,biaozhi.num]
            etx0 = weizhi[1,1,biaozhi.num]
            boxes['start'] = (stx0,sty0)
            boxes['end'] = (etx0,ety0)
            zuobiao[0,biaozhi.num] = (stx0 + etx0)/2
            zuobiao[1,biaozhi.num] = (sty0 + ety0)/2
            cv2.rectangle(img,boxes['start'],boxes['end'],(0, 255, 0), 2)
            biaozhi.num += 1
        cv2.imshow('real image', img)    
    else:
        break
   
    if cv2.waitKey(80) == 100:
       
        break
cap.release()

#kalmanfilter
cdata = zuobiao
N=cdata.shape[1]
data = cdata
for i in xrange(0, N):
    if data[0,i] == 0:
       data[0,i] = cdata[0,i-3] 
    if data[1,i] == 0:
       data[1,i] = cdata[1,i-3]

Q = np.eye(4, dtype=int)
P = np.zeros((4,4,N), np.float)
PP = np.zeros((4,4,N), np.float)
P[:,:,0] = np.eye(4, dtype=int)
X_k = np.zeros((4,1,N), np.float)
X_kk = np.zeros((4,1,N), np.float)
Xk = np.zeros(N)
Yk = np.zeros(N)
a = np.array([data[0,0], data[1,0], 0, 0])
X_k[:,:,0] = a.reshape(4,1)

Z_k = np.zeros((2,1,N), np.float)
for i in xrange(0, N):
    a = np.array([data[0,i], data[1,i]])
    Z_k[:,:,i] = a.reshape(2,1)
print Z_k    
Xk[0] = data[0,0]
Yk[0] = data[1,0]
R = np.eye(2, dtype=int)
F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]
            ])
            
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]
            ])
#predict equation
def kf_predict(X, P, F, Q, B, U):
    X = dot(F, X) + dot(B, U)
    P = dot(F, dot(P, F.T)) + Q
    return(X,P)

#update equation
def kf_update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(IS, K.T))
    return (X,P)
for i in xrange(1, N):
    X_kk[:,:,i],PP[:,:,i] = kf_predict(X_k[:,:,i-1], P[:,:,i-1], F, Q, 0, 0)
    X_k[:,:,i], P[:,:,i] = kf_update(X_kk[:,:,i], PP[:,:,i], Z_k[:,:,i], H, R)
    Xk[i] = X_k[0,0,i]
    Yk[i] = X_k[1,0,i]
mp.figure()
mp.plot(Xk,Yk,"r+")
mp.plot(zuobiao[0,:],zuobiao[1,:],"g*")  
mp.xlim([0,360])
mp.ylim([0,200]) 
mp.show()