'''
Created on 2014-5-18

@author: angelds
'''
"""
===============================
Histogram of Oriented Gradients
===============================


"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import cv2.cv as cv
from skimage.feature import hog
from skimage import data, color, exposure

img=cv2.imread("test1.png")
image = color.rgb2gray(img)
cv2.imwrite('testgrale.png',image)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
a=hog_image_rescaled
print a
#b=a.shpae[0]
c=np.size(a)
print c
