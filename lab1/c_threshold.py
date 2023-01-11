# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

image_rgb = load_sample_image('flower.jpg')   
image_bgr = np.flip(image_rgb, axis=-1)

# Window name in which image is displayed
window_name = 'c_image'

img = cv2.split(image_bgr)

#Color Channels Histograms // 0-blue; 1-green; 2-red
b_his = cv2.calcHist(img, [0], None, [256], [0, 255])
g_his = cv2.calcHist(img, [1], None, [256], [0, 255])
r_his = cv2.calcHist(img, [2], None, [256], [0, 255])

def ConvertToSingleChannel(img, idx):
    new = np.zeros_like(img)
    new[:,:,idx] = img[:,:,idx]
    return new
  
img_onlyB = ConvertToSingleChannel(image_bgr, 0)
img_onlyG = ConvertToSingleChannel(image_bgr, 1)
img_onlyR = ConvertToSingleChannel(image_bgr, 2)

r_his[r_his>10000] = 10000

plt.plot(r_his, label='red', color='r')
plt.plot(g_his, label="green", color='g')
plt.plot(b_his, label="blue", color='b')

# limit of the histogram
plt.xlim([0, 255])
plt.title('histogram')
# Place a legend on the Axes.
plt.legend()
#plt.show()

#cv2.imshow("red only", img_onlyR)

# Setting a threshold to segment the flower
img_onlyR[img_onlyR<100] = 0    

#cv2.imshow("red only with threshold", img_onlyR)

# the index whose R channel == 0
zero_idx = np.where(img_onlyR[:,:,2] == 0) 
# Makes the same indices zero in B channel
img_onlyB[zero_idx[0], zero_idx[1], 0] = 0   
img_onlyG[zero_idx[0], zero_idx[1], 1] = 0

# Since they all have one different non-zero channel, we can add them together.
img_segmented_BGR = img_onlyR+img_onlyB+img_onlyG   
#cv2.imshow("segmented_BGR", img_segmented_BGR)


# THRESH_BINARY:     bigger -> 255;   smaller -> 0
# THRESH_BINARY_INV: bigger -> 0;     smaller -> 255
# THRESH_TOZERO:     bigger -> color; smaller -> 0
# THRESH_TOZERO_INV: bigger -> 0;     smaller -> color
# THRESH_TRUNC:      bigger -> thre;  smaller -> color

#convert an image from one color space to another
img_gray = cv2.cvtColor(img_segmented_BGR, cv2.COLOR_BGR2GRAY)
#cv2.imshow("img_gray", img_gray)

# (Source image, threshold, mapped value (if >threshold), threshold_mode)
_, img_thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)   
#cv2.imshow("img_thresh", img_thresh)

# Automatically locates contours - an outline or a bounding shape
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )   

canvas = np.zeros(img_segmented_BGR.shape)
# (255,0,0), 3 -> color, thickness
cv2.drawContours(canvas, contours, -1, (255,0,0), 1)   # -1 to draw all contours,index of contours
#cv2.imshow("canvas", canvas)


# How to locate the contour around the flower?
#print(hierarchy.shape)
# Remove axes of length one
hierarchy = np.squeeze(hierarchy)
#print(hierarchy)
 
# [Next, Previous, First_Child, Parent]
# https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
# If Parent = -1, it is the top of the hierarchy. If Child = -1, it is the bottom of the hierarchy.
# Next and Previous refer to the other contours on the same hierarchical level
top = hierarchy[hierarchy[:,3] == -1]
#print(top)
# [[ 1 -1 -1 -1] 0
# [ 7  0  2 -1]  1
# [-1  1  8 -1]] 7

hierarchy[7,:]    # How do we know 7? It has many children
mask = np.zeros(img_gray.shape, np.uint8)
cv2.drawContours(mask, contours[1], -1, (255,255,255),1)
#cv2.imshow("mask", mask)


# Get the area inside
area_inside = np.empty(img_gray.shape, dtype=np.int8)
for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        # Determines whether the point is inside a contour, outside, or lies on an edge
        area_inside[i,j] = cv2.pointPolygonTest(contours[7], (j,i), measureDist=False)      

area_inside[area_inside == -1] = 0    # Get rid of -1 values
gray_inside = area_inside * 255
x,y = np.where(area_inside == 0)
img_segmented_BGR[x,y,:] = 0
#cv2.imshow("img_segmented_BGR",img_segmented_BGR)


cv2.waitKey(0) 
cv2.destroyAllWindows() 