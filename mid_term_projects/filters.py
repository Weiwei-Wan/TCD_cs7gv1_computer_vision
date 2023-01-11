import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

image_rgb = load_sample_image('flower.jpg')   
image_bgr = np.flip(image_rgb, axis=-1)

# Averaging filter
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(image_bgr,-1,kernel)

#cv2.imshow("image_bgr", image_bgr)
#cv2.imshow("dst", dst)

# Gaussian filter
print(cv2.getGaussianKernel(5, sigma = 1))  # Returns coefficients in 1D
blur = cv2.GaussianBlur(image_bgr,(5,5),0)
#cv2.imshow("blur", blur)

# Apply this to Gaussian filter in the lab
# How to get 2D kernel - https://stackoverflow.com/questions/61394826/how-do-i-get-to-show-gaussian-kernel-for-2d-opencv#:~:text=To%20see%20the%20full%202D,invariant%20functions%20(%3D%3Dconvolution%20filters).

img=np.zeros((7,7))
img[3,3]=1

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
