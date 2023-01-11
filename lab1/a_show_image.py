
import numpy as np
import cv2 as cv

from sklearn.datasets import load_sample_image

image = load_sample_image('flower.jpg')   

# Window name in which image is displayed
window_name = 'image'

# flip the array
img_bgr = np.flip(image, axis=-1)

# Using cv2.imshow() method 
# Displaying the image 
cv.imshow(window_name, img_bgr)

#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv.waitKey(0) 
  
#closing all open windows 
cv.destroyAllWindows() 

