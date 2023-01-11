import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

image_rgb = load_sample_image('flower.jpg')   
image_bgr = np.flip(image_rgb, axis=-1)

'''
Hue
Red falls between 0 and 60 degrees.
Yellow falls between 61 and 120 degrees.
Green falls between 121 and 180 degrees.
Cyan falls between 181 and 240 degrees.
Blue falls between 241 and 300 degrees.
Magenta falls between 301 and 360 degrees.
Saturation
Saturation describes the amount of gray in a particular color, from 0 to 100 percent. 
Reducing this component toward zero introduces more gray and produces a faded effect. 
Sometimes, saturation appears as a range from 0 to 1, where 0 is gray, and 1 is a primary color.
Value (or Brightness)
Value works in conjunction with saturation and describes the brightness or intensity of the color, 
from 0 to 100 percent, where 0 is completely black, and 100 is the brightest and reveals the most color.
'''
#HSV Color Model (hue, saturation, value)
img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_hsv)
#cv2.imshow("h",h)
#cv2.imshow("s",s)
#cv2.imshow("v",v)

# Hue
for i in range(5):
  h = h.astype(int)   # Originally, type is uint8 - only takes values (0,255)
  h = np.clip(h+i*40, 0, 180)
  h = h.astype('uint8') 
  img_hsv_saturated = cv2.merge((h,s,v))
  img_bgr_saturated = cv2.cvtColor(img_hsv_saturated, cv2.COLOR_HSV2BGR)
  cv2.imshow("img_bgr_saturated"+str(i), img_bgr_saturated)

# Saturation
h,s,v = cv2.split(img_hsv)
for i in range(5):
  s = s.astype(int)   # Originally, type is uint8 - only takes values (0,255)
  s = np.clip(s-i*50, 0, 255)
  s = s.astype('uint8') 
  img_hsv_saturated = cv2.merge((h,s,v))
  img_bgr_saturated = cv2.cvtColor(img_hsv_saturated, cv2.COLOR_HSV2BGR)
  #cv2.imshow("img_bgr_saturated"+str(i), img_bgr_saturated)

h_his = cv2.calcHist(h, [2], None, [256], [0, 255])
plt.plot(h_his, label='h', color='r')
plt.show()
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 