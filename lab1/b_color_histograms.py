import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

image_rgb = load_sample_image('flower.jpg')   
image_bgr = np.flip(image_rgb, axis=-1)

# Window name in which image is displayed
window_name = 'b_image'

#plot these color channels histograms.
cv2.imshow("image_bgr", image_bgr)

def ConvertToSingleChannel(img, idx):
    new = np.zeros_like(img)
    new[:,:,idx] = img[:,:,idx]
    return new
    
img_onlyB = ConvertToSingleChannel(image_bgr, 0)
img_onlyG = ConvertToSingleChannel(image_bgr, 1)
img_onlyR = ConvertToSingleChannel(image_bgr, 2)

cv2.imshow("red only", img_onlyR)
cv2.imshow("green only", img_onlyG)
cv2.imshow("blue only", img_onlyB)

img = cv2.split(image_bgr)

#Color Channels Histograms // 0-blue; 1-green; 2-red
b_his = cv2.calcHist(img, [0], None, [256], [0, 255])
g_his = cv2.calcHist(img, [1], None, [256], [0, 255])
r_his = cv2.calcHist(img, [2], None, [256], [0, 255])

plt.plot(r_his, label='red', color='r')
plt.plot(g_his, label="green", color='g')
plt.plot(b_his, label="blue", color='b')

# limit of the histogram
plt.xlim([0, 255])
plt.title('histogram')
# Place a legend on the Axes.
plt.legend()

plt.show()