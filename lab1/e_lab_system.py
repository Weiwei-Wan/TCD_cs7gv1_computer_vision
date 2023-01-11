import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

image_rgb = load_sample_image('flower.jpg')   
image_bgr = np.flip(image_rgb, axis=-1)

'''L*: Lightness
a*: Red/Green Value
b*: Blue/Yellow Value'''
# LAB Color Model
img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
l,a,b = cv2.split(img_lab)

# cv2.imshow("l",l)
# cv2.imshow("a",a)
# cv2.imshow("b",b)

def ConvertToSingleChannel(img, idx):
    new = np.zeros_like(img)
    new[:,:,idx] = img[:,:,idx]
    return new

img_onlyL = ConvertToSingleChannel(img_lab, 0)
img_onlyA = ConvertToSingleChannel(img_lab, 1)
img_onlyB = ConvertToSingleChannel(img_lab, 2)


cv2.imshow("img_onlyL", cv2.cvtColor(img_onlyL, cv2.COLOR_Lab2BGR))
cv2.imshow("img_onlyA", cv2.cvtColor(img_onlyA, cv2.COLOR_Lab2BGR))
cv2.imshow("img_onlyB", cv2.cvtColor(img_onlyB, cv2.COLOR_Lab2BGR))

l_his = cv2.calcHist(l, [2], None, [256], [0, 255])
plt.plot(l_his, label='h', color='r')
plt.show()