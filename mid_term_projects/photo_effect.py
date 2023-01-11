import numpy as np
import math
import cv2

# read the image 
img_name = 'img_1' 
img_bgr = cv2.imread(img_name+".jpg")

# Brightness and Contrast adjustments // linear
# g(i,j)=α⋅f(i,j)+β
def change_contrast(img, alpha, belta):
    new_img = np.copy(img)
    for a in range(len(img)):
        for b in range(len(img[0])):
            for c in range(len(img[0][0])):
                temp = int(img[a][b][c]*alpha + belta)
                new_img[a][b][c] = (temp > 255 and 255) or (temp > 0 and temp) or 0
    return new_img

# solarization
def solarization(img, threshold):
    new_img = np.copy(img)
    for a in range(len(img)):
        for b in range(len(img[0])):
            for c in range(len(img[0][0])):
                if img[a][b][c] > threshold:
                    new_img[a][b][c] = 255-img[a][b][c]
    return new_img

# exposure
def invert(img, threshold):
    new_img = np.copy(img)
    for a in range(len(img)):
        for b in range(len(img[0])):
            for c in range(len(img[0][0])):
                new_img[a][b][c] = 255-img[a][b][c]
    return new_img

# Sepia Filter
def sepia(img):
    new_img = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            r = int(img[i][j][2])
            g = int(img[i][j][1])
            b = int(img[i][j][0])
            R = int(0.393*r + 0.769*g + 0.189*b)
            G = int(0.349*r + 0.686*g + 0.168*b)
            B = int(0.272*r + 0.534*g + 0.131*b)
            new_img[i][j][2] = (R > 255 and 255) or (R > 0 and R) or 0
            new_img[i][j][1] = (G > 255 and 255) or (G > 0 and G) or 0
            new_img[i][j][0] = (B > 255 and 255) or (B > 0 and B) or 0
    return new_img

# comic Filter
def comic(img):
    new_img = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            r = int(img[i][j][2])
            g = int(img[i][j][1])
            b = int(img[i][j][0])
            R = abs(b-g+b+r)*g/256
            G = abs(b-g+b+r)*r/256
            B = abs(g-b+g+r)*r/256
            gray = (R+G+B)/3
            R = gray + 10
            G = gray + 10
            B = gray
            new_img[i][j][2] = (R > 255 and 255) or (R > 0 and R) or 0
            new_img[i][j][1] = (G > 255 and 255) or (G > 0 and G) or 0
            new_img[i][j][0] = (B > 255 and 255) or (B > 0 and B) or 0     
    return new_img

def casting(img):
    new_img = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            r = int(img[i][j][2])
            g = int(img[i][j][1])
            b = int(img[i][j][0])
            R = r*128/(g+b+1)
            G = g*128/(r+b+1)
            B = b*128/(g+r+1)
            new_img[i][j][2] = (R > 255 and 255) or (R > 0 and R) or 0
            new_img[i][j][1] = (G > 255 and 255) or (G > 0 and G) or 0
            new_img[i][j][0] = (B > 255 and 255) or (B > 0 and B) or 0     
    return new_img

def frozen(img):
    new_img = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            r = int(img[i][j][2])
            g = int(img[i][j][1])
            b = int(img[i][j][0])
            R = abs(r-g-b)*3/2
            G = abs(g-b-r)*3/2
            B = abs(b-g-r)*3/2
            new_img[i][j][2] = (R > 255 and 255) or (R > 0 and R) or 0
            new_img[i][j][1] = (G > 255 and 255) or (G > 0 and G) or 0
            new_img[i][j][0] = (B > 255 and 255) or (B > 0 and B) or 0     
    return new_img

def show_effect():
    img_contrast = change_contrast(img_bgr, 1.5, 20)
    img_solar = solarization(img_bgr, 130)
    img_exposure = invert(img_bgr, 128)
    img_sepia = sepia(img_bgr)
    img_comic = comic(img_bgr)
    img_casting = casting(img_bgr)
    img_frozen = frozen(img_bgr)
    
    img_line1 = np.hstack((img_bgr, img_contrast, img_solar, img_exposure))
    img_line2 = np.hstack((img_sepia, img_comic, img_casting, img_frozen))
    imgs = np.vstack((img_line1, img_line2))
    cv2.imwrite(img_name+"_effect.jpg", imgs)
    cv2.imshow("photo_effect_"+img_name, imgs)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

show_effect()
