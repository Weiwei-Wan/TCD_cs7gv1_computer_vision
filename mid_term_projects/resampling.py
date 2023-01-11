import numpy as np
import cv2
import math
import photo_enhance as enh

# read the image 
def read_img(name): 
    img_bgr = cv2.imread(name+".jpg")
    print(img_bgr.shape)
    return img_bgr

# Nearest Neighbor Interpolation
def nearInter(img, fold):
    new_img = np.resize(img, (int(len(img)*fold), int(len(img[0])*fold), len(img[0][0])))
    print(new_img.shape)
    for i in range(len(new_img)):
        for j in range(len(new_img[0])):
            x = i/fold
            y = j/fold
            # treat 0.5
            if x - int(x) == 0.5:
                x = int(x)
            else:
                x = round(x)
            if y - int(y) == 0.5:
                y = int(y)
            else:
                y = round(y)
            for k in range(len(new_img[0][0])):
                new_img[i][j][k] = img[x][y][k]
    return new_img

# Bilinear Interpolation
def BiliInter(img, fold):
    new_img = np.resize(img, (int(len(img)*fold), int(len(img[0])*fold), len(img[0][0])))
    for i in range(len(new_img)):
        for j in range(len(new_img[0])):
            i_left = int(i/fold)
            j_left = int(j/fold)
            i_right = math.ceil(i/fold)
            j_right = math.ceil(j/fold)
            x = i/fold - i_left
            y = j/fold - j_left
            # treat edge
            if i_right > len(img)-1:
                i_right = len(img)-1
                x = 0
            if j_right > len(img[0])-1:
                j_right = len(img[0])-1
                y = 0
            for k in range(len(new_img[0][0])):
                new_img[i][j][k] = (1-y)*(x*img[i_right][j_left][k] + (1-x)*img[i_left][j_left][k]) + y*(x*img[i_right][j_right][k] + (1-x)*img[i_left][j_right][k])
    return new_img

# Bicubic Interpolation
def BicubicInter(img, fold, a):
    new_img = np.resize(img, (int(len(img)*fold), int(len(img[0])*fold), len(img[0][0])))
    # add extra edges
    for i in range(2):
        img = np.insert(img, 0, img[0], axis = 0)
        img = np.insert(img, len(img), img[len(img)-1], axis = 0)
        img = np.insert(img, 0, img[:, 0, :], axis = 1)
        img = np.insert(img, len(img[0]), img[:, len(img[0])-1, :], axis = 1)
    for i in range(len(new_img)):
        for j in range(len(new_img[0])):
            for k in range(len(new_img[0][0])):
                i_left = int(i/fold)-1
                j_left = int(j/fold)-1
                temp = 0
                for m in range(i_left, i_left+4):
                    for n in range(j_left, j_left+4):
                        temp += GetBicubicWeigh(a, abs(i/fold-m))*GetBicubicWeigh(a, abs(j/fold-n))*img[m+2][n+2][k]
                new_img[i][j][k] = (temp > 255 and 255) or (temp > 0 and temp) or 0
    return new_img

def GetBicubicWeigh(a, x):
    w = 0
    if x <= 1:
        w = (a+2)*pow(x,3) - (a+3)*pow(x,2) + 1
    elif x < 2:
        w = a*pow(x,3) - 5*a*pow(x,2) + 8*a*x - 4*a
    return w

# Lanczos Interpolation
def LanczosInter(img, fold, a):
    new_img = np.resize(img, (int(len(img)*fold), int(len(img[0])*fold), len(img[0][0])))
    # add extra edges
    for i in range(a):
        img = np.insert(img, 0, img[0], axis = 0)
        img = np.insert(img, len(img), img[len(img)-1], axis = 0)
        img = np.insert(img, 0, img[:, 0, :], axis = 1)
        img = np.insert(img, len(img[0]), img[:, len(img[0])-1, :], axis = 1)
    for i in range(len(new_img)):
        for j in range(len(new_img[0])):
            for k in range(len(new_img[0][0])):
                i_left = int(i/fold)-a+1
                j_left = int(j/fold)-a+1
                temp = 0
                for m in range(i_left, i_left+2*a):
                    for n in range(j_left, j_left+2*a):
                        temp += GetLanczosWeigh(a, i/fold-m)*GetLanczosWeigh(a, j/fold-n)*img[m+a][n+a][k]
                new_img[i][j][k] = (temp > 255 and 255) or (temp > 0 and temp) or 0
    return new_img

def GetLanczosWeigh(a, x):
    w = 0
    if x == 0:
        w = 1
    elif abs(x) < a:
        w = a*math.sin(math.pi*x)*math.sin(math.pi*x/a)/pow(math.pi*x, 2)
    return w


# laplacian pyramid
def subtract(img_a, img_b):
    while len(img_a) > len(img_b):
        img_a = np.delete(img_a, len(img_a)-1, axis=0)
    while len(img_a[0]) > len(img_b[0]):
        img_a = np.delete(img_a, len(img_a[0])-1, axis=1)

    result_img = np.copy(img_a)
    for i in range(len(img_a)):
        for j in range(len(img_a[0])):
            for k in range(len(img_a[0][0])):
                result = int(img_a[i][j][k]) - int(img_b[i][j][k])
                result = (result > 255 and 255) or (result > 0 and result) or 0
                result_img[i][j][k] = result
    return result_img
        
# generate Gaussian pyramid
def gaussian_pyrDown(img, level):
    temp_img = np.copy(img)
    pyrDown_imgs = []
    kernel = np.array([[ 1,  4,  6,  4,  1],
                       [ 4, 16, 24, 16,  4],
                       [ 6, 24, 36, 24,  6],
                       [ 4, 16, 24, 16,  4],
                       [ 1,  4,  6,  4,  1]])/256
    # temp_img : the Gaussian/original image
    # temp : list, generating new image
    # temp_i : list, row of new image
    for f in range(level):
        temp_img = enh.convolve(temp_img, kernel)
        temp = []
        for i in range(len(temp_img)):
            if i%2 == 1:
                temp_i = []
                for j in range(len(temp_img[0])):
                    if j%2 == 1:
                        temp_i.append(temp_img[i][j])
                temp.append(temp_i)
        temp_img = np.array(temp)
        cv2.imwrite("gaussian_pyrDown_"+str(f+1)+".jpg", temp_img)
        #cv2.imshow("gaussian_pyrDown_"+str(f+1), temp_img)
        pyrDown_imgs.append(temp_img)
    return pyrDown_imgs

def gaussian_pyrUp(img, level):
    temp_img = np.copy(img)
    pyrUp_imgs = []
    kernel = np.array([[ 1,  4,  6,  4,  1],
                       [ 4, 16, 24, 16,  4],
                       [ 6, 24, 36, 24,  6],
                       [ 4, 16, 24, 16,  4],
                       [ 1,  4,  6,  4,  1]])/64
    for f in range(level):
        i = 0
        while i < 2*len(img):
            temp_img = np.insert(temp_img, i, values = np.zeros((len(temp_img[0]), 3)), axis = 0)
            i += 2
        j = 0
        while j < 2*len(img[0]):
            temp_img = np.insert(temp_img, j, values = np.zeros((len(temp_img), 3)), axis = 1)
            j += 2
        temp_img = enh.convolve(temp_img, kernel) 
        #cv2.imwrite("gaussian_pyrUp_"+str(f+1)+".jpg", temp_img)
        #cv2.imshow("gaussian_pyrUp_"+str(f+1), temp_img)
        pyrUp_imgs.append(temp_img)
    return pyrUp_imgs

def laplacian_up(img, level):
    # need information from gaussian_pyrDown
    pyramid_imgs = gaussian_pyrDown(img, level)
    for i in range(level-1,-1,-1):
        if i == 0:
            expand = gaussian_pyrUp(pyramid_imgs[i], 1)[0]
            lapls = subtract(img, expand)
        else:
            expand = gaussian_pyrUp(pyramid_imgs[i], 1)[0]
            lapls = subtract(pyramid_imgs[i-1],expand)
        cv2.imwrite("gaussian_up_"+str(i+1)+".jpg", expand)
        cv2.imwrite("laplacian_up_"+str(i+1)+".jpg", lapls)
        #cv2.imshow("lapls_down_"+str(i+1),lapls)

def show_imgs(name):
    img_bgr = read_img(name)
    #temp_img = laplacian_up(img_bgr, 3) 
    
    temp_img = img_bgr
    for i in range(1, 4):
        temp_img = BicubicInter(temp_img, 0.5, -0.5) 
        cv2.imwrite("/Users/wanjiang/Documents/TCD/cs7gv1_computer_vision/mid_term_projects/BicubicInter-down-"+str(i)+".jpg", temp_img)
    for i in range(1, 4):
        temp_img = BicubicInter(temp_img, 2, -0.5) 
        cv2.imwrite("/Users/wanjiang/Documents/TCD/cs7gv1_computer_vision/mid_term_projects/BicubicInter-up-"+str(i)+".jpg", temp_img)

    temp_img = img_bgr
    for i in range(1, 4):
        temp_img = nearInter(temp_img, 0.5) 
        cv2.imwrite("/Users/wanjiang/Documents/TCD/cs7gv1_computer_vision/mid_term_projects/nearInter-down-"+str(i)+".jpg", temp_img)
    for i in range(1, 4):
        temp_img = nearInter(temp_img, 2) 
        cv2.imwrite("/Users/wanjiang/Documents/TCD/cs7gv1_computer_vision/mid_term_projects/nearInter-up-"+str(i)+".jpg", temp_img)

    temp_img = img_bgr
    for i in range(1, 4):
        temp_img = BiliInter(temp_img, 0.5) 
        cv2.imwrite("/Users/wanjiang/Documents/TCD/cs7gv1_computer_vision/mid_term_projects/BiliInter-down-"+str(i)+".jpg", temp_img)
    for i in range(1, 4):
        temp_img = BiliInter(temp_img, 2) 
        cv2.imwrite("/Users/wanjiang/Documents/TCD/cs7gv1_computer_vision/mid_term_projects/BiliInter-up-"+str(i)+".jpg", temp_img)
    temp_img = img_bgr
    for i in range(1, 4):
        temp_img = LanczosInter(temp_img, 0.5, 2) 
        cv2.imwrite("/Users/wanjiang/Documents/TCD/cs7gv1_computer_vision/mid_term_projects/LanczosInter-down-"+str(i)+".jpg", temp_img)
    for i in range(1, 4):
        temp_img = LanczosInter(temp_img, 2, 2) 
        cv2.imwrite("/Users/wanjiang/Documents/TCD/cs7gv1_computer_vision/mid_term_projects/LanczosInter-up-"+str(i)+".jpg", temp_img)
    

show_imgs('img_1')
