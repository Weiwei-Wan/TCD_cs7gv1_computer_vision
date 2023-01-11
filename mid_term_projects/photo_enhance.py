import numpy as np
import math
import cv2
from scipy.linalg import pascal

# read the image 
def read_img(name): 
    img_bgr = cv2.imread(name+".jpg")
    print(img_bgr.shape)
    return img_bgr

# convolve image with the kernel
def convolve(img, kernel):
    x, y = kernel.shape 
    new_img = np.copy(img)
    # add edges for convolve
    for i in range(x//2):
        img = np.insert(img, 0, values = np.zeros((len(img[0]), 3)), axis = 0)
        img = np.insert(img, len(img), values = np.zeros((len(img[0]), 3)), axis = 0)
    for i in range(y//2):
        img = np.insert(img, 0, values = np.zeros((len(img), 3)), axis = 1)
        img = np.insert(img, len(img[0]), values = np.zeros((len(img), 3)), axis = 1)

    for a in range(len(new_img)):
        for b in range(len(new_img[0])):
            im_sum = np.zeros(len(new_img[0][0]))
            for i in range(x):
                for j in range(y):
                    for t in range(len(im_sum)):
                        img_val = int(img[a+i][b+j][t])
                        im_sum[t] += kernel[i][j] * img_val
            for t in range(len(im_sum)):
                im_sum[t] = (im_sum[t]>255 and 255) or (im_sum[t]>0 and im_sum[t]) or 0
            new_img[a][b] = im_sum
    print("wooooooooorking")
    return new_img

# box_blur
def box_blur(img, fold):
    kernel = np.ones((fold,fold))/fold/fold
    return convolve(img, kernel)

# find edge 
def find_edge(img):
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1],])
    return convolve(img, kernel)

def sharpen(img):
    kernel = np.array([[0,  -1,  0],
                       [-1,  5, -1],
                       [0,  -1,  0],])
    return convolve(img, kernel)

def gaussian_blur(img, fold, sigma):
    kernel = np.zeros((fold, fold))
    for i in range(fold):
        for j in range(fold):
            kernel[i][j] = math.exp(-((i-fold//2)**2+(j-fold//2)**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    
    sum_kernel = sum(sum(kernel))
    kernel = kernel/sum_kernel
    #kernel = np.array([[ 1,  4,  6,  4,  1],
    #                   [ 4, 16, 24, 16,  4],
    #                   [ 6, 24, 36, 24,  6],
    #                   [ 4, 16, 24, 16,  4],
    #                   [ 1,  4,  6,  4,  1]])/256
    return convolve(img, kernel)

def gaussian_blur2(img):
    kernel = np.array([[ 1,  4,  6,  4,  1],
                       [ 4, 16, 24, 16,  4],
                       [ 6, 24, 36, 24,  6],
                       [ 4, 16, 24, 16,  4],
                       [ 1,  4,  6,  4,  1]])/64
    return convolve(img, kernel)

def median_blur(img, fold):
    new_img = np.copy(img)
    dis = fold//2
    for a in range(len(img)):
        for b in range(len(img[0])):
            im_sum = np.zeros(len(img[0][0]))
            row_left = ((a-dis > 0) and (a-dis)) or 0
            row_right = ((a+dis < len(img)) and (a+dis+1)) or (len(img)+1)
            col_left = ((b-dis > 0) and (b-dis)) or 0
            col_right = ((b+dis < len(img[0])) and (b+dis+1)) or (len(img[0])+1)
            for t in range(len(im_sum)):
                im_sum[t] = np.median(img[row_left:row_right, col_left:col_right, t])
            new_img[a][b] = im_sum
    return new_img

def bilateral_blur(img, size, sigmaColor, sigmaSpace):
    dis = size//2
    new_img = np.copy(img)
    # add edges for convolve
    for i in range(dis):
        img = np.insert(img, 0, values = np.zeros((len(img[0]), 3)), axis = 0)
        img = np.insert(img, len(img), values = np.zeros((len(img[0]), 3)), axis = 0)
        img = np.insert(img, 0, values = np.zeros((len(img), 3)), axis = 1)
        img = np.insert(img, len(img[0]), values = np.zeros((len(img), 3)), axis = 1)

    for a in range(len(new_img)):
        for b in range(len(new_img[0])):
            for c in range(len(new_img[0][0])):
                numerator = 0
                denominator = 0
                fij = int(new_img[a][b][c])
                for k in range(size):
                    for l in range(size):
                        fkl = int(img[a-dis+k][b-dis+l][c])
                        color_fac = math.exp(-((fij-fkl)**2)/(2*sigmaColor**2))
                        space_fac = math.exp(-((k-dis)**2+(l-dis)**2)/(2*sigmaSpace**2))
                        w_ijkl = color_fac*space_fac
                        numerator += fkl * w_ijkl
                        denominator += w_ijkl
                new_img[a][b][c] = numerator/denominator
    return new_img

# binomial filter 
def Binomial2D(img, R):
    mask1d = np.diag(np.fliplr(pascal(2*R+1))) / pow(4, R)
    mask2d = np.multiply(mask1d, np.array([mask1d]).T)
    new_img = convolve(img, mask2d)
    return new_img

def show_effects(name):
    img_bgr = read_img(name)
    img_box_blur = box_blur(img_bgr, 5)
    #img_edge = find_edge(img_bgr)
    #img_sharpen = sharpen(img_bgr)
    img_gaussian_blur = gaussian_blur(img_bgr, 9, 1)
    img_median_blur = median_blur(img_bgr, 9)
    img_bilateral_blur = bilateral_blur(img_bgr, 9, 20, 20)
    img_bionomial_blur = Binomial2D(img_bgr, 9)

    img_line1 = np.hstack((img_bgr, img_box_blur, img_gaussian_blur))
    img_line2 = np.hstack((img_median_blur, img_bilateral_blur, img_bionomial_blur))
    imgs = np.vstack((img_line1, img_line2))
    cv2.imwrite(name+"_blur.jpg", imgs)
    cv2.imshow("photo_enhance_"+name, imgs)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

#show_effects('img_14')