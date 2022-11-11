import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import fnmatch

# Read image only DAPI
def getPath(path, type='DAPI'):
    paths = []
    regex = '*' + type + '.tif'
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, regex):
            paths.append(file)
    return paths

# Plot image
def plotImage(img, title = 'Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Image slicer
def imgSlicer(img, type = 'd8'):
    # Slice ratio (base on 1920x1536)
    slice_ratio = {
        'd2': [960, 768, 2],
        'd4': [480, 384, 4],
        'd8': [240, 192, 8],
        'd16': [120, 96, 16],
    }
    # Slice image
    img_slice = []
    x = 0
    y = 0
    for i in range(0, slice_ratio[type][2]):
        for j in range(0, slice_ratio[type][2]):
            temp = img[y:y+slice_ratio[type][1]-1, x:x+slice_ratio[type][0]-1]
            img_slice.append(temp)
            x += slice_ratio[type][0]
        x = 0
        y += slice_ratio[type][1]
    return img_slice

# Apply contrast stretching
def contrastAbs(img, alpha = 2, beta = -1):
    img_con = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img_con

# Add more sharp in image
def sharpening(img, kernel):
    kernel_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    kernel_2 = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    if kernel == 1:
        im = cv2.filter2D(img, -1, kernel_1)
    elif kernel == 2:
        im = cv2.filter2D(img, -1, kernel_2)
    else:
        return print('Kernel not found: Use 1 or 2')
    return im

# Set Threshold
### Adaptive Threshold
def adaptiveThresholding(img, blockSize = 15, C = 3):
    img_athr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, C)
    return img_athr
### Normal Threshold
def normalThresholding(img, thresh = 127, maxval = 255, type = cv2.THRESH_TOZERO):
    img_nthr = cv2.threshold(img, thresh, maxval, type)[1]
    return img_nthr
### Gausian blur + Osthu threshold
def gaussianBlur(img, ksize = 5, sigmaX = 0):
    blur = cv2.GaussianBlur(img, (ksize,ksize), sigmaX)
    return blur
def ostuThresholding(img, minVal = 128, maxVal = 255):
    ostu, output = cv2.threshold(img, minVal, maxVal, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ostu , output
### CLAHE method
def clahe(img, clipLimit=5.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_clahe = clahe.apply(img)
    return img_clahe

# Morphological Operations
### Erosion
def morp_op_erosion(img, kernel = np.ones((3,3), np.uint8), iter = 1):
    img_ero = cv2.erode(img, kernel, iterations = iter)
    return img_ero
### Dilation
def morp_op_dilation(img, kernel = np.ones((3,3), np.uint8), iter = 1):
    img_dil = cv2.dilate(img, kernel, iterations = iter)
    return img_dil
### Set Opening
def morp_op_opening(img, kernel = np.ones((9,9), np.uint8)):
    img_opn = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img_opn
### Set Closing
def morp_op_closing(img, kernel = np.ones((9,9), np.uint8)):
    img_cls = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img_cls

# Plot Histogram of image
def plotHistogram(img):
    plt.hist(img.ravel(), 256, [0,256], color='crimson')
    plt.ylabel("Number Of Pixels", color='crimson')
    plt.xlabel("Pixel Intensity- From 0-255", color='crimson')
    plt.title("Histogram Showing Pixel Intensity And Corresponding Number Of Pixels", color='crimson')
    plt.show()

# Blob detection
def blobDetection(img):
    #Parameter setup
    params = cv2.SimpleBlobDetector_Params()
    #Set Threshold
    #params.minThreshold = 25
    #params.maxThreshold = 255
    #Set Filter by Area
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 1000
    #Set filter by Color (0 = black, 255 = white)
    params.filterByColor = False
    params.blobColor = 0
    #Set Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.maxCircularity = 1
    #Set Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1
    params.maxConvexity = 1
    #Set Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.maxInertiaRatio = 1

    #Create a detector with the parameters and detect blobs
    detector = cv2.SimpleBlobDetector_create(parameters=params)
    keypoint = detector.detect(img)
    return keypoint, len(keypoint)