import cv2
import matplotlib.pyplot as plt
import numpy as np

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
    cv2.imshow('Contrast', img_con)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_con

# Add more sharp in image
def sharpening(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = cv2.filter2D(img, -1, kernel)
    cv2.imshow('Sharpening', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return im

# Set Threshold
### Adaptive Threshold
def adaptiveThresholding(img, blockSize = 15, C = 3):
    img_athr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, C)
    cv2.imshow('Adaptive Threshold', img_athr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_athr
### Normal Threshold
def normalThresholding(img, thresh = 127, maxval = 255, type = cv2.THRESH_TOZERO):
    img_nthr = cv2.threshold(img, thresh, maxval, type)[1]
    cv2.imshow('Normal Threshold', img_nthr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_nthr
### Gausian blur + Osthu threshold
def gaussianBlur(img, ksize = 5, sigmaX = 0):
    blur = cv2.GaussianBlur(img, (ksize,ksize), sigmaX)
    cv2.imshow('Gaussian Blur', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return blur
def ostuThresholding(img, minVal = 24, maxVal = 255, type = cv2.THRESH_BINARY + cv2.THRESH_OTSU):
    ostu = cv2.threshold(img, minVal, maxVal, type)[1]
    cv2.imshow('Ostu Threshold', ostu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return ostu
### CLAHE method
def clahe(img, clipLimit=5.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_clahe = clahe.apply(img)
    cv2.imshow('CLAHE', img_clahe)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_clahe

# Morphological Operations
### Erosion
def morp_op_erosion(img, kernel = np.ones((3,3), np.uint8), iter = 1):
    img_ero = cv2.erode(img, kernel, iterations = iter)
    cv2.imshow('Erosion', img_ero)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_ero
### Dilation
def morp_op_dilation(img, kernel = np.ones((3,3), np.uint8), iter = 1):
    img_dil = cv2.dilate(img, kernel, iterations = iter)
    cv2.imshow('Dilation', img_dil)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_dil
### Set Opening
def morp_op_opening(img, kernel = np.ones((9,9), np.uint8)):
    img_opn = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Opening', img_opn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_opn
### Set Closing
def morp_op_closing(img, kernel = np.ones((9,9), np.uint8)):
    img_cls = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Closing', img_cls)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
    print('Number of Keypoint is ',len(keypoint))

    #Draw blobs on the image
    img_with_blob = cv2.drawKeypoints(img, keypoint, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Cell_with_Keypoints', img_with_blob)
    cv2.waitKey(0)
    cv2.destroyAllWindows()