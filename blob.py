import cv2 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#Read the image
img = cv2.imread('Dataset/Cell_test.jpeg', 0)
print(img.shape)
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Apply contrast stretching
alpha = 2
beta = -1
img_con = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
cv2.imshow('Contrast', img_con)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Set Threshold
###Adaptive Threshold
img_athr = cv2.adaptiveThreshold(img_con, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow('Adaptive Threshold', img_athr)
###Normal Threshold
img_nthr = cv2.threshold(img_con, 25, 255, cv2.THRESH_TOZERO)[1]
cv2.imshow('Normal Threshold', img_nthr)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_thr = img_nthr

#Set Opening
kernel = np.ones((2,2), np.uint8)
img_opn = cv2.morphologyEx(img_thr, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', img_opn)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Image to plot
img_plot = img_opn

#Plot Histogram of image
#plt.hist(img_plot.ravel(), 256, [0,256], color='crimson')
#plt.ylabel("Number Of Pixels", color='crimson')
#plt.xlabel("Pixel Intensity- From 0-255", color='crimson')
#plt.title("Histogram Showing Pixel Intensity And Corresponding Number Of Pixels", color='crimson')
#plt.show()


#Parameter setup
params = cv2.SimpleBlobDetector_Params()

#Set Threshold
#params.minThreshold = 25
#params.maxThreshold = 255

#Set Filter by Area
params.filterByArea = True
params.minArea = 250
params.maxArea = 100000

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
keypoint = detector.detect(img_plot)
print('Number of Keypoint is ',len(keypoint))

#Draw blobs on the image
img_with_blob = cv2.drawKeypoints(img_plot, keypoint, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Cell_with_Keypoints', img_with_blob)
cv2.waitKey(0)
cv2.destroyAllWindows()