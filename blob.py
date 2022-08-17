import cv2 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#Read the image
img = cv2.imread('Dataset/Cell_test.jpeg', 0)
print(img.shape)

#Apply contrast stretching
alpha = 1.5
beta = 0
adj_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

cv2.imshow('original', img)
cv2.imshow('adjusted', adj_img)
cv2.waitKey()

#Plot Histogram of image
plt.hist(adj_img.ravel(), 256, [0,256], color='crimson')
plt.ylabel("Number Of Pixels", color='crimson')
plt.xlabel("Pixel Intensity- From 0-255", color='crimson')
plt.title("Histogram Showing Pixel Intensity And Corresponding Number Of Pixels", color='crimson')
plt.show()


#Parameter setup
params = cv2.SimpleBlobDetector_Params()
#Set Threshold
params.minThreshold = 25
params.maxThreshold = 255
#Set Filter by Area
params.filterByArea = True
params.minArea = 240
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
detector = cv2.SimpleBlobDetector_create(params)
keypoint = detector.detect(adj_img)
print('Number of Keypoint is ',len(keypoint))

#Draw blobs on the image
img_with_blob = cv2.drawKeypoints(adj_img, keypoint, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Cell_with_Keypoints', img_with_blob)
cv2.waitKey(0)
cv2.destroyAllWindows()