import cv2 
import pandas as pd
import numpy as np
from image_function import *
from metadata_function import *

paths = getPath('Immunofluorescence images/')

for p in paths:
    #Read the image
    path = 'Immunofluorescence images/' + p
    img = cv2.imread(path, 0)
    print(img.shape)
    cv2.imshow('Original', img)
    cv2.waitKey(100)
    cv2.destroyAllWindows()

    img = clahe(img, clipLimit=8.0, timer=100)
    img = gaussianBlur(img, ksize=3, sigmaX=1, timer=100)
    img = sharpening(img, timer=100)
    img_list = imgSlicer(img, timer=100)

    no = 0
    for i in range(len(img_list)):
        keypoint, key_no = blobDetection(img_list[i])
        no = no + key_no
        #Draw blobs on the image
        img_with_blob = cv2.drawKeypoints(img_list[i], keypoint, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Cell_with_Keypoints', img_with_blob)
        cv2.waitKey(30)
        cv2.destroyAllWindows()
    print('Total number of cells: ', no)