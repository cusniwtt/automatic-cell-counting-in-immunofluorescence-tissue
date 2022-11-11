import pandas as pd
import numpy as np
import cv2
from image_function import *
from metadata_function import *

paths = getPath('Immunofluorescence images/')

for p in paths:
    #Read the image

    # Fix path before run code

    path = 'Immunofluorescence images/' + p
    img = cv2.imread(path, 0)
    cv2.imshow('Original', img)
    print(img.shape)

    img = clahe(img, clipLimit=8.0, tileGridSize=(8,8))
    plotImage(img, 'CLAHE')
    img = gaussianBlur(img, ksize=5, sigmaX=3)
    plotImage(img, 'Gaussian Blur')
    img = sharpening(img, 1)
    plotImage(img, 'Sharpening')
    otsu, img = ostuThresholding(img, minVal=24, maxVal=255)
    plotImage(img, 'OSTU Thresholding')
    img = morp_op_erosion(img, kernel=np.ones((3,3), np.uint8))
    plotImage(img, 'Erosion')

    # If image not slice, run this code
    #img_list = imgSlicer(img)

    no = 0
    for i in range(len(img_list)):
        keypoint, key_no = blobDetection(img_list[i])
        no = no + key_no
        #Draw blobs on the image
        img_with_blob = cv2.drawKeypoints(img_list[i], keypoint, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #Save the image
        cv2.imwrite('Blob detected/' + p[:-4] + '_' + str(i) + '.jpg', img_with_blob)
        cv2.imshow('Blob detected', img_with_blob)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print('Total number of cells: ', no)