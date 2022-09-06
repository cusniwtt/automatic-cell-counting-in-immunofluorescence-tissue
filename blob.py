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

    img = clahe(img, clipLimit=8.0, tileGridSize=(8,8))
    img = gaussianBlur(img, ksize=5, sigmaX=1)
    img = sharpening(img)
    img_list = imgSlicer(img)

    no = 0
    for i in range(len(img_list)):
        keypoint, key_no = blobDetection(img_list[i])
        no = no + key_no
        #Draw blobs on the image
        img_with_blob = cv2.drawKeypoints(img_list[i], keypoint, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #Save the image
        cv2.imwrite('Blob detected/' + p[:-4] + '_' + str(i) + '.jpg', img_with_blob)
    print('Total number of cells: ', no)