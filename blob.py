import cv2 
import pandas as pd
import numpy as np
from function import *

if __name__ == '__main__':
    #Read the image
    img = cv2.imread('Immunofluorescence images/1H_Nrf2_No_ADT_1_DAPI.tif', 0)
    print(img.shape)
    cv2.imshow('Original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = clahe(img)
    img = opening(img)
    imgSlicer(img)
    img = blobDetection(img)
