import cv2 
import pandas as pd
import numpy as np
import time

img = cv2.imread('Cell_test.jpeg', 1)
print(img.shape)
cv2.imshow('Cell_test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()