import numpy as np
import cv2
from scipy.ndimage import gaussian_gradient_magnitude
from skimage.measure import find_contours

from image_function import *
from metadata_function import *

paths = getPath('Immunofluorescence images/')
path = 'Immunofluorescence images/' + paths[0]
img = cv2.imread(path, 0)
print(img.shape)
plotImage(img, 'Original')

img = clahe(img, clipLimit=8.0, tileGridSize=(8,8))
plotImage(img, 'CLAHE')
img = gaussianBlur(img, ksize=5, sigmaX=3)
plotImage(img, 'Gaussian Blur')
img = sharpening(img, 1)
plotImage(img, 'Sharpening')

otsu, img = ostuThresholding(img, minVal=0, maxVal=255)
plotImage(img, 'OSTU Thresholding')

img = mo