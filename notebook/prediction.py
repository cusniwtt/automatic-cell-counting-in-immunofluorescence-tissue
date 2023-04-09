# Importing libraries
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import plotly.express as px
from skimage import io
import PIL
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

# Importing specific functions
from imagemks_function import get_default_parameter, cell_counting
from skimage.io import imread
from skimage.transform import resize

def dsc_cal(img1, img2):
    img1 = img1.astype(bool)
    img2 = img2.astype(bool)
    intersection = np.logical_and(img1, img2)
    dsc = 2 * intersection.sum() / (img1.sum() + img2.sum())
    return dsc

def iou_cal(img1, img2):
    img1 = img1.astype(bool)
    img2 = img2.astype(bool)
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    iou = intersection.sum() / union.sum()
    return iou

if __name__ == '__main__':
    # Get Test data
    # Get and resize test images
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3
    TEST_PATH = 'stage1_test/'
    test_ids = next(os.walk(TEST_PATH))[1]
    dir_path = ''


    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_

        #Read images iteratively
        img = imread(dir_path + path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

        #Get test size
        sizes_test.append([img.shape[0], img.shape[1]])

        #Resize image to match training data
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        #Append image to numpy array for test dataset
        X_test[n] = img

        #Read corresponding mask files iteratively
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

        #Looping through masks
        for mask_file in next(os.walk(path + '/masks/'))[2]:

            # Remove .DS_Store file
            if mask_file == '.DS_Store':
                continue
            
            #Read individual masks
            mask_ = imread(dir_path + path + '/masks/' + mask_file)

            #Expand individual mask dimensions
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)

            #Overlay individual masks to create a final mask for corresponding image
            try:
                mask = np.maximum(mask, mask_)
            except:
                print(mask_file)

        #Append mask to numpy array for train dataset
        Y_test[n] = mask

    print('Done!')

    # Predict with ImageMKS
    # Get default parameters
    params = get_default_parameter()
    Y_pred = []
    for img in X_test:
        # Predict
        img_label, pred = cell_counting(img, params)
        # Visualize
        pred = np.array(pred, dtype=np.uint8)
        pred = np.expand_dims(pred, axis=-1)
        Y_pred.append(pred)
    Y_pred = np.array(Y_pred)

    # Calculate DSC and IoU
    dsc = []
    iou = []
    for i in range(len(Y_pred)):
        dsc.append(dsc_cal(Y_pred[i], Y_test[i]))
        iou.append(iou_cal(Y_pred[i], Y_test[i]))
    df = pd.DataFrame({'DSC': dsc, 'IoU': iou})
    print(df)
    print(df.describe())