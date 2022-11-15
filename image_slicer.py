# For slicer image in ratio
import os
import cv2
from tqdm import tqdm
from image_function import imgSlicer

# Set path
paths = 'Immunofluorescence images/DAPI'

for file in tqdm(sorted(os.listdir(paths))):
    path = paths + '/' + file
    img = cv2.imread(path, 1)
    img_list = imgSlicer(img, type='d4')

    init = 0
    for each in img_list:
        row = init // 4
        col = init % 4
        filename = file[:-4] + '_' + str(row) + '_' + str(col) + '.png'
        cv2.imwrite('Dataset/4x4/' + filename, each)
        init += 1
        if init == 16:
            init = 0
