# For slicer image in ratio
import os
import cv2
from tqdm import tqdm

# Image slicer
def imgSlicer(img, type = 'd8'):
    # Slice ratio (base on 1920x1536)
    slice_ratio = {
        'd2': [960, 768, 2],
        'd4': [480, 384, 4],
        'd8': [240, 192, 8],
        'd16': [120, 96, 16],
    }
    # Slice image
    img_slice = []
    x = 0
    y = 0
    for i in range(0, slice_ratio[type][2]):
        for j in range(0, slice_ratio[type][2]):
            temp = img[y:y+slice_ratio[type][1]-1, x:x+slice_ratio[type][0]-1]
            img_slice.append(temp)
            x += slice_ratio[type][0]
        x = 0
        y += slice_ratio[type][1]
    return img_slice

################# Start Here #################
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
