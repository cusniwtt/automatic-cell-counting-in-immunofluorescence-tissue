# For slicer image in ratio
import os
import cv2
from tqdm import tqdm

# Image slicer
def imgSlicer(img, type = 'd8'):
    # Slice ratio (base on 1920x1536)
    slice_ratio = {
        'd2': [960, 768, 2],
        'd3': [640, 512, 3],
        'd4': [480, 384, 4],
        'd6': [320, 256, 6],
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
paths = 'Dataset/Sample Image/'
split = 6
split_type = 'd6'
save_path = 'Dataset/Sample6x6/'

for file in tqdm(sorted(os.listdir(paths))):
    if file == '.DS_Store':
        continue
    path = paths + file
    img = cv2.imread(path, 1)
    img_list = imgSlicer(img, type=split_type)

    init = 0
    for each in img_list:
        row = init // split
        col = init % split
        filename = file[:-4] + '_' + str(row) + '_' + str(col) + '.png'
        cv2.imwrite(save_path + filename, each)
        init += 1
        if init == split**2:
            init = 0
