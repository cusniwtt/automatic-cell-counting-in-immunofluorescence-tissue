# Importing libraries
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import plotly.express as px
from skimage import io
import os
import PIL
import time
import sys

# Importing specific functions
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.exposure import equalize_adapthist
from skimage.feature import corner_peaks
from skimage.segmentation import relabel_sequential, watershed
from skimage.measure import regionprops
from skimage.color import label2rgb

# Importing imageMKS_custom functions
from imagemks_custom.filters.fftgaussian import fftgauss
from imagemks_custom.filters.fftedges import local_avg
from imagemks_custom.filters.morphological import smooth_binary
from imagemks_custom.structures.shapes import circle, donut
from imagemks_custom.masking.fourier import maskfourier
from imagemks_custom.visualization.borders import make_boundary_image
from imagemks_custom.workflows.fluorescentcells import segment_fluor_cells, default_parameters

def vis(img, title=None, imgsize=(640, 512), mode='gray'):
    fig = px.imshow(img, title=title, color_continuous_scale='gray')
    fig.update_layout(autosize=False, width=imgsize[0], height=imgsize[1],)
    fig.show()

def labelvis(A, L, bg_color='b', engine='plotly'):
    bg_color_code = {
        'b': (0.1,0.1,0.5),
        'g': (0.1,0.5,0.1),
    }
        
    A = label2rgb(L, A, bg_label=0, bg_color=bg_color_code[bg_color], alpha=0.1, image_alpha=1)

    A = np.interp(A, (0,1), (0,255)).astype(np.uint8)

    A = make_boundary_image(L, A)
    if engine == 'plotly':
        vis(A, title='Labeled cells', mode='rgb')
    elif engine == 'PIL':
        A = PIL.Image.fromarray(A)
        A.show()
    elif engine== 'plotly_export':
        return A

def paramcheck():
    while True:
        paramcheck = input('Do you want to use default parameters? (y/n): ')
        # default parameters
        p = {
                'smooth_size': 1,
                'intensity_curve': 1,
                'short_th_radius': 10,
                'long_th_radius': 120,
                'max_size_of_small_objects_to_remove': 50,
                'peak_min_distance': 10,
                'size_after_watershed_to_remove': 50,
                'zoomLev': 1,
            }
        if paramcheck == 'y':
            print('Using default parameters: ', p)
            time.sleep(1)
            break
        elif paramcheck == 'n':
            try:
                smooth_size = int(input('Enter the size of the smoothing_size [int>0, default = {}]: '.format(p['smooth_size'])))
                intensity_curve = int(input('Enter the intensity curve [int>0, default = {}]: '.format(p['intensity_curve'])))
                short_th_radius = int(input('Enter the short threshold radius [int>0, default = {}]: '.format(p['short_th_radius'])))
                long_th_radius = int(input('Enter the long threshold radius [int>0, default = {}]: '.format(p['long_th_radius'])))
                max_size_of_small_objects_to_remove = int(input('Enter the maximum size of small objects to remove [int>0, default = {}]: '.format(p['max_size_of_small_objects_to_remove'])))
                peak_min_distance = int(input('Enter the minimum distance between peaks [int>0, default = {}]: '.format(p['peak_min_distance'])))
                size_after_watershed_to_remove = int(input('Enter the size after watershed to remove [int>0, default = {}]: '.format(p['size_after_watershed_to_remove'])))
                zoomLev = int(input('Enter the zoom level [int>0, default = {}]: '.format(p['zoomLev'])))

                p = {
                    'smooth_size': smooth_size,
                    'intensity_curve': intensity_curve,
                    'short_th_radius': short_th_radius,
                    'long_th_radius': long_th_radius,
                    'max_size_of_small_objects_to_remove': max_size_of_small_objects_to_remove,
                    'peak_min_distance': peak_min_distance,
                    'size_after_watershed_to_remove': size_after_watershed_to_remove,
                    'zoomLev': zoomLev,
                }
                print('Using parameters: ', p)
                time.sleep(1)
                break
            except:
                print('The value is not valid. Try again.\n')
                time.sleep(1)
                continue
        else:
            print('Answer with y or n. Try again.\n')
            time.sleep(1)
            continue
    return p

def cell_counting(path, p):
    # Parameters
    smooth_size = p['smooth_size']
    intensity_curve=p['intensity_curve']
    short_th_radius=p['short_th_radius']
    long_th_radius=p['long_th_radius']
    max_size_of_small_objects_to_remove=p['max_size_of_small_objects_to_remove']
    peak_min_distance=p['peak_min_distance']
    size_after_watershed_to_remove=p['size_after_watershed_to_remove']
    zoomLev = p['zoomLev']

    # Read image
    Ni = io.imread(path)

    Ni = np.sum(np.array(Ni), axis=2)
    Ni = ( (( Ni-np.amin(Ni)) / np.ptp(Ni)) )

    # Step 1: smoothing intensity values and smoothing out peaks
    Ni = fftgauss(Ni, smooth_size, pad_type='edge')

    # Step 2: contrast enhancement by scaling intensities (from 0-1) on a curve
    ########  many other methods can be implemented for this step which could benefit the segmentation
    Ni = np.power(Ni/np.amax(Ni), intensity_curve)

    # Step 3: short range local avg threshold
    th_short = Ni > local_avg(Ni, short_th_radius)

    # Step 4: long range local avg threshold
    th_long = Ni > local_avg(Ni, long_th_radius)

    # Step 5: long && short
    th_Ni = (th_short*th_long)

    # Step 8: remove small objects
    th_Ni = remove_small_objects(th_Ni, 20)
    th_Ni = remove_small_objects(th_Ni, max_size_of_small_objects_to_remove * (zoomLev))

    # Step 9: distance transform
    distance = ndi.distance_transform_edt(th_Ni)

    # Step 10: mark the maxima in the distance transform and assign labels
    peak_markers = corner_peaks(distance, min_distance=peak_min_distance, indices=False)
    peak_markers = ndi.label(peak_markers)[0]

    # Step 11: separate touching nuclei using the watershed markers
    label_Ni = watershed(th_Ni, peak_markers, mask=th_Ni)

    # Step 12: removing small regions after the watershed segmenation
    label_Ni = remove_small_objects(label_Ni, size_after_watershed_to_remove * (zoomLev))

    # Step 13: reassigning labels, so that they are continuously numbered
    old_labels = np.unique(label_Ni)
    for i in range(len(old_labels)):
        label_Ni[label_Ni == old_labels[i]] = i

    labelvis(Ni, label_Ni, engine='PIL')

def each():
    while True:
        path = input('Enter the path to the image: ')
        try:
            image = io.imread(path)
            print('The path is valid.\n')
            time.sleep(1)
            break
        except:
            print('The path is not valid. Try again.\n')

    p = paramcheck()
    
    cell_counting(path, p)
    
    while True:
        go = input('Do you want to continue? (y/n): ')
        if go == 'y':
            each()
        elif go == 'n':
            print('Goodbye!')
            time.sleep(1)
            break
        else:
            print('Answer with y or n. Try again.\n')
            time.sleep(1)
            continue
    
    sys.exit()

def all():
    while True:
        path = input('Enter the path of folder: ')
        try:
            paths = os.listdir(path)
            paths = [x for x in paths if x.endswith('.png')]
            paths.sort()
            print('The path is valid.\n')
            time.sleep(1)
            break
        except:
            print('The csv is not valid. Try again.\n')
            time.sleep(1)
            continue
    
    p = paramcheck()

    try:
        for img in paths:
            input_p = path + '/' + img
            cell_counting(input_p, p)
            print('Counted: ', img)
            time.sleep(1)
            input('Press Enter to continue...')
        print('Done!')
        time.sleep(1)
    except Exception as e:
        print('Something Error wrong. Try again.\n')
        print(e)
        time.sleep(1)
    sys.exit()

def main():
    while True:
        mode = input('Do you want to count one image or all images in a folder? (one[o] / all[a]): ')
        if mode == 'o':
            each()
        elif mode == 'a':
            all()
        else:
            print('Answer with one or all. Try again.\n')
            time.sleep(1)
            continue

if __name__ == '__main__':
    main()


