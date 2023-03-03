# Workflows Parameters

## smooth_size
`(int, pixels)` – The sigma of the Gaussian blur. **Default = 3**, **Modify = 3**
```
from imagemks_custom.filters.fftgaussian import fftgauss
```

## intensity_curve
`(int)` – Exponent of the curve used to fit intensities on range [0,1] for contrast enhancement. **Default** = 2, **Modify** = 2
```
np.power(Ni/np.amax(Ni), intensity_curve)
```

## short_th_radius
`(int, pixels)` – Radius of neighborhood used to calculate a local average threshold. **Default = 50**, **Modify = 20**
```
from imagemks_custom.filters.fftedges import local_avg

th_short = Ni > local_avg(Ni, short_th_radius)
```

## long_th_radius
`(int, pixels)` – Radius of neighborhood used to calculate a local average threshold. **Default = 600**, **Modify = 120**
```
from imagemks_custom.filters.fftedges import local_avg

th_long = Ni > local_avg(Ni, long_th_radius)
```

## max_size_of_small_objects_to_remove
`(float, micrometers^2)` Size beneath which no cells can exist. **Default = 300**, **Modify = 30**
```
from skimage.morphology import remove_small_objects

th_Ni_cvt = remove_small_objects(th_Ni_cvt, max_size_of_small_objects_to_remove * (zoomLev))
```

## peak_min_distance
`(int, pixels)` – Min distance between nuclei for mark the maxima in the distance transform and assign labels. **Default = 10**, **Modify = 5**
```
from skimage.feature import corner_peaks

peak_markers = corner_peaks(distance, min_distance=peak_min_distance, indices=False)
```

## size_after_watershed_to_remove
`(float, micrometers^2)` – Size beneath which no cells can exist. Calculated after watershed for removing small regions after the watershed segmenation **Default = 300**, **Modify = 50**
```
from skimage.morphology import remove_small_objects

label_Ni = remove_small_objects(label_Ni, size_after_watershed_to_remove * (zoomLev))
```

## zoomLev
`(int)` – Real magnification of the image. **Default = 2**, **Modify = 1**

## Summary
| Parameter   |   Default     |   Modify   |
|    --      |       --      |   -- |
|smooth_size |       3       |   3  |
|intensity_curve | 2 |   2|
|short_th_radius|   50  |   20|
|long_th_radius|    600 |   120|
|max_size_of_small_objects_to_remove| 300 | 30|
|peak_min_distance| 10 | 5 |
|size_after_watershed_to_remove| 300 | 50 |
|zoomLev| 2 |   1|