import numpy as np
from ._fftconvolve import fftconvolve2d
from ..structures import circle

def local_avg(img, rad, mask=None, pad_type=None, **kwargs):
    '''
    Returns the local average of a neighborhood taking into account that the
    edges need to be treated differently. Averages are only calculated at the
    edges using values inside the image and the neighborhood.

    Parameters
    ----------
    img : (M,N) array
        An image.
    rad : numeric
        The radius of the neighborhood.
    mask : (M,N) binary array, optional
        A binary array that can be used to define what is outside the
        image.
    pad_type : string, optional
        The padding type to be used. For additional information see numpy.pad .
        Defaults to constant.
    kwargs : varies
        See numpy.pad . Defaults to constant_values=0.

    Returns
    -------
    local_averages : (M,N) array
        The local averages at all pixel locations.
    '''
    s = img.shape

    if mask:
        counts = mask.copy()
    else:
        counts = np.ones(s)

    pad_r = rad+1

    K = circle(rad, size=s)

    sums = fftconvolve2d(img, K, r=pad_r, pad_type=pad_type)
    norms = fftconvolve2d(counts, K, r=pad_r, pad_type=pad_type)

    return np.divide(sums, norms)
