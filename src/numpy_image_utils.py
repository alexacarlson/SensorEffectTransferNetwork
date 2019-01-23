"""
Utilities for working with images as numpy arrays
"""
import numpy as np


def fit_image(image, w, h, pad_method='symmetric'):
    """

    :param image: 3d numpy array
    :param w: desired width
    :param h: desired height
    :param pad_method: how to pad (e.g 'edge', 'mean', 'symmetric', 'mirror')
    :return: a 3d numpy array padded and cropped according to desired dimensions
    """
    fitted = image

    actual_h, actual_w = image.shape[:2]

    delta_w = w - actual_w
    delta_h = h - actual_h

    if any([el > 0 for el in (delta_w, delta_h)]):
        pad_left, extra = divmod(delta_h, 2)
        pad_right = pad_left + extra
        pad_top, extra = divmod(delta_w, 2)
        pad_bottom = pad_top + extra
        fitted = np.pad(
            fitted,
            ((max(0, pad_left), max(0, pad_right)),
             (max(0, pad_top), max(0, pad_bottom)),
             (0, 0)),
            pad_method)

    if delta_h < 0:
        slice_h = abs(delta_h)
        slice_top, extra = divmod(slice_h, 2)
        slice_bottom = slice_top + extra
        fitted = fitted[slice_top:actual_h - slice_bottom, :, :]
    if delta_w < 0:
        slice_w = abs(delta_w)
        slice_left, extra = divmod(slice_w, 2)
        slice_right = slice_left + extra
        fitted = fitted[:, slice_left:actual_w - slice_right, :]

    return fitted


