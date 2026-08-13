import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def minmax(x):
    if x.max() - x.min() == 0:
        return x
    else:
        return x / x.max()
 
def zscore(x):
    if x.max() - x.min() == 0:
        return x
    else:
        return (x - x.mean()) / (x.std() + 1e-5)
    
def spatial_padd(img, roi):
    pad_width = []

    for dim_size, roi_size in zip(img.shape, roi):
        if dim_size < roi_size:
            total_pad = roi_size - dim_size
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))

    padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    return padded_img

def load_nib(path):
    _nib = nib.load(path)
    return _nib.get_fdata(), _nib.affine, _nib.header

def save_to_nib(arr, affine, header, save_path):
    _nib = nib.Nifti1Image(arr, affine=affine, header=header)
    nib.save(_nib, save_path)


def extract_roi(inputs, min_coords, max_coords, margin_percent_decimal):
    margin = ((max_coords - min_coords) * margin_percent_decimal).astype(int)
    min_coords = np.maximum(min_coords - margin, 0)
    max_coords = np.minimum(max_coords + margin, inputs[0].shape)
    processed = []
    for _input in inputs:
        tmp = _input[tuple(slice(min_, max_) for min_, max_ in zip(min_coords, max_coords))]
        processed.append(tmp)

    return processed

def adjust_roi_size(rois, target_shape, methods):
    if isinstance(target_shape, int):
        target_shape = [target_shape, target_shape, target_shape]

    processed_rois = []
    for ii, (roi, method) in enumerate(zip(rois, methods)):
        if method == "resize" or not np.where((np.array(target_shape) - roi.shape) > 0, 1, 0).all():
            zoom_factors = [t / c for t, c in zip(target_shape, roi.shape)]
            processed_rois.append(zoom(roi, zoom_factors, order=1))
        
        elif method == "pad":

            padded = np.zeros(target_shape, dtype=roi.dtype)
            padded[
                (target_shape[0] - roi.shape[0]) // 2 : ((target_shape[0] + roi.shape[0]) // 2),
                (target_shape[1] - roi.shape[1]) // 2 : ((target_shape[1] + roi.shape[1]) // 2),
                (target_shape[2] - roi.shape[2]) // 2 : ((target_shape[2] + roi.shape[2]) // 2),
            ] = roi
            processed_rois.append(padded)
        
        else:
            raise ValueError('Invalid resizing method.')

    return processed_rois
