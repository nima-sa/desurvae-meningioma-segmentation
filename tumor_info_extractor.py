from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm
import pandas as pd


akl_segmentations = Path('...')
b23 = Path('...')
b24 = Path('...')


segs = (
    [('brats23', i) for i in list(b23.glob('*/*seg*.nii.gz'))] +
    [('brats24', i) for i in list(b24.glob('*/*gtv*.nii.gz'))] +
    [('akl', i) for i in list(akl_segmentations.glob('*.nii.gz'))]
)

tumor_sizes = []

for family, seg_path in tqdm(segs):
    seg = nib.load(seg_path)
    data = seg.get_fdata()
    data = np.where((data == 1) | (data == 3), 1, 0)
    if np.count_nonzero(data) == 0:
        continue

    labeled, num_features = label(data > 0)
    zooms = np.prod(seg.header.get_zooms())
    if num_features == 0:
        tumor_sizes.append({
            'PID': seg_path.name.split('.nii')[0].split('_')[0].replace('-seg', ''),
            'Family': family,
            'Filename': seg_path.name.replace('.nii.gz', ''),
            'TumorID': 0,
            'Width': 0,
            'Height': 0,
            'Depth': 0,
            'BBOX': 0,
            'Pixels': 0,
            'Volume': 0,
            })
        continue

    for tumor_id in range(1, num_features + 1):
        overlap = np.where(labeled == tumor_id, 1, 0)
        coords = np.array(np.where(labeled == tumor_id))
        if coords.size == 0:
            continue

        x, y, z = coords
        w = x.max() - x.min()
        h = y.max() - y.min()
        d = z.max() - z.min()
        if w == 0 or h == 0 or d == 0:
            continue
        if family == 'akl':
            pid = seg_path.name.split('.nii')[0].split('_')[0].replace('-seg', '')
        else:
            pid = '-'.join(seg_path.name.split('.nii')[0].split('_')[0].replace('-seg', '').split('-')[:-1])
            
        tumor_sizes.append({
            'PID': pid,
            'Family': family,
            'Filename': seg_path.name.replace('.nii.gz', ''),
            'TumorID': tumor_id,
            'Width': w,
            'Height': h,
            'Depth': d,
            'BBOX': w * h * d,
            'Pixels': overlap.sum(),
            'Volume': overlap.sum() * zooms,
        })

tumor_df = pd.DataFrame(tumor_sizes)
tumor_df.to_csv('tumors.csv', index=False)
