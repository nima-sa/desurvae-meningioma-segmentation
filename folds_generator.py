from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from pathlib import Path
import json
seed = 42
np.random.seed(seed)

akl_mri = Path('...')
akl_segmentation = Path('...')
b23_path = Path('...')
b24_path = Path('...')



orgdf = pd.read_csv('tumors.csv')

orgdf = pd.read_csv('tumors.csv')
df = orgdf.groupby(['PID', 'Family']).agg(
        Mean_Vol=("Volume", "mean"),
    ).reset_index()

for family in df['Family'].unique():
    df.loc[df['Family'] == family, 'Vol_q'] = pd.qcut(df[df['Family'] == family]['Mean_Vol'], q=5, labels=[0, 1, 2, 3, 4])

folds = 5

inclusions = ['akl', 'brats23', 'brats24']
# inclusions = ['brats24']

keys = ['train', 'test']
families = df['Family'].unique()
folds_container = [{k: [] for k in keys} for _ in range(folds)]

for family in families:
    if family not in inclusions:
        continue
    subdf = df[df['Family'] == family]
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pids = subdf['PID'].values
    volume_q = subdf['Vol_q'].values
    for fold, splits in enumerate(kf.split(pids, volume_q)):
        for _key, _split in zip(keys, splits):
            split_pids = pids[_split]
            for _pid in split_pids:
                
                if family == 'akl':
                    file_ids = orgdf[orgdf['PID'] == _pid]['Filename'].unique().tolist()
                    for f_id in file_ids:
                        folds_container[fold][_key].extend([{
                            't1c': str(akl_mri / (f_id + '.nii.gz')),
                            'seg': str(akl_segmentation / (f_id + '.nii.gz')),
                        }])
                    
                elif family == 'brats23':
                    file_ids = orgdf[orgdf['PID'] == _pid]['Filename'].unique().tolist()
                    for f_id in file_ids:
                        f_id = f_id.split('-seg')[0]
                        folds_container[fold][_key].extend([{
                                't1c': str(b23_path / f_id / (f_id + '-t1c.nii.gz')),
                                'seg': str(b23_path / f_id / (f_id + '-seg.nii.gz')),
                            }])
                    
                elif family == 'brats24':
                    file_ids = orgdf[orgdf['PID'] == _pid]['Filename'].unique().tolist()
                    for f_id in file_ids:
                        f_id = f_id.split('_')[0]
                        folds_container[fold][_key].extend([{
                                't1c': str(b24_path / f_id / (f_id + '_t1c.nii.gz')),
                                'seg': str(b24_path / f_id / (f_id + '_gtv.nii.gz')),
                            }])

with open('dataset-5-fold.json', 'w') as j:
    json.dump(folds_container, j, indent=2)

for _key in keys:
    for paths in folds_container[0][_key]:
        for values in paths.values():
            if not Path(values).exists():
                print(f"File does not exist: {values}")

