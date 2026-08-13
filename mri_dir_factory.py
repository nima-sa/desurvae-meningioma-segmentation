from pathlib import Path
import constants

class MRIConfigFactory:
    def __init__(self, home_path, mri_signature, duplicate_mri_path_for_keys=None):
        self.home_path = Path(home_path)
        self.mri_signature = mri_signature
        self.duplicate_mri_path_for_keys = duplicate_mri_path_for_keys or []
        self.path_configs = []
        
    def traverse_when_files_are_in_separate_folders(self):
        folders = sorted([d for d in self.home_path.iterdir() if d.is_dir()])
        
        path_configs = []
        for d in folders:
            mri_path = list(d.glob(self.mri_signature))[0]
            _dict = {
                constants.A_RAW_MRI_PATH_KEY: mri_path,
                constants.A_RAW_ID_KEY: mri_path.parts[-2]
            }
            for key in self.duplicate_mri_path_for_keys:
                _dict[key] = mri_path
                
            path_configs.append(_dict)
        
        self.replace(path_configs)
        return path_configs
    
    def traverse_when_files_are_in_the_same_folder(self):
        mris = sorted(list(self.home_path.glob(self.mri_signature)))
        
        path_configs = []
        for d in mris:
            mri_path = d
            _dict = {
                constants.A_RAW_MRI_PATH_KEY: mri_path,
                constants.A_RAW_ID_KEY: mri_path.stem.split('.nii')[0]
            }
            for key in self.duplicate_mri_path_for_keys:
                _dict[key] = mri_path
                
            path_configs.append(_dict)
        
        self.replace(path_configs)
        return path_configs

    def dummy_traverse_and_add_segmentations(self, segmentation_path, total=None):
        segmentation_path = Path(segmentation_path) if not isinstance(segmentation_path, Path) else segmentation_path
        mris = sorted(list(self.home_path.glob(self.mri_signature)))
        
        path_configs = []
        for idx, d in enumerate(mris):
            if total is not None and idx > total:
                break
            mri_path = d
            _dict = {
                constants.A_RAW_MRI_PATH_KEY: mri_path,
                constants.A_RAW_ID_KEY: mri_path.stem.split('.nii')[0]
            }
            for key in self.duplicate_mri_path_for_keys:
                _dict[key] = mri_path
                
            _dict[constants.A_MANUAL_SEGMENTATION_PATH_KEY] = segmentation_path / (_dict[constants.A_RAW_ID_KEY] + '.nii.gz')  
            _dict[constants.A_AUTO_SEGMENTATION_PATH_KEY] = segmentation_path / (_dict[constants.A_RAW_ID_KEY] + '.nii.gz')  
            
            path_configs.append(_dict)
        
        self.replace(path_configs)
        return path_configs
        
    def _check(self):
        if len(self.path_configs) == 0:
            raise RuntimeError('Nothing found in the specified path')

    def replace(self, path_configs):
        self.path_configs = path_configs
        self._check()

    def __len__(self):
        return len(self.path_configs)

    def __iter__(self):
        for k in self.path_configs:
            yield k