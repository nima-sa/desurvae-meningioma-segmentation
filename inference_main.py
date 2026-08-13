import os


from pathlib import Path

from brain_atlas_registration_and_skull_stripping import BrainAtlasRegistrationAndSkullStripping
from mri_dir_factory import MRIConfigFactory
from segmentation_manager import SegmentationHandler
import import_once
import constants
import argparse

parser = argparse.ArgumentParser(description="Run inference on NIfTI files.")

parser.add_argument('-i', '--input_dir', type=str, default='input', help='Path to the input directory containing .nii.gz files.')
parser.add_argument('-o', '--output_dir', type=str, default='output', help='Path to the output directory to save results.')

args = parser.parse_args()
MRI_DIR = args.input_dir
OUTPUT_DIR = args.output_dir


CAPTK_EXECUTABLE_PATH = './CaPTk_1.9.0/CaPTk/1.9.0/captk'
DEEPMEDIC_MODEL_DIR = './CaPTk_1.9.0/saved_models/skullStripping_modalityAgnostic'
WORKING_DIR = Path('/tmpn')


PROCESSINGS_DIR = WORKING_DIR / f'processings'
SEGMENTATION_HOME = Path('./segmentation_model_files')

def cleanup_extra_items(mricf):
    keys_to_remove = {
        constants.A_PROCESSED_MRI_PATH_KEY, 
        constants.A_AUTO_SEGMENTATION_PATH_KEY,
        constants.A_AFFINE_MAT_PATH_KEY,
    }
    for path_config in mricf:
        for key in keys_to_remove:
            if key in path_config:
                try:
                    path_config[key].unlink()
                except Exception as e:
                    pass

def main():
    mricf = MRIConfigFactory(
        home_path=MRI_DIR, 
        mri_signature='*.nii*', 
    )

    mricf.traverse_when_files_are_in_the_same_folder()
    
    barss = BrainAtlasRegistrationAndSkullStripping(
        captk_executable_path=CAPTK_EXECUTABLE_PATH,
        deepmedic_model_dir=DEEPMEDIC_MODEL_DIR,
        working_dir=PROCESSINGS_DIR, 
        output_dir=OUTPUT_DIR, 
        brain_mask_probability_threshold=0.1,
        resume=True,
    )
    mricf = barss.process_all(mricf)

    sh = SegmentationHandler(
        model_path=SEGMENTATION_HOME / 'desurvae_final.pth',
        roi=(128, 128, 128), 
        segmentation_binarizing_threshold=65, 
        keys_for_model=constants.A_PROCESSED_MRI_PATH_KEY, 
        output_dir=OUTPUT_DIR, 
        supp_model_keys={
            'feature_size':48,
            'use_residual':True,
            'include_vae':True,
            'include_unet':True,
            'add_mirror_convs':True,
        },
        resume=True
    )

    mricf = sh.infer_all(mricf)
    cleanup_extra_items(mricf)
    

if __name__ == '__main__':
    main()