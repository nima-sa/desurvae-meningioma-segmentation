import os
import subprocess
from scipy.ndimage import affine_transform, label, binary_fill_holes, binary_dilation
from skimage.measure import regionprops
# import nibabel as nib
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import shared_functions
import constants

class BrainAtlasRegistrationAndSkullStripping:
    SKULL_STRIPPED_MRI_SUFFIX = '_skull_stripped_mri'
    
    def __init__(self, captk_executable_path, deepmedic_model_dir, working_dir, output_dir, brain_mask_probability_threshold, resume):
        self.captk_executable_path = captk_executable_path
        self.deepmedic_model_dir = deepmedic_model_dir
        self.working_dir = Path(working_dir)
        self.output_dir = Path(output_dir)
        self.brain_mask_probability_threshold = brain_mask_probability_threshold
        self.resume = resume
    
    def to_sri(self, mri_path, tmp_dir):
        cmd = [
            self.captk_executable_path, 
            "BraTSPipeline",
            "-t1c", f'{mri_path}',
            "-t1", f'{mri_path}',
            "-t2", f'{mri_path}',
            "-fl", f'{mri_path}',
            "-o", f'{tmp_dir}',
            "-s", "0", 
            "-b", "0"
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        
        with open(tmp_dir / ('brats_pipeline_stderr_log.txt'), 'w') as l:
            l.write(res.stderr)

        with open(tmp_dir / ('brats_pipeline_stdout_log.txt'), 'w') as l:
            l.write(res.stdout)
            
        if 'Finished, please perform manual quality-check of generated brain mask before applying to input images' not in res.stdout:
            raise RuntimeError(res.stdout)
        
        processed_mri_path = tmp_dir / constants.CAPTK_OUTPUT_KEY__T1CE_TO_SRI_NIB
        affine_mat_path = tmp_dir / constants.CAPTK_OUTPUT_KEY__T1CE_TO_SRI_TRANSLATION_MAT
        return processed_mri_path, affine_mat_path

    def strip_skull(self, processed_mri_path, tmp_dir, normalized_before=1):
        cmd = [
            self.captk_executable_path, 
            'DeepMedic',
            '-i', f'{processed_mri_path}',
            '-o', f'{tmp_dir}',
            '-md', self.deepmedic_model_dir,
            '-zn', f'{normalized_before}'
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        with open(tmp_dir/ ('skull_stripping_stderr_log.txt'), 'w') as l:
            l.write(res.stderr)

        with open(tmp_dir/ ('skull_stripping_stdout_log.txt'), 'w') as l:
            l.write(res.stdout)
        
        if 'DeepMedic exited with code !=0.' in res.stderr:
            raise RuntimeError(res.stderr)

        fin_masks_folder = tmp_dir / constants.CAPTK_OUTPUT_KEY__DEEPMEDIC_SKULL_STRIPPING_OUTPUT
        # bg_mask_probability_file_name = constants.CAPTK_OUTPUT_KEY__BACKGROUND_MASK
        
        return fin_masks_folder

    def apply_skull_stripping_to_image(self, processed_mri_path, brain_mask_probabilities, tmp_dir):
        brainmask_prob_map, _, _ = shared_functions.load_nib(brain_mask_probabilities)
        brainmask_prob_map_thresholded = brainmask_prob_map
        brainmask_prob_map_thresholded[brainmask_prob_map_thresholded < self.brain_mask_probability_threshold] = 0
        brainmask_prob_map_thresholded[brainmask_prob_map_thresholded >= self.brain_mask_probability_threshold] = 1

        # find largest CC
        labeled_array, num_features = label(brainmask_prob_map_thresholded)
        regions = regionprops(labeled_array)
        largest_region = max(regions, key=lambda r: r.area)
        largest_component = labeled_array == largest_region.label

        # fill holes
        filled_component = binary_fill_holes(largest_component)

        # dilate
        structure = np.ones((3, 3, 3), dtype=int)
        dilated_component = binary_dilation(filled_component, structure=structure)
        processed_brain_mask = dilated_component.astype(np.uint8)

        mri_fname = processed_mri_path.stem.split('.')[0]

        masked_img_outfile_path = tmp_dir / f'{mri_fname}{self.SKULL_STRIPPED_MRI_SUFFIX}.nii.gz'

        registered_img_raw, registered_img_raw_affine, registered_img_raw_header = shared_functions.load_nib(processed_mri_path)
        registered_img_masked = registered_img_raw * processed_brain_mask
        shared_functions.save_to_nib(registered_img_masked, registered_img_raw_affine, registered_img_raw_header, masked_img_outfile_path)

        return masked_img_outfile_path
    
    def seg_to_sri(self, reference_mri_path, segmentation_path, affine_mat_path, tmp_output_dir):
        seg_data, seg_affine, _ = shared_functions.load_nib(segmentation_path)
        reference_mri, reference_mri_affine, reference_mri_header = shared_functions.load_nib(reference_mri_path)

        with open(affine_mat_path, 'r') as f:
            T1_to_SRI_transform = np.loadtxt(f).reshape(4, 4)

        M = np.linalg.inv(seg_affine) @ T1_to_SRI_transform @ reference_mri_affine
        matrix = M[:3, :3]
        offset = M[:3, 3]

        seg_resampled = affine_transform(
            seg_data,
            matrix=matrix,
            offset=offset,
            output_shape=reference_mri.shape,
            order=1,
            mode='constant',
            cval=0
        )
        seg_resampled_filled = binary_fill_holes(seg_resampled > 0).astype(np.uint8)
        segmentation_output_path = tmp_output_dir / f"{segmentation_path.stem.split('.')[0]}.nii.gz"
        shared_functions.save_to_nib(seg_resampled_filled, reference_mri_affine, reference_mri_header, segmentation_output_path)

        return segmentation_output_path

    def process_instance(self, mri_path, tmp_output, output_dir, segmentation_path=None):
        processed_mri_path, affine_mat_path = self.to_sri(mri_path, tmp_output)
        
        if segmentation_path is not None:
            self.seg_to_sri(mri_path, segmentation_path, affine_mat_path, tmp_output)
            
        brain_mask_probability_file = self.strip_skull(processed_mri_path, tmp_output)
        # brain_mask_probability_file = Path('/Users/nsad315/Datasets/nospacedir/processings/BraTS-MEN-RT-0002-1/predictions/testApiSession/predictions/ProbMapClass1.nii.gz')
        masked_img_outfile_path = self.apply_skull_stripping_to_image(processed_mri_path, brain_mask_probability_file, tmp_output)

        final_mri_name = mri_path.name if mri_path.name.endswith('.nii.gz') else f'{mri_path.stem.split(".nii")[0]}.nii.gz'

        shutil.copyfile(masked_img_outfile_path, output_dir / final_mri_name)
        shutil.copyfile(affine_mat_path, output_dir / affine_mat_path.name)
        return output_dir / final_mri_name, output_dir / affine_mat_path.name

    def sri_to_org(self, reference_path, target_path, affine_mat_path, tmp_output_dir):     
        target_data, target_affine, _ = shared_functions.load_nib(target_path)
        reference_data, reference_affine, reference_header = shared_functions.load_nib(reference_path)

        with open(affine_mat_path, 'r') as f:
            T1_to_SRI_transform = np.loadtxt(f).reshape(4, 4)

        # Map SRI voxel coordinates → T1 voxel coordinates
        M = np.linalg.inv(reference_affine) @ T1_to_SRI_transform @ target_affine
        M = np.linalg.inv(M)
        matrix = M[:3, :3]
        offset = M[:3, 3]

        # Resample segmentation from SRI space back to original MRI space
        seg_resampled = affine_transform(
            target_data,
            matrix=matrix,
            offset=offset,
            output_shape=reference_data.shape,
            order=0,
            mode='constant',
            cval=0
        )

        seg_resampled = seg_resampled / seg_resampled.max() * 255
        # seg_registered = nib.Nifti1Image(seg_resampled.astype(np.uint8), reference_mri_affine)
        output_path = tmp_output_dir / f"{reference_path.stem}_sri_to_sega.nii.gz"
        # nib.save(seg_registered, output_path)
        shared_functions.save_to_nib(seg_resampled.astype(np.uint8), reference_affine, reference_header, output_path)

        return output_path

    def process_all(self, mri_path_configs):
        new_path_configs = []
        with tqdm(total=len(mri_path_configs)) as pbar:
            for path_config in mri_path_configs:
                _id = path_config[constants.A_RAW_ID_KEY]
                mri_path = path_config[constants.A_RAW_MRI_PATH_KEY]
                
                pbar.set_description(f'Preprocessing {_id}')
                
                tmp_output = Path(self.working_dir / _id)
                tmp_output.mkdir(parents=True, exist_ok=True)
                
                final_destination = self.output_dir / _id
                final_destination.mkdir(parents=True, exist_ok=True)

                if self.resume and len(list(final_destination.glob(constants.CAPTK_OUTPUT_KEY__T1CE_TO_SRI_TRANSLATION_MAT))) > 0:
                    processed_mri_path = final_destination / f'{_id}.nii.gz'
                    affine_mat_path = final_destination / constants.CAPTK_OUTPUT_KEY__T1CE_TO_SRI_TRANSLATION_MAT

                else:
                    segmentation_path = path_config.get(constants.A_MANUAL_SEGMENTATION_PATH_KEY, None)
                    processed_mri_path, affine_mat_path = self.process_instance(mri_path, tmp_output, final_destination, segmentation_path)

                path_config[constants.A_PROCESSED_MRI_PATH_KEY] = processed_mri_path
                path_config[constants.A_AFFINE_MAT_PATH_KEY] = affine_mat_path

                shutil.rmtree(tmp_output)
                new_path_configs.append(path_config)
                pbar.update(1)

            mri_path_configs.replace(new_path_configs)
            pbar.set_description('Preprocessing done')
            return mri_path_configs
            
