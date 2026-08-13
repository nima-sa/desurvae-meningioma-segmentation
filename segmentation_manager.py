import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform, binary_fill_holes
from monai.inferers import sliding_window_inference
from tqdm import tqdm

import constants
import segmentation_model_files.desurvae as desurvae
import shared_functions

from pathlib import Path

class SegmentationHandler:
    def __init__(self, model_path, roi, segmentation_binarizing_threshold, keys_for_model, output_dir, supp_model_keys, resume):
        self.model_path = model_path
        self.roi = roi
        self.segmentation_binarizing_threshold = segmentation_binarizing_threshold / 100.0 if segmentation_binarizing_threshold >= 1 else segmentation_binarizing_threshold
        self.keys_for_model = keys_for_model if isinstance(keys_for_model, list) else [keys_for_model]
        self.output_dir = output_dir
        self.supp_model_keys = supp_model_keys
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resume = resume

    def build_model(self):
        model = desurvae.DeSURVAE(
            img_size=list(self.roi),
            in_channels=len(self.keys_for_model),
            out_channels=1, 
            **self.supp_model_keys
        ).to(self.device)
        
        return model

    def load_instance(self, path_config):
        k = []
        affine, header, shape = 0, 0, 0
        for key, path in path_config.items():
            if key not in self.keys_for_model:
                continue
            
            img, affine, header = shared_functions.load_nib(path)
            shape = img.shape
            img = shared_functions.spatial_padd(img, self.roi)
            img = shared_functions.minmax(img)
            k.append(img)

        assert len(k) > 0, f'Inputs not found for {path_config[constants.A_RAW_ID_KEY]}'

        k = torch.from_numpy(np.array(k)[None, ...]).float()

        return k, affine, header, shape
       
    def infer_instance(self, predictor, path_config):
        input_vols, affine, header, org_shape = self.load_instance(path_config)
        input_vols = input_vols.to(self.device)

        preds = sliding_window_inference(
            inputs=input_vols,
            roi_size=self.roi,
            sw_batch_size=1,
            predictor=predictor,
            overlap=0.25,
            device=self.device,
        )

        preds = preds.cpu().numpy().squeeze()
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        preds_sig = sigmoid(preds)
        preds_sig_shape = preds_sig.shape

        threshed_pred_org = np.where(preds_sig >= self.segmentation_binarizing_threshold, 1, 0).astype('uint8')

        threshed_pred_org = threshed_pred_org[
            preds_sig_shape[0] // 2 - org_shape[0] // 2:preds_sig_shape[0] // 2 - org_shape[0] // 2 + org_shape[0],
            preds_sig_shape[1] // 2 - org_shape[1] // 2:preds_sig_shape[1] // 2 - org_shape[1] // 2 + org_shape[1],
            preds_sig_shape[2] // 2 - org_shape[2] // 2:preds_sig_shape[2] // 2 - org_shape[2] // 2 + org_shape[2],
        ]
        threshed_pred_org[threshed_pred_org > 0] = 1
        return threshed_pred_org, affine, header

    def revert_affine(self, reference_mri_path, segmentation_output, segmentation_affine, affine_mat_path):
        if not isinstance(reference_mri_path, Path):
            reference_mri_path = Path(reference_mri_path)

        reference_mri = nib.load(reference_mri_path)

        with open(affine_mat_path, 'r') as f:
            T1_to_SRI_transform = np.loadtxt(f).reshape(4, 4)

        M = np.linalg.inv(reference_mri.affine) @ T1_to_SRI_transform @ segmentation_affine
        M = np.linalg.inv(M)
        matrix = M[:3, :3]
        offset = M[:3, 3]
        seg_resampled = affine_transform(
            segmentation_output,
            matrix=matrix,
            offset=offset,
            output_shape=reference_mri.shape,
            order=0,
            mode='constant',
            cval=0
        )
        seg_resampled = binary_fill_holes(seg_resampled > 0).astype(np.uint8)
        return seg_resampled, reference_mri.affine, reference_mri.header


    def infer_all(self, mri_path_configs):
        model = self.build_model()
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))

        def predictor(*args, **kwargs):
            pred = model(*args, **kwargs)
            return pred[0] if isinstance(pred, tuple) else pred
        
        new_path_configs = []
        
        with torch.no_grad():
            model.eval()
            with tqdm(total=len(mri_path_configs)) as pbar:
                for mri_path_config in mri_path_configs.path_configs:
                    _id = mri_path_config[constants.A_RAW_ID_KEY]
                    pbar.set_description(f'Segmenting {_id}')
                    # save_path = self.output_dir / _id
                    save_path = self.output_dir
                    save_path.mkdir(parents=True, exist_ok=True)
                    save_path_sri = save_path / f'{_id}_{constants.A_SEGMENTATION_SUFFIX}.nii.gz'
                    save_path_rvt = save_path / f'{_id}{constants.A_REVERTED_TO_ORG_FROM_SRI_SEGMENTATION_SUFFIX}.nii.gz'
                    if self.resume and save_path_sri.exists():
                        pass

                    else:
                        pred, affine, header = self.infer_instance(predictor, mri_path_config)
                        # shared_functions.save_to_nib(pred, affine, header, save_path_sri)
                        
                        reverted_pred, reverted_affine, reverted_header = self.revert_affine(
                            mri_path_config[constants.A_RAW_MRI_PATH_KEY], 
                            pred, affine, 
                            mri_path_config[constants.A_AFFINE_MAT_PATH_KEY]
                        )
                        shared_functions.save_to_nib(reverted_pred, reverted_affine, reverted_header, save_path_rvt)

                    mri_path_config[constants.A_AUTO_SEGMENTATION_PATH_KEY] = save_path_sri
                    mri_path_config[constants.A_REVERTED_TO_ORG_FROM_SRI_SEGMENTATION_PATH_KEY] = save_path_rvt
                    new_path_configs.append(mri_path_config)
                    pbar.update(1)

                pbar.set_description('Segmentation finished')
                
        mri_path_configs.replace(new_path_configs)

        del model
        torch.cuda.empty_cache()
        return mri_path_configs


    
if __name__ == '__main__':
    sh = SegmentationHandler(
        model_path=None,
        roi=(128, 128, 128), 
        segmentation_binarizing_threshold=65, 
        keys_for_model=constants.A_PROCESSED_MRI_PATH_KEY, 
        output_dir='.', 
        supp_model_keys={
            'feature_size':48,
            'use_residual':True,
            'include_vae':True,
            'include_unet':True,
            'add_mirror_convs':True,
        },
        resume=True
    )
    
    sh.build_model()