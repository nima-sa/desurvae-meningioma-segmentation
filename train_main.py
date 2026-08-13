import os

key_for_visualization = 't1c'
keys_for_model = ['t1c']
roi = (128, 128, 128)

include_vae = True
include_unet = True
include_residual = True

folds_json = 'desurvae-training-folds.json'

data_to_load = list(set(keys_for_model + [key_for_visualization, 'seg']))
init_filters = 48
batch_size = 1
max_epochs = 500
fold = 0
uid = f'desurvae-f{fold}'

continue_training = False

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ConvertToMultiChannelBasedOnBratsClassesd,
    # StripNRRDHeader,
    SpatialPadd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAffined,
    CropForegroundd,
    Orientationd,
    Spacingd,
    SqueezeDimd,
)

from monai.transforms import MapTransform
from monai.utils import ensure_tuple
from monai.data import CacheDataset, DataLoader
from monai.losses import DiceLoss, TverskyLoss
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice as compute_meandice
from monai.metrics import DiceMetric
from monai.data import decollate_batch

import torch
from segmentation_model_files.desurvae import DeSURVAE

import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CSVLoggerCallback:
    def __init__(self, file_path, overwrite=False):
        self.file_path = Path(file_path)
        self.overwrite = overwrite
        self.keys = None

        # Prepare the file
        if overwrite and self.file_path.exists():
            self.file_path.unlink()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs):
        """Log the metrics and loss to the CSV file."""
        logs = logs or {}
        
        # Flatten nested dictionaries, e.g., {'train': {'loss': tensor}, 'val': {'loss': tensor}}
        flat_logs = {"epoch": epoch + 1}  # Epochs are 0-indexed
        for outer_key, inner_dict in logs.items():
            if isinstance(inner_dict, dict):  # Handle nested dictionary
                for key, value in inner_dict.items():
                    # Convert tensors to floats; handle non-tensor values gracefully
                    if isinstance(value, torch.Tensor):
                        flat_logs[f"{outer_key}_{key}"] = value.item()
                    else:
                        flat_logs[f"{outer_key}_{key}"] = value
            else:  # Handle non-dict values
                if isinstance(inner_dict, torch.Tensor):
                    flat_logs[outer_key] = inner_dict.item()
                else:
                    flat_logs[outer_key] = inner_dict

        # Initialize keys if this is the first epoch
        if self.keys is None:
            self.keys = list(flat_logs.keys())

        # Write header if file does not exist or overwrite is True
        write_header = not self.file_path.exists() or (self.overwrite and epoch == 0)

        with self.file_path.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.keys)
            if write_header:
                writer.writeheader()
            writer.writerow(flat_logs)

class OneHotToLabeld(MapTransform):
    def __init__(self, keys, channel_dim=0):
        super().__init__(keys)
        self.channel_dim = channel_dim

    def __call__(self, data):
        data = self.transform(data)
        return data

    def transform(self, data):
        for key in self.keys:
            seg = data[key]
            if seg.ndimension() > 1:
                seg = seg.to(torch.float32)
                seg = seg.argmax(dim=self.channel_dim)
            
            data[key] = seg
        return data
    
class ExpandDimsd(MapTransform):
    def __init__(self, keys, target_dims, channel_dim=0):
        super().__init__(keys)
        self.target_dims = target_dims
        self.channel_dim = channel_dim

    def __call__(self, data):
        data = self.transform(data)
        return data

    def transform(self, data):
        for key in self.keys:
            img = data[key]
            current_dims = img.ndimension()
            if current_dims < self.target_dims:
                missing_dims = self.target_dims - current_dims
                for _ in range(missing_dims):
                    img = img.unsqueeze(0)

            data[key] = img
        return data
    
class Normalizer(MapTransform):
    def __init__(self, keys, func='minmax'):
        super().__init__(keys)
        self.func = func

    def __call__(self, data):
        data = self.transform(data)
        return data

    def transform(self, data):
        for key in self.keys:
            img = data[key]
            if self.func == 'zscore':
                img = self.zscore(img)
            else:
                img = self.minmax(img)
                
            data[key] = img
        return data
    
    def minmax(self, img):
        if img.max() != img.min():
            img = (img - img.min()) / (img.max() - img.min())
        return img.astype(np.float32)
    
    def zscore(self, img):
        if img.std() != 0:
            img = (img - img.mean()) / img.std()
        return img.astype(np.float32)

class Binarize(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        data = self.transform(data)
        return data

    def transform(self, data):
        for key in self.keys:
            img = data[key]
            img = np.where((img == 1) | (img == 3), 1, 0).astype('float32')
            data[key] = img
        return data

class ConcatTensorsd(MapTransform):
    def __init__(self, keys, output_key, dim=0, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        self.dim = dim

    def __call__(self, data):
        tensors = [torch.as_tensor(data[k]) for k in self.keys if k in data]
        if len(tensors) > 1:
            data[self.output_key] = torch.cat(tensors, dim=self.dim)
        elif tensors:
            data[self.output_key] = tensors[0]  # No concatenation needed if there's only one tensor
        return data

class ToTensor(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        """
        Args:
            keys (list): List of keys whose tensor values will be concatenated.
            output_key (str): The key where the concatenated tensor will be stored.
            dim (int): The dimension along which to concatenate tensors.
            allow_missing_keys (bool): Whether to ignore missing keys.
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            if not isinstance(data[key], torch.Tensor):
                data[key] = torch.as_tensor(data[key])
        return data
    
train_transform = Compose(
    [
        LoadImaged(keys=data_to_load),
        EnsureChannelFirstd(keys=data_to_load),
        EnsureTyped(keys=data_to_load),
        ExpandDimsd(keys="seg", target_dims=4),
        SpatialPadd(keys=data_to_load,
                    spatial_size=[roi[0], roi[1], roi[2]],
                method='symmetric',
                mode='constant'),

        Normalizer(keys=[key_for_visualization]),
        RandSpatialCropd(
            keys=data_to_load,
            roi_size=[roi[0], roi[1], roi[2]],
            random_size=False,
        ),
        RandFlipd(keys=data_to_load, prob=0.5, spatial_axis=0),
        RandFlipd(keys=data_to_load, prob=0.5, spatial_axis=1),
        RandFlipd(keys=data_to_load, prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=keys_for_model, factors=0.1, prob=1.0),
        RandShiftIntensityd(keys=keys_for_model, offsets=0.1, prob=1.0),
        RandAffined(keys=[*keys_for_model, "seg"], prob=0.1),
        Binarize(keys=['seg']),
        ConcatTensorsd(keys=keys_for_model, output_key='image'),
        ToTensor(keys=[*keys_for_model, 'seg']),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=data_to_load),
        EnsureChannelFirstd(keys=data_to_load),
        EnsureTyped(keys=data_to_load),
        ExpandDimsd(keys="seg", target_dims=4),
        SpatialPadd(keys=data_to_load,
                    spatial_size=[roi[0], roi[1], roi[2]],
                    method='symmetric',
                    mode='constant'),
        Normalizer(keys=[key_for_visualization]),
        RandSpatialCropd(
            keys=data_to_load,
            roi_size=[roi[0], roi[1], roi[2]],
            random_size=False,
        ),
        Binarize(keys=['seg']),
        ConcatTensorsd(keys=keys_for_model, output_key='image'),
        ToTensor(keys=data_to_load),
    ]
)

test_transform = Compose(
    [
        LoadImaged(keys=data_to_load),
        EnsureChannelFirstd(keys=data_to_load),
        EnsureTyped(keys=data_to_load),
        ExpandDimsd(keys="seg", target_dims=4),
        SpatialPadd(keys=data_to_load,
                    spatial_size=[roi[0], roi[1], roi[2]],
                    method='symmetric',
                    mode='constant'),
        Normalizer(keys=[key_for_visualization]),
        Binarize(keys=['seg']),
        ConcatTensorsd(keys=keys_for_model, output_key='image'),
        ToTensor(keys=data_to_load),
    ]
)

def compute_metrics(pred, target):
    dice = compute_meandice(pred.unsqueeze(0), target.unsqueeze(0), include_background=False, ignore_empty=False).item()
    return dice

def train_epoch(model, loader, loss_function, optimizer, device, epoch):
    model.train()
    metrics = {'loss': 0, 'dice': 0}

    for batch_data in tqdm(loader, desc=f"Epoch {epoch + 1} training"):
        inputs, labels = batch_data['image'].to(device), batch_data['seg'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs, vae_loss = outputs
        else:
            vae_loss = None
            
        loss = loss_function(outputs, labels) + (vae_loss if vae_loss is not None else 0)
        loss.backward()
        optimizer.step()

        dice = compute_metrics(outputs, labels)
        metrics['dice'] += dice
        metrics['loss'] += loss.item()
        
    # Average loss and metrics
    metrics = {k: v / len(loader) for k, v in metrics.items()}
    return metrics


def validate_epoch(model, loader, loss_function, device, epoch):
    
    """Validate the model for one epoch."""
    model.eval()
    metrics = {'loss': 0, 'dice': 0}

    with torch.no_grad():
        for batch_data in tqdm(loader, desc=f"Epoch {epoch + 1} validation"):
            inputs, labels = batch_data['image'].to(device), batch_data['seg'].to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = loss_function(outputs, labels).item()

            # Update metrics
            dice = compute_metrics(outputs, labels)
            metrics['dice'] += dice
            metrics['loss'] += loss

    # Average loss and metrics
    metrics = {k: v / max(1, len(loader)) for k, v in metrics.items()}
    return metrics


class PathCacheDataset(CacheDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        data["file_path"] = self.data[index]["seg"]
        return data
    
if __name__ == '__main__':
    with open(folds_json, 'r') as j:
        dataset_paths_all_folds = json.load(j)
        dataset_paths = dataset_paths_all_folds[fold]

    model = DeSURVAE(
        img_size=list(roi),
        in_channels=len(keys_for_model),
        out_channels=1,
        feature_size=init_filters,
        use_residual=include_residual,
        include_vae=include_vae,
        include_unet=include_unet,
        add_mirror_convs=True
    ).to(device)

    Path(f'{uid}').mkdir(parents=True, exist_ok=True)

    epoch = 0
    train_dataset = CacheDataset(data=dataset_paths['train'], transform=train_transform, cache_rate=0.1, num_workers=4)
    val_dataset = PathCacheDataset(data=dataset_paths['val'], transform=val_transform, cache_rate=0.1, num_workers=4)
    test_dataset = PathCacheDataset(data=dataset_paths['test'], transform=test_transform, cache_rate=0.1, num_workers=4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    csv_logger = CSVLoggerCallback(f'{uid}/training_log.csv', overwrite=not continue_training)


    if continue_training != False:
        if isinstance(continue_training, str):
            if os.path.exists(continue_training):
                model.load_state_dict(torch.load(continue_training, map_location=device, weights_only=True))
            else:
                model.load_state_dict(torch.load(f'{uid}/{continue_training}', map_location=device, weights_only=True))
        else:
            model.load_state_dict(torch.load(f"{uid}/epoch_model.pth", map_location=device, weights_only=True))

        continue_training = True
        epoch = len(pd.read_csv(f'{uid}/training_log.csv'))

    loss_function = DiceLoss(smooth_nr=0.001, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    
    for _ in range(max_epochs - epoch):
        try:
            train_metrics = train_epoch(model, train_loader, loss_function, optimizer, device, epoch)
            val_metrics = validate_epoch(model, val_loader, loss_function, device, epoch)
            csv_logger.on_epoch_end(epoch, {'train': train_metrics, 'val': val_metrics})

            lr_scheduler.step()
            torch.save(model.state_dict(), f"{uid}/epoch_model.pth")
            
        except KeyboardInterrupt:
            break
        
        if (epoch + 1) % 10 == 0:
            try:
                torch.save(model.state_dict(), f"{uid}/inf_{epoch+1}.pth")
            except KeyboardInterrupt, StopIteration:
                pass
        epoch += 1
    torch.save(model.state_dict(), f"{uid}/fin_model.pth")
