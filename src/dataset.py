"""
Dataset + transforms for FREUID.

Pipeline per sample:
  read RGB (cv2) -> [optional] RecaptureSimulator (numpy/cv2, see augment_recapture.py)
  -> albumentations (Resize + flips + Normalize + ToTensorV2; only version-stable ops).

The recapture simulation is the main domain-gap lever and is applied to BOTH classes so the
model cannot use "looks digital/looks recaptured" as a fraud shortcut. p_recapture controls how
often a *training* image is pushed through the analog hole.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
from albumentations.pytorch import ToTensorV2

from augment_recapture import RecaptureSimulator

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FREUIDDataset(Dataset):
    def __init__(self, df, transform, recapture=None, p_recapture=0.0,
                 path_col="abs_path", label_col="label", return_id=False, is_test=False):
        """
        df          : DataFrame with `path_col` (absolute image path) and (unless is_test) `label_col`.
        transform   : albumentations transform producing a CHW float tensor.
        recapture   : RecaptureSimulator instance (or None).
        p_recapture : probability of applying the recapture simulator to a sample.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.recapture = recapture
        self.p_recapture = p_recapture
        self.path_col = path_col
        self.label_col = label_col
        self.return_id = return_id
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def _read(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # robust fallback via PIL for odd encodings
            from PIL import Image
            img = np.array(Image.open(path).convert("RGB"))
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._read(row[self.path_col])

        if self.recapture is not None and self.p_recapture > 0:
            # Uses the worker-level numpy RNG so train.py/train_fast.py can seed workers.
            if np.random.random() < self.p_recapture:
                image = self.recapture(image)

        image = self.transform(image=image)["image"]

        if self.is_test:
            out_label = torch.tensor(0.0, dtype=torch.float32)
        else:
            out_label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)

        if self.return_id:
            return image, out_label, str(row.get("id", idx))
        return image, out_label


def get_transforms(img_size=512):
    """Version-stable albumentations transforms. Heavy domain aug lives in RecaptureSimulator."""
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(std_range=(0.02, 0.05), p=1.0),
        ], p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(int(img_size*0.02), int(img_size*0.08)),
                        hole_width_range=(int(img_size*0.02), int(img_size*0.08)),
                        fill=0, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_transform, val_transform


def get_recapture(strength="medium", force_macro=False):
    return RecaptureSimulator(strength=strength, force_macro=force_macro)
