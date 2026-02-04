import os
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class KidneySegmentationDataset(Dataset):
    """
    Dataset for kidney-only binary segmentation.
    Uses:
        data/raw/ct_slices      -> images
        data/annotations/ct_masks -> masks
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        file_list,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.file_list[idx]
        img_path = os.path.join(self.images_dir, fname)
        mask_name = fname.replace(".png", "_mask.png")
        mask_path = os.path.join(self.masks_dir, mask_name)

        # read image & mask
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # normalize to 0-1 float
        img = img.astype("float32") / 255.0
        mask = (mask > 0).astype("float32")  # binary: 0 or 1

        # HWC -> add channel dimension
        img = np.expand_dims(img, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # HWC -> CHW for PyTorch
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)

        return img_tensor, mask_tensor
