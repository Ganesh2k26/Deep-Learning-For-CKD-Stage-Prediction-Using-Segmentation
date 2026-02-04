import os
import csv
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class CKDClassificationDataset(Dataset):
    """
    Dataset for CKD stage classification from kidney ROI images.

    Expects:
      - images_dir: folder with .png images (kidney ROIs, e.g. data/classification/kidney_rois)
      - labels_csv: CSV with columns: filename, stage
          example rows:
              filename,stage
              case_00001.png,2
              case_00002.png,4

      stage is an integer in {1,2,3,4} (we usually ignore -1 = unlabeled).
    """

    def __init__(
        self,
        images_dir: str,
        labels_csv: str,
        transform: A.BasicTransform = None,
        ignore_label: int = -1,
    ):
        self.images_dir = images_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        missing_count = 0
        total_rows = 0

        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1

                # ---- read filename & stage from CSV ----
                fname = row["filename"]
                stage = int(row["stage"])

                # skip unlabeled rows (stage == -1)
                if stage == ignore_label:
                    continue

                img_path = os.path.join(images_dir, fname)

                # ---- skip rows where image file is missing ----
                if not os.path.isfile(img_path):
                    missing_count += 1
                    continue

                # keep this (filename, stage)
                self.samples.append((fname, stage))

        if missing_count > 0:
            print(
                f"[CKDClassificationDataset] Skipped {missing_count} rows "
                f"because ROI image did not exist."
            )

        if len(self.samples) == 0:
            print("[WARNING] CKDClassificationDataset: no valid samples found!")

        # build label mapping: sorted unique stages
        self.classes = sorted({s for _, s in self.samples})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        print(f"[CKDClassificationDataset] Loaded {len(self.samples)} samples "
              f"from {total_rows} CSV rows. Classes: {self.classes}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, stage = self.samples[idx]
        img_path = os.path.join(self.images_dir, fname)

        # We already checked existence in __init__, but be safe
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found (unexpected): {img_path}")

        # normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        # H, W -> H, W, 1  (Albumentations expects channel-last)
        img = np.expand_dims(img, axis=2)

        if self.transform is not None:
            out = self.transform(image=img)
            img_tensor = out["image"]       # already CHW float32
        else:
            # convert to CHW tensor manually
            img = np.transpose(img, (2, 0, 1))   # 1, H, W
            img_tensor = torch.from_numpy(img)

        # map true stage (1–4) to class index (0–3)
        label_idx = self.class_to_idx[stage]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return img_tensor, label_tensor
