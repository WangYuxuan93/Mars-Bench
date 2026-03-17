"""
Wrapper class to adapt any segmentation dataset for Mask2Former.
"""

import numpy as np
import cv2
from torch.utils.data import Dataset


class Mask2FormerWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        # HFSegmentation stores PIL objects in self.images / self.masks;
        # file-based datasets store real paths in image_paths / gts.
        if hasattr(self.dataset, "images") and self.dataset.images:
            image = np.array(self.dataset.images[index].convert("RGB")).astype("uint8")
            mask = np.array(self.dataset.masks[index].convert("L")).astype("float32")
        else:
            image = cv2.imread(str(self.dataset.image_paths[index]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {self.dataset.image_paths[index]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("uint8")
            mask = cv2.imread(str(self.dataset.gts[index]), 0)  # gts, not ground
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {self.dataset.gts[index]}")
            mask = mask.astype("float32")

        transformed = self.dataset.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        orig_image = image.clone()
        orig_mask = mask.clone()

        return image, mask, orig_image, orig_mask
