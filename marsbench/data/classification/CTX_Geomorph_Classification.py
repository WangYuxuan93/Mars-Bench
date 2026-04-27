"""
CTX geomorphology classification dataset built from Mars shapefile annotations.
"""

import os
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import pandas as pd

from .BaseClassificationDataset import BaseClassificationDataset


class CTX_Geomorph_Classification(BaseClassificationDataset):

    def __init__(
        self,
        cfg,
        data_dir,
        transform,
        annot_csv: Union[str, os.PathLike],
        split: Literal["train", "val", "test"] = "train",
    ):
        self.split = split
        self.annot = pd.read_csv(annot_csv)
        self.annot = self.annot[self.annot["split"] == split]
        super(CTX_Geomorph_Classification, self).__init__(cfg, data_dir, transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_ids = self.annot["file_id"].astype(str).tolist()
        gts = self.annot["label"].astype(int).tolist()
        # file_id 已包含类别子目录，直接拼在 images/ 下
        image_paths = [
            os.path.join(self.data_dir, "images", file_id.replace("\\", "/"))
            for file_id in image_ids
        ]
        return image_paths, gts
