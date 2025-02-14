"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "sentence": ann["sentence"],
                "label": ann["label"],
                "image": sample["image"],
            }
        )


class ImageNetLT_Dataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


    def __getitem__(self, index):
        ann = self.annotation[index]

        image_id = ann["image_id"]
        image_path = os.path.join(self.vis_root, ann['image'])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        sentence = self.text_processor(ann["text_input"])

        return {
            "image": image,
            "text_input": sentence,
            "label": ann["label"],
            "image_id": image_id
        }
