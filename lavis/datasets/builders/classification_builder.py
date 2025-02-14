"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.nlvr_datasets import NLVRDataset, NLVREvalDataset
from lavis.datasets.datasets.snli_ve_datasets import SNLIVisualEntialmentDataset
from lavis.datasets.datasets.ImageNetLT_datasets import ImageNetLT_Dataset
from lavis.datasets.datasets.iNat18_datasets import iNat18_Dataset
from lavis.datasets.datasets.PlacesLT_datasets import PlacesLT_Dataset

@registry.register_builder("nlvr")
class NLVRBuilder(BaseDatasetBuilder):
    train_dataset_cls = NLVRDataset
    eval_dataset_cls = NLVREvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/nlvr/defaults.yaml"}


@registry.register_builder("snli_ve")
class SNLIVisualEntailmentBuilder(BaseDatasetBuilder):
    train_dataset_cls = SNLIVisualEntialmentDataset
    eval_dataset_cls = SNLIVisualEntialmentDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/snli_ve/defaults.yaml"}

@registry.register_builder("ImageNetLT")
class ImageNetLTBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageNetLT_Dataset
    eval_dataset_cls = ImageNetLT_Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/imagenet/defaults_lt_cls.yaml"}

@registry.register_builder("iNat18")
class iNat18Builder(BaseDatasetBuilder):
    train_dataset_cls = iNat18_Dataset
    eval_dataset_cls = iNat18_Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/iNat18/defaults_lt_cls.yaml"}

@registry.register_builder("PlacesLT")
class PlacesLTBuilder(BaseDatasetBuilder):
    train_dataset_cls = PlacesLT_Dataset
    eval_dataset_cls = PlacesLT_Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/PlacesLT/defaults_lt_cls.yaml"}
