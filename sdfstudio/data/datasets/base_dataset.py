# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from rich.progress import Console, track
from torch.utils.data import Dataset
from torch import Tensor as TensorType

from sdfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from sdfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from sdfstudio.utils.images import BasicImages


class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.has_masks = dataparser_outputs.mask_filenames is not None
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.image_cache = {}

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
        # mask_filename = str(image_filename).replace("dense/images", "masks").replace(".jpg", ".npy")
        # mask = np.load(mask_filename)
        # image = image * mask[..., None]

        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int) -> TensorType:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert image.shape[-1] == 4
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        else:
            image = image[:, :, :3]
        return image

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        if image_idx in self.image_cache:
            image = self.image_cache[image_idx]
        else:
            image = self.get_image(image_idx)
            self.image_cache[image_idx] = image

        data = {"image_idx": image_idx}
        data["image"] = image
        for _, data_func_dict in self._dataparser_outputs.additional_inputs.items():
            assert "func" in data_func_dict, "Missing function to process data: specify `func` in `additional_inputs`"
            func = data_func_dict["func"]
            assert "kwargs" in data_func_dict, "No data to process: specify `kwargs` in `additional_inputs`"
            data.update(func(image_idx, **data_func_dict["kwargs"]))
        if self.has_masks:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    # pylint: disable=no-self-use
    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        del data
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data


class GeneralizedDataset(InputDataset):
    """Dataset that returns images, possibly of different sizes.

    The only thing that separates this from the inputdataset is that this will return
    image / mask tensors inside a list, meaning when collate receives the images, it will
    simply concatenate the lists together. The concatenation of images of different sizes would
    fail otherwise.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        h = None
        w = None
        all_hw_same = True
        for filename in track(
            self._dataparser_outputs.image_filenames, transient=True, description="Checking image sizes"
        ):
            image = Image.open(filename)
            if h is None:
                h = image.height
                w = image.width

            if image.height != h or image.width != w:
                all_hw_same = False
                break

        self.all_hw_same = all_hw_same

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        # If all images are the same size, we can just return the image and mask tensors in a regular way
        if self.all_hw_same:
            return super().get_data(image_idx)

        # Otherwise return them in a custom struct
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx}
        data["image"] = BasicImages([image])
        for _, data_func_dict in self._dataparser_outputs.additional_inputs.items():
            assert "func" in data_func_dict, "Missing function to process data: specify `func` in `additional_inputs`"
            func = data_func_dict["func"]
            assert "kwargs" in data_func_dict, "No data to process: specify `kwargs` in `additional_inputs`"
            data.update(func(image_idx, **data_func_dict["kwargs"]))
        if self.has_masks:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = BasicImages([get_image_mask_tensor_from_path(filepath=mask_filepath)])
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data
