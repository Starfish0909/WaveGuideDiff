# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

import random
import os

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data_yield(loader):
    while True:
        yield from loader


def load_data_inpa_skip_existing(
        *,
        gt_path=None,
        mask_path=None,
        batch_size,
        image_size,
        class_cond=False,
        deterministic=True,
        # random_crop=False,
        # random_flip=False,
        return_dataloader=False,
        return_dict=False,
        max_len=None,
        drop_last=False,
        conf=None,
        offset=0,
        skip_existing=True,
        **kwargs
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param gt_path: ground truth images directory path.
    :param mask_path: mask images directory path.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: DISABLED for medical images.
    :param random_flip: DISABLED for medical images.
    :param skip_existing: if True, skip images that already have output files.
    """

    gt_dir = os.path.expanduser(gt_path)
    mask_dir = os.path.expanduser(mask_path)

    gt_paths = sorted(_list_image_files_recursively(gt_dir))
    mask_paths = sorted(_list_image_files_recursively(mask_dir))

    print(f"Loading data: {len(gt_paths)} GT images, {len(mask_paths)} mask images")

    assert len(gt_paths) == len(mask_paths)

    # If skip existing files feature is enabled
    if skip_existing and conf is not None:
        # Get output directory path
        dset = 'eval'
        eval_name = conf.get_default_eval_name()
        srs_dir = os.path.expanduser(conf['data'][dset][eval_name]['paths']['srs'])

        # Filter out images that already have output files
        filtered_gt_paths = []
        filtered_mask_paths = []
        skipped_count = 0

        for gt_path, mask_path in zip(gt_paths, mask_paths):
            img_name = os.path.basename(gt_path)
            srs_path = os.path.join(srs_dir, img_name)

            if os.path.exists(srs_path):
                skipped_count += 1
                if skipped_count <= 5:  # Only show first 5 skipped files
                    print(f"Skipping already inferred image: {img_name}")
                elif skipped_count == 6:
                    print("... (more skipped images)")
            else:
                filtered_gt_paths.append(gt_path)
                filtered_mask_paths.append(mask_path)

        gt_paths = filtered_gt_paths
        mask_paths = filtered_mask_paths

        print(f"Filter result: skipped {skipped_count} already inferred images, {len(gt_paths)} remaining to infer")

    classes = None
    if class_cond:
        raise NotImplementedError()

    dataset = ImageDatasetInpaSkip(
        image_size,
        gt_paths=gt_paths,
        mask_paths=mask_paths,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=False,  # Force disable
        random_flip=False,  # Force disable
        return_dict=return_dict,
        max_len=max_len,
        conf=conf,
        offset=offset
    )

    # Get num_workers from config, default to 0 to avoid shared memory issues
    num_workers = kwargs.get('num_workers', 0)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last
        )

    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last
        )

    if return_dataloader:
        return loader
    else:
        return load_data_yield(loader)


def load_data_inpa(
        *,
        gt_path=None,
        mask_path=None,
        batch_size,
        image_size,
        class_cond=False,
        deterministic=True,
        # random_crop=False,
        # random_flip=False,
        return_dataloader=False,
        return_dict=False,
        max_len=None,
        drop_last=False,
        conf=None,
        offset=0,
        **kwargs
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Designed specifically for medical images, disables all data augmentation operations, uses high-quality resize to ensure consistency with training.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param gt_path: ground truth images directory path.
    :param mask_path: mask images directory path.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: DISABLED for medical images.
    :param random_flip: DISABLED for medical images.
    """

    gt_dir = os.path.expanduser(gt_path)
    mask_dir = os.path.expanduser(mask_path)

    gt_paths = sorted(_list_image_files_recursively(gt_dir))
    mask_paths = sorted(_list_image_files_recursively(mask_dir))

    print(f"Loading data: {len(gt_paths)} GT images, {len(mask_paths)} mask images")

    assert len(gt_paths) == len(mask_paths)

    classes = None
    if class_cond:
        raise NotImplementedError()

    dataset = ImageDatasetInpa(
        image_size,
        gt_paths=gt_paths,
        mask_paths=mask_paths,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=False,  # Force disable
        random_flip=False,  # Force disable
        return_dict=return_dict,
        max_len=max_len,
        conf=conf,
        offset=offset
    )

    # Get num_workers from config, default to 0 to avoid shared memory issues
    num_workers = kwargs.get('num_workers', 0)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last
        )

    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last
        )

    if return_dataloader:
        return loader
    else:
        return load_data_yield(loader)


def _list_image_files_recursively(data_dir):
    """Recursively list all image files in directory"""
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "tif", "tiff", "bmp"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            # Recursively process subdirectories
            results.extend(_list_image_files_recursively(full_path))
    return sorted(results)


class ImageDatasetInpaSkip(Dataset):
    def __init__(
            self,
            resolution,
            gt_paths,
            mask_paths,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,  # Already disabled
            random_flip=False,  # Already disabled
            return_dict=False,
            max_len=None,
            conf=None,
            offset=0
    ):
        super().__init__()
        self.resolution = resolution

        # Note: gt_paths and mask_paths are already sorted lists when passed in
        # If offset is needed, it should have been handled before passing in (e.g., filtered in load_data_inpa_skip_existing)
        # No need to sort again here, just use directly
        self.local_gts = gt_paths[shard:][::num_shards]
        self.local_masks = mask_paths[shard:][::num_shards]

        self.local_classes = None if classes is None else classes[shard:][::num_shards]

        # Force disable all data augmentation operations
        self.random_crop = False
        self.random_flip = False
        self.return_dict = return_dict
        self.max_len = max_len

        # Warn if user tries to enable these operations
        if random_crop:
            print("WARNING: random_crop has been disabled for medical images.")
        if random_flip:
            print("WARNING: random_flip has been disabled for medical images.")

    def __len__(self):
        if self.max_len is not None:
            return self.max_len

        return len(self.local_gts)

    def __getitem__(self, idx):
        gt_path = self.local_gts[idx]
        pil_gt = self.imread(gt_path)

        mask_path = self.local_masks[idx]
        pil_mask = self.imread(mask_path)

        # Use high-quality resize method fully consistent with training
        arr_gt = self.resize_to_array_high_quality(pil_gt, self.resolution)
        arr_mask = self.resize_to_array_high_quality(pil_mask, self.resolution)

        # Medical images disable all data augmentation operations
        # No random transformations

        # Normalization processing - consistent with training
        arr_gt = arr_gt.astype(np.float32) / 127.5 - 1  # GT: [-1, 1]
        arr_mask = arr_mask.astype(np.float32) / 255.0  # Mask: [0, 1]

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        if self.return_dict:
            name = os.path.basename(gt_path)
            return {
                'GT': np.transpose(arr_gt, [2, 0, 1]),
                'GT_name': name,
                'gt_keep_mask': np.transpose(arr_mask, [2, 0, 1]),
            }
        else:
            raise NotImplementedError()

    def imread(self, path):
        with open(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image

    def resize_to_array_high_quality(self, pil_image, image_size):
        """
        High-quality resize method - fully consistent with training code
        Uses three-step method to ensure best resize quality, minimizing distortion
        """
        # Step 1: High-quality downsampling - gradually reduce when image is very large to improve quality
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        # Step 2: Scale proportionally to appropriate size, making shortest edge close to target size
        scale = image_size / min(*pil_image.size)
        new_size = tuple(round(x * scale) for x in pil_image.size)
        pil_image = pil_image.resize(new_size, resample=Image.BICUBIC)

        # Step 3: Final resize to exact target size
        pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)

        return np.array(pil_image)


class ImageDatasetInpa(Dataset):
    """Standard image dataset class (does not skip already inferred images)"""
    def __init__(
            self,
            resolution,
            gt_paths,
            mask_paths,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,  # Already disabled
            random_flip=False,  # Already disabled
            return_dict=False,
            max_len=None,
            conf=None,
            offset=0
    ):
        super().__init__()
        self.resolution = resolution

        gt_paths = sorted(gt_paths)[offset:]
        mask_paths = sorted(mask_paths)[offset:]

        self.local_gts = sorted(gt_paths)[shard:][::num_shards]
        self.local_masks = sorted(mask_paths)[shard:][::num_shards]

        self.local_classes = None if classes is None else classes[shard:][::num_shards]

        # Force disable all data augmentation operations
        self.random_crop = False
        self.random_flip = False
        self.return_dict = return_dict
        self.max_len = max_len

        # Warn if user tries to enable these operations
        if random_crop:
            print("WARNING: random_crop has been disabled for medical images.")
        if random_flip:
            print("WARNING: random_flip has been disabled for medical images.")

    def __len__(self):
        if self.max_len is not None:
            return self.max_len

        return len(self.local_gts)

    def __getitem__(self, idx):
        gt_path = self.local_gts[idx]
        pil_gt = self.imread(gt_path)

        mask_path = self.local_masks[idx]
        pil_mask = self.imread(mask_path)

        # Use high-quality resize method fully consistent with training
        arr_gt = self.resize_to_array_high_quality(pil_gt, self.resolution)
        arr_mask = self.resize_to_array_high_quality(pil_mask, self.resolution)

        # Medical images disable all data augmentation operations
        # No random transformations

        # Normalization processing - consistent with training
        arr_gt = arr_gt.astype(np.float32) / 127.5 - 1  # GT: [-1, 1]
        arr_mask = arr_mask.astype(np.float32) / 255.0  # Mask: [0, 1]

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        if self.return_dict:
            name = os.path.basename(gt_path)
            return {
                'GT': np.transpose(arr_gt, [2, 0, 1]),
                'GT_name': name,
                'gt_keep_mask': np.transpose(arr_mask, [2, 0, 1]),
            }
        else:
            raise NotImplementedError()

    def imread(self, path):
        with open(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image

    def resize_to_array_high_quality(self, pil_image, image_size):
        """
        High-quality resize method - fully consistent with training code
        Uses three-step method to ensure best resize quality, minimizing distortion
        """
        # Step 1: High-quality downsampling - gradually reduce when image is very large to improve quality
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        # Step 2: Scale proportionally to appropriate size, making shortest edge close to target size
        scale = image_size / min(*pil_image.size)
        new_size = tuple(round(x * scale) for x in pil_image.size)
        pil_image = pil_image.resize(new_size, resample=Image.BICUBIC)

        # Step 3: Final resize to exact target size
        pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)

        return np.array(pil_image)

