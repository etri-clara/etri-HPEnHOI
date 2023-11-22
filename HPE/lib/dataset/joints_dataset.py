# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils.transforms import get_affine_transform
from lib.utils.transforms import affine_transform
from lib.utils.transforms import fliplr_joints
from lib.utils.post_transforms import get_warp_matrix
from lib.utils.post_transforms import warp_affine_joints
from lib.utils.io import imfrombytes

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set
        self.output_path = cfg.OUTPUT_DIR

        self.dataset_name = cfg.DATASET.DATASET
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.shift_factor = cfg.DATASET.SHIFT_FACTOR
        self.shift_prob = cfg.DATASET.SHIFT_PROB
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.heatmap_type = cfg.MODEL.EXTRA.HEATMAP_TYPE
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.transform = transform
        self.use_udp = cfg.TEST.USE_UDP
        self.db = []

        # for uncertainty
        self.uncertainty = cfg.LOSS.UNCERTAINTY
        self.normalized_map = cfg.LOSS.NORMALIZED_MAP
        self.is_target_keypoints = cfg.DATASET.TARGET_KEYPOINT
        self.is_target_heatmap = cfg.DATASET.TARGET_HEATMAP

        # Half body transform
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(
        self,
    ):
        return len(self.db)

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, "rb") as f:
            value_buf = f.read()
        return value_buf

    def _read_image(self, path):
        filepath = str(path)
        with open(filepath, "rb") as f:
            value_buf = f.read()
        img_bytes = value_buf
        img = imfrombytes(img_bytes, "color", channel_order="rgb")
        if img is None:
            raise ValueError(f"Fail to read {path}")
        return img

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec["image"]
        filename = db_rec["filename"] if "filename" in db_rec else ""
        bbox = db_rec["bbox"] if "bbox" in db_rec else ""
        bbox_id = db_rec["bbox_id"] if "bbox_id" in db_rec else ""

        # sigma = db_rec["sigma"] if "sigma" in db_rec else ""
        idx = db_rec["idx"] if "idx" in db_rec else ""

        data_numpy = self._read_image(image_file)

        if data_numpy is None:
            logger.error("=> fail to read {}".format(image_file))
            raise ValueError("Fail to read {}".format(image_file))

        joints = db_rec["joints_2d"]
        joints_vis = db_rec["joints_2d_vis"]

        center = db_rec["center"]
        scale = db_rec["scale"]
        score = db_rec["score"] if "score" in db_rec else 1
        rotation = 0

        if self.is_train:
            # TopDownGetRandomScaleRotation
            sf = self.scale_factor
            rf = self.rotation_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rotation = (
                np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
            )
            # TopDownRandomFlip
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs
                )
                center[0] = data_numpy.shape[1] - center[0] - 1

            # RandomShiftBboxCenter
            pixel_std: float = 200.0
            if np.random.rand() < self.shift_prob:
                center += np.random.uniform(-1, 1, 2) * self.shift_factor * scale * pixel_std

            # Half body transform
            if (
                np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body
            ):
                c, s = self.half_body_transform(joints, joints_vis)
                if c is not None and s is not None:
                    center = c
                    scale = s

        # change #
        # TopDownAffine
        if self.use_udp:
            trans = get_warp_matrix(rotation, center * 2.0, self.image_size - 1.0, scale * 200.0)
            input_img = cv2.warpAffine(
                data_numpy,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR,
            )
            joints[:, 0:2] = warp_affine_joints(joints[:, 0:2].copy(), trans)

        else:
            trans = get_affine_transform(center, scale, rotation, self.image_size)
            input_img = cv2.warpAffine(
                data_numpy,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR,
            )

            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        # ToTensor & NormalizeTensor
        if self.transform:
            input_img = self.transform(input_img)

        meta = {
            "image": image_file,
            "filename": filename,
            "joints": joints,
            "joints_vis": joints_vis,
            "center": center,
            "scale": scale,
            "rotation": rotation,
            "score": score,
            "bbox": bbox,
            "bbox_id": bbox_id,
            "flip_pairs": self.flip_pairs,
        }

        target_joints = torch.from_numpy(joints)
        target_joints_vis = torch.from_numpy(joints_vis[:, 1])

        if self.is_target_heatmap:
            heatmap, heatmap_weight = self._generate_heatmap(joints, joints_vis)
            heatmap = torch.from_numpy(heatmap)
            heatmap_weight = torch.from_numpy(heatmap_weight)
            if self.is_target_keypoints:
                return (
                    input_img,
                    target_joints,
                    target_joints_vis,
                    heatmap,
                    heatmap_weight,
                    meta,
                )
            return input_img, heatmap, heatmap_weight, meta
        return input_img, target_joints, target_joints_vis, meta

    def _generate_heatmap(self, joints, joints_vis):
        """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: heatmap, heatmap_weight(1: visible, 0: invisible)
        """
        heatmap_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        heatmap_weight[:, 0] = joints_vis[:, 0]

        assert self.heatmap_type == "gaussian", "Only support gaussian map now!"

        if self.heatmap_type == "gaussian":
            heatmap = np.zeros(
                (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32,
            )

            tmp_size = self.sigma * 3
            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if (
                    ul[0] >= self.heatmap_size[0]
                    or ul[1] >= self.heatmap_size[1]
                    or br[0] < 0
                    or br[1] < 0
                ):
                    # If not, just return the image as is
                    heatmap_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = heatmap_weight[joint_id]
                if v > 0.5:
                    heatmap[joint_id][img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[
                        g_y[0] : g_y[1], g_x[0] : g_x[1]
                    ]

        return heatmap, heatmap_weight

    def half_body_transform(self, joints, joints_visible):
        """Get center&scale for half-body transform."""
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_visible[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        elif len(lower_joints) > 2:
            selected_joints = lower_joints
        else:
            selected_joints = upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)

        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        aspect_ratio = self.image_size[0] / self.image_size[1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * 1.5
        return center, scale
