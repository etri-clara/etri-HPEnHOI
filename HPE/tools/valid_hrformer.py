# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

home_dir = os.path.abspath(os.getcwd())
print(home_dir)
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)

from lib.dataset.coco import COCODataset
from lib.models.backbones.hrformer import HRFormer
from lib.models.heads.topdown_heatmap_identity_head import TopdownHeatmapIdentityHead
from lib.models.extra.hrformer_small_uncertainty_config import extra
from lib.models.extra.hrformer_small_uncertainty_config import norm_cfg
from lib.models.hrformer_pose import HRFormerPose
from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import update_dir
from lib.models.losses import JointsMSELoss
from lib.core.function import validate
from lib.utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument(
        "--frequent", help="frequency of logging", default=config.PRINT_FREQ, type=int
    )
    parser.add_argument("--gpus", help="gpus", type=str)
    parser.add_argument("--workers", help="num of dataloader workers", type=int)
    parser.add_argument("--model-file", help="model state file", type=str)
    parser.add_argument("--use-detect-bbox", help="use detect bbox", action="store_true")
    parser.add_argument("--flip-test", help="use flip test", action="store_true")
    parser.add_argument("--post-process", help="use post process", action="store_true")
    parser.add_argument("--shift-heatmap", help="shift heatmap", action="store_true")
    parser.add_argument("--coco-bbox-file", help="coco detection bbox file", type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file


def main():
    args = parse_args()
    reset_config(config, args)
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, "valid")

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    backbone = HRFormer(extra, 3, norm_cfg=norm_cfg)
    keypoint_head = TopdownHeatmapIdentityHead(
        in_channels=extra["stage4"]["num_channels"][0], out_channls=extra["joint_num"]
    )
    model = HRFormerPose(backbone, keypoint_head)

    checkpoint = home_dir + "/checkpoints/hrformer_small_best.pth"
    model.custom_init_weights(model, checkpoint)

    gpus = [int(i) for i in config.GPUS.split(",")]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_dataset = COCODataset(
        cfg=config,
        root=config.DATASET.ROOT,
        image_set=config.DATASET.TEST_SET,
        is_train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
    )

    # evaluate on validation set
    validate(
        config,
        valid_loader,
        valid_dataset,
        model,
        criterion,
        final_output_dir,
        backbone=backbone,
        keypoint_head=keypoint_head,
    )


if __name__ == "__main__":
    main()
