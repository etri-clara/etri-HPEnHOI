# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

home_dir = os.path.abspath(os.getcwd())
print(home_dir)
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)

import pprint
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from lib.dataset.coco import COCODataset
from lib.models.backbones.vit import ViT
from lib.models.heads import TopdownHeatmapSimpleHead
from lib.models.vit_pose import ViTPose
from lib.models.losses import JointsMSELoss

from lib.core.config import config
from lib.core.config import update_config
from lib.core.function import validate

from lib.utils.utils import create_logger
from lib.utils.utils import ShellColors as sc
# python tools/valid_vit.py --cfg experiments/coco/vit/vit_small.yaml --weight checkpoints/vitpose_small.pth --gpus 0
# python tools/valid_vit.py --cfg experiments/coco/vit/vit_large.yaml --weight checkpoints/vitpose_large.pth --gpus 0

def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    
    # general
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument("--weight", help="checkpoint name", required=True, type=str)
    parser.add_argument("--wandb", help="use wandb", action="store_true")

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

def show_info(args, config):
    print(f"{'='*20} Info {'='*20}")
    print(f"{sc.COLOR_GREEN}CONFIG: {sc.ENDC}{args.cfg}")
    print(f"{sc.COLOR_GREEN}WEIGHT: {sc.ENDC}{args.weight}")
    print(f"{sc.COLOR_GREEN}TEST BATCH SIZE: {sc.ENDC}{config.TEST.BATCH_SIZE}")
    print(f"{sc.COLOR_CYAN}USE UDP: {sc.ENDC}{config.TEST.USE_UDP}")
    print(f"{sc.COLOR_CYAN}USE FLIP: {sc.ENDC}{config.TEST.FLIP_TEST}")
    print(f"{sc.COLOR_CYAN}USE GT BBOX: {sc.ENDC}{config.TEST.USE_GT_BBOX}")
    print(f"{sc.COLOR_CYAN}USE UNCERTAINTY: {sc.ENDC}{config.LOSS.UNCERTAINTY}")
    print(f"{sc.COLOR_CYAN}USE WANDB: {sc.ENDC}{args.wandb}")
    print(f"{'='*46}")

def main():
    args = parse_args()
    reset_config(config, args)

    wdb = None
    if args.wandb and gpus == 0:
        import wandb
        wdb = wandb
        wdb.init(
            config=config,
            project="HPE_Validation_230911",
            name=f"{config.MODEL.NAME}_{config.MODEL.TYPE}_{config.DATASET.DATASET}_{config.LOSS.HM_LOSS}_{config.LOSS.UNC_LOSS}_{config.TRAIN.END_EPOCH}",
        )

    logger, final_output_dir = create_logger(config, args.cfg, "valid")
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    show_info(args, config)
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED


    if "small" in args.cfg:
        from lib.models.extra.vit_small_uncertainty_config import extra
    elif "large" in args.cfg:
        from lib.models.extra.vit_large_uncertainty_config import extra
    elif "huge" in args.cfg:
        from lib.models.extra.vit_huge_uncertainty_config import extra
    else:
        raise FileNotFoundError(f"Check config file name!!")
    
    backbone = ViT(
        img_size=extra["backbone"]["img_size"],
        patch_size=extra["backbone"]["patch_size"],
        embed_dim=extra["backbone"]["embed_dim"],
        in_channels=3,
        num_heads=extra["backbone"]["num_heads"],
        depth=extra["backbone"]["depth"],
        qkv_bias=True,
    )

    deconv_head = TopdownHeatmapSimpleHead(
        in_channels=extra["keypoint_head"]["in_channels"],
        num_deconv_layers=extra["keypoint_head"]["num_deconv_layers"],
        num_deconv_filters=extra["keypoint_head"]["num_deconv_filters"],
        num_deconv_kernels=extra["keypoint_head"]["num_deconv_kernels"],
        extra=dict(final_conv_kernel=1),
        out_channels=17,
    )

    model = ViTPose(backbone, deconv_head)

    checkpoint_path = os.path.join(home_dir, args.weight)
    
    model = model.custom_init_weights(model, checkpoint_path)
    # model.init_weights(checkpoint_path, map_location="cpu")

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
        keypoint_head=deconv_head,
        wdb=wdb
    )


if __name__ == "__main__":
    main()
