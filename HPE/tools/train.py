# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import pprint
import shutil

home_dir = os.path.abspath(os.getcwd())
print(home_dir)
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)


import wandb
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms


from lib.models.vit_pose import ViTPose
from lib.models.heads import TopdownHeatmapSimpleHead
from lib.models.backbones.vit import ViT
from lib.models.losses import JointsMSELoss

from lib.core.scheduler import MultistepWarmUpRestargets
from lib.core.config import config
from lib.core.config import update_config
from lib.core.function import validate, train
from lib.core.config import get_model_name
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.dataset.coco import COCODataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument(
        "--frequent", help="frequency of logging", default=config.PRINT_FREQ, type=int
    )
    parser.add_argument("--gpus", help="gpus", type=str)
    parser.add_argument("--workers", help="num of dataloader workers", type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    args = parse_args()
    reset_config(config, args)
    config.LOSS.UNCERTAINTY = False
    config.DATASET.TARGET_KEYPOINT = False
    config.DATASET.TARGET_HEATMAP = True
    wnb = wandb
    wnb.init(config=config, project="VIT_uncertainty", name=config.MODEL.NAME)

    logger, final_output_dir, _ = create_logger(config, args.cfg, "train")

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    # Load Model
    backbone = ViT(img_size=(256, 192), embed_dim=384)
    deconv_head = TopdownHeatmapSimpleHead(
        in_channels=384,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
        out_channels=17,
    )
    vit_pose = ViTPose(backbone, deconv_head)

    checkpoint = home_dir + "/checkpoints/vitpose_small.pth"
    print(checkpoint)
    vit_pose.init_weights(checkpoint)
    model = vit_pose.to(device)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, "../lib/models/backbones", config.MODEL.NAME + ".py"),
        final_output_dir,
    )

    dump_input = torch.rand(
        (
            config.TRAIN.BATCH_SIZE,
            3,
            config.MODEL.IMAGE_SIZE[1],
            config.MODEL.IMAGE_SIZE[0],
        )
    )
    dump_input = dump_input.to(device)
    gpus = [int(i) for i in config.GPUS.split(",")]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=True).cuda()

    optimizer = get_optimizer(config, model)

    lr_scheduler = MultistepWarmUpRestargets(
        optimizer,
        milestones=config.TRAIN.LR_STEP,
    )

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = COCODataset(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
    )

    validate(
        config,
        valid_loader,
        valid_dataset,
        model,
        criterion,
        final_output_dir,
        wnb=wnb,
    )

    best_perf = 0.0
    perf_indicator = 0.0
    best_model = False

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        lr_ = lr_scheduler.get_lr()[0]
        wnb.log({"learning_rate": float(lr_)})

        # train for one epoch
        train(
            config,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            final_output_dir,
            wnb=wnb,
        )

        # evaluate on validation set
        if epoch % config.EVALUATION.INTERVAL == 0:
            perf_indicator = validate(
                config,
                valid_loader,
                valid_dataset,
                model,
                criterion,
                final_output_dir,
                wnb=wnb,
            )

            model_state_file = os.path.join(
                final_output_dir, f"{config.MODEL.NAME}_{epoch}.pth.tar"
            )
            logger.info("saving final model state to {}".format(model_state_file))
            torch.save(model.module.state_dict(), model_state_file)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True

        else:
            best_model = False

        if best_model == True:
            logger.info("=> saving checkpoint to {}".format(final_output_dir))
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": get_model_name(config),
                    "state_dict": model.state_dict(),
                    "perf": perf_indicator,
                    "optimizer": optimizer.state_dict(),
                },
                best_model,
                final_output_dir,
            )

    final_model_state_file = os.path.join(final_output_dir, "final_state.pth.tar")
    logger.info("saving final model state to {}".format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)


if __name__ == "__main__":
    main()
