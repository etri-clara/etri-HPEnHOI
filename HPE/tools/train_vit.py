import sys, os
import os
import argparse
import torchvision.transforms as transforms

home_dir = os.path.dirname(os.path.abspath(__file__ + "/../"))
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from lib.models.backbones.vit import ViT
from lib.models.heads import TopdownHeatmapSimpleHead
from lib.dataset.coco import COCODataset

from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import get_model_name
from lib.core.scheduler import MultistepWarmUpRestargets
from lib.core.function import train, validate 

from lib.utils.utils import get_optimizer, get_vit_optimizer
from lib.utils.utils import get_loss
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.utils import ShellColors as sc
from lib.utils.utils import init_random_seed, set_random_seed
from lib.utils.utils import load_checkpoint

def show_info(gpu, args, config):
    print(f"{'='*20} Info {'='*20}")
    print(f"{sc.COLOR_GREEN}CURRENT GPU: {sc.ENDC}{gpu}")
    print(f"{sc.COLOR_GREEN}CONFIG: {sc.ENDC}{args.cfg}")
    print(f"{sc.COLOR_GREEN}WEIGHT: {sc.ENDC}{args.weight}")
    print(f"{sc.COLOR_GREEN}SEED: {sc.ENDC}{args.seed}")
    print(f"{sc.COLOR_GREEN}TRAIN BATCH SIZE: {sc.ENDC}{config.TRAIN.BATCH_SIZE}")
    print(f"{sc.COLOR_GREEN}TEST BATCH SIZE: {sc.ENDC}{config.TEST.BATCH_SIZE}")
    print(f"{sc.COLOR_CYAN}USE AMP: {sc.ENDC}{config.MODEL.USE_AMP}")
    print(f"{sc.COLOR_CYAN}USE UDP: {sc.ENDC}{config.TEST.USE_UDP}")
    print(f"{sc.COLOR_CYAN}USE FLIP: {sc.ENDC}{config.TEST.FLIP_TEST}")
    print(f"{sc.COLOR_CYAN}USE GT BBOX: {sc.ENDC}{config.TEST.USE_GT_BBOX}")
    print(f"{sc.COLOR_CYAN}USE UNCERTAINTY: {sc.ENDC}{config.LOSS.UNCERTAINTY}")
    print(f"{sc.COLOR_CYAN}USE WARMUP: {sc.ENDC}{args.warmup}")
    print(f"{sc.COLOR_CYAN}USE WANDB: {sc.ENDC}{args.wandb}")
    print(f"{'='*46}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="")
    parser.add_argument("--weight", help="checkpoint name", required=True, type=str)
    parser.add_argument("--wandb", help="use wandb", action="store_true")
    parser.add_argument("--warmup", help="use warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)
    parser.add_argument("--lr", default=0.1, help="")
    parser.add_argument("--resume", default=None, help="")
    parser.add_argument("--batch_size", type=int, default=768, help="")
    parser.add_argument("--num_workers", type=int, default=4, help="")
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="gpu numbers", required=True)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:3456", type=str, help="")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="")
    parser.add_argument("--rank", default=0, type=int, help="")
    parser.add_argument("--world_size", default=1, type=int, help="")
    parser.add_argument("--distributed", action="store_true", help="")
    args = parser.parse_args()

    return args

args = parse_args()
gpus = ",".join([str(id) for id in args.gpus])
print("gpus: ", gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def main():
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    wdb = None
    if args.wandb and gpu == 0:
        import wandb
        wdb = wandb
        wdb.init(
            config=config,
            project="HPE_Validation_230915",
            name=f"{config.MODEL.NAME}_{config.MODEL.TYPE}_{config.DATASET.DATASET}_{config.LOSS.HM_LOSS}_{config.LOSS.UNC_LOSS}_{config.TRAIN.END_EPOCH}",
        )
        
    show_info(gpu, args, config)
    args.gpu = gpu

    # Load model param linked to the name of the config file
    if "small" in args.cfg:
        from lib.models.extra.vit_small_uncertainty_config import extra
    elif "large" in args.cfg:
        from lib.models.extra.vit_large_uncertainty_config import extra
    elif "huge" in args.cfg:
        from lib.models.extra.vit_huge_uncertainty_config import extra
    else:
        raise FileNotFoundError(f"Check config file name!!")

    ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(args.gpu))
    print("ngpus_per_ndoe : ", ngpus_per_node)
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    logger, final_output_dir = create_logger(config, args.cfg, f"train_{args.gpu}")
    if args.seed is not None:
        seed = init_random_seed(args.seed)
        logger.info(f"Set random seed to {seed}")
        set_random_seed(seed)

    ## Backbone - ViT
    backbone = ViT(
        img_size=extra["backbone"]["img_size"],
        patch_size=extra["backbone"]["patch_size"],
        embed_dim=extra["backbone"]["embed_dim"],
        in_channels=3,
        num_heads=extra["backbone"]["num_heads"],
        depth=extra["backbone"]["depth"],
        qkv_bias=True,
        drop_path_rate=extra["backbone"]["drop_path_rate"]
    )
 
    ## HEAD - Heatmap Simple Head
    deconv_head = TopdownHeatmapSimpleHead(
        in_channels=extra["keypoint_head"]["in_channels"],
        num_deconv_layers=extra["keypoint_head"]["num_deconv_layers"],
        num_deconv_filters=extra["keypoint_head"]["num_deconv_filters"],
        num_deconv_kernels=extra["keypoint_head"]["num_deconv_kernels"],
        extra=dict(final_conv_kernel=1),
        out_channels=17,
    )

    if "uncertainty" in args.cfg:
        from lib.models.heads import UncertaintyDeconvDepthWiseChannelHead
        from lib.models.uncertainty_pose import UncertaintyPose
        uncertainty_head = UncertaintyDeconvDepthWiseChannelHead(
            extra["uncertainty_channel_head"], norm_cfg=dict(type="BN")
        )
        model = UncertaintyPose(backbone, deconv_head, uncertainty_head, config)
    else:
        from lib.models.vit_pose import ViTPose
        model = ViTPose(backbone, deconv_head)

    checkpoint_path = os.path.join(home_dir, args.weight)

    if "mae" in checkpoint_path:
        # model = model.vit_mae_init(model, checkpoint_path)
        load_checkpoint(model.backbone, checkpoint_path)
    else:
        # model.init_weights(checkpoint_path)
        model = model.custom_init_weights(model, checkpoint_path)
        
    torch.cuda.set_device(args.gpu)
    config.gpu = gpu
    model.cuda(args.gpu)
    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)

    criterion = get_loss(config)
    optimizer = get_vit_optimizer(config, model, extra)    

    print(f"{sc.OKBLUE} Optimizer : {optimizer} {sc.ENDC}")
    print()

    lr_scheduler = MultistepWarmUpRestargets(
        optimizer, milestones=config.TRAIN.LR_STEP, gamma=config.TRAIN.LR_FACTOR
    )
    
    warmup_scheduler = None
    if args.warmup:
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: config.TRAIN.WARMUP_RATIO + (1.0 - config.TRAIN.WARMUP_RATIO) * step / config.TRAIN.WARMUP_ITERS
        )
        warmup_scheduler.count = 0 

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=train_sampler,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
    )

    best_perf = 0.0
    perf_indicator = 0.0
    best_model = False
    step_scale_count = 0 
    if config.MODEL.FREEZE_NAME:
        print("Freeze Group : ", config.MODEL.FREEZE_NAME)
    if config.MODEL.DIFF_NAME:
        print("Diff Group : ", config.MODEL.DIFF_NAME)
    

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        train(
            config,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            final_output_dir,
            wdb=wdb,
            warmup_scheduler = warmup_scheduler
        )

        if args.warmup:
            if epoch in lr_scheduler.milestones:
                step_scale_count += 1
                warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda step: (config.TRAIN.WARMUP_RATIO + (1.0 - config.TRAIN.WARMUP_RATIO) * step)*(0.1**step_scale_count) / config.TRAIN.WARMUP_ITERS
                )
                warmup_scheduler.count=0
        else:
            lr_ = lr_scheduler.get_lr()
            for i, g in enumerate(optimizer.param_groups):
                g["lr"] = lr_[i]

        # evaluate on validation set
        if epoch % config.EVALUATION.INTERVAL == 0:
            model_state_file = os.path.join(
                final_output_dir,
                f"{config.MODEL.NAME}_{config.LOSS.HM_LOSS}_{config.LOSS.UNC_LOSS}_{epoch}.pth.tar",
            )
            logger.info("saving final model state to {}".format(model_state_file))
            torch.save(model.module.state_dict(), model_state_file)

            perf_indicator = 0.
            perf_indicator = validate(
                config,
                valid_loader,
                valid_dataset,
                model,
                criterion,
                final_output_dir,
                backbone=backbone,
                keypoint_head=deconv_head,
                wdb=wdb,
            )

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if best_model:
            logger.info("=> saving checkpoint to {}".format(final_output_dir))
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": get_model_name(config),
                    "state_dict": model.state_dict(),
                    "perf": perf_indicator,
                    "optimizer": optimizer.state_dict(),
                    "HM_LOSS": config.LOSS.HM_LOSS,
                    "unc_loss": config.LOSS.UNC_LOSS,
                },
                best_model,
                final_output_dir,
            )

    final_model_state_file = os.path.join(final_output_dir, "final_state.pth.tar")
    logger.info("saving final model state to {}".format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
