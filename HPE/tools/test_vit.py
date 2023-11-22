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

from lib.models.backbones.vit import ViT
from lib.models.heads import TopdownHeatmapSimpleHead
from lib.dataset.coco import COCODataset

from lib.core.config import config
from lib.core.config import update_config
from lib.core.function import validate 

from lib.utils.utils import get_loss
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
    print(f"{sc.COLOR_GREEN}TEST BATCH SIZE: {sc.ENDC}{config.TEST.BATCH_SIZE}")
    print(f"{sc.COLOR_CYAN}USE UDP: {sc.ENDC}{config.TEST.USE_UDP}")
    print(f"{sc.COLOR_CYAN}USE FLIP: {sc.ENDC}{config.TEST.FLIP_TEST}")
    print(f"{sc.COLOR_CYAN}USE GT BBOX: {sc.ENDC}{config.TEST.USE_GT_BBOX}")
    print(f"{sc.COLOR_CYAN}USE UNCERTAINTY: {sc.ENDC}{config.LOSS.UNCERTAINTY}")
    print(f"{sc.COLOR_CYAN}USE WANDB: {sc.ENDC}{args.wandb}")
    print(f"{'='*46}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="")
    parser.add_argument("--weight", help="checkpoint name", required=True, type=str)
    parser.add_argument("--wandb", help="use wandb", action="store_true")
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
        batch_size=config.TEST.BATCH_SIZE,
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
        backbone=backbone,
        keypoint_head=deconv_head,
        wdb=wdb
    )



if __name__ == "__main__":
    main()
