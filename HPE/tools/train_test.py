import sys, os
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

home_dir = os.path.dirname(os.path.abspath(__file__ + "/../"))
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)
from lib.utils.utils import init_random_seed, set_random_seed
seed = init_random_seed(0)
set_random_seed(seed)
print(torch.randn(4))
from lib.models.backbones.vit import ViT
# from lib.models.backbones.vit_custom import ViT
from lib.models.heads import TopdownHeatmapSimpleHead
from lib.dataset.coco import COCODataset
from lib.core.config import config
from lib.core.config import update_config


cfg = home_dir + '/experiments/coco/vit/vit_large.yaml'
update_config(cfg)
config.DATASET.TARGET_HEATMAP = False
config.MODEL.EXTRA

train_dataset = COCODataset(
    cfg=config, 
    root=home_dir + '/data/coco/', 
    image_set='train2017', 
    is_train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
)

test_dataset = COCODataset(
    cfg=config, 
    root=home_dir + '/data/coco/', 
    image_set='val2017', 
    is_train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=config.WORKERS,
    pin_memory=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config.WORKERS,
    pin_memory=True,
)

if "small" in cfg:
    from lib.models.extra.vit_small_uncertainty_config import extra
elif "large" in cfg:
    from lib.models.extra.vit_large_uncertainty_config import extra
elif "huge" in cfg:
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
    drop_path_rate=extra["backbone"]["drop_path_rate"]
)

deconv_head = TopdownHeatmapSimpleHead(
    in_channels=extra["keypoint_head"]["in_channels"],
    num_deconv_layers=extra["keypoint_head"]["num_deconv_layers"],
    num_deconv_filters=extra["keypoint_head"]["num_deconv_filters"],
    num_deconv_kernels=extra["keypoint_head"]["num_deconv_kernels"],
    extra=dict(final_conv_kernel=1),
    out_channels=17,
)

if "uncertainty" in cfg:
    from lib.models.heads import UncertaintyDeconvDepthWiseChannelHead
    from lib.models.uncertainty_pose import UncertaintyPose
    uncertainty_head = UncertaintyDeconvDepthWiseChannelHead(
        extra["uncertainty_channel_head"], norm_cfg=dict(type="BN")
    )
    model = UncertaintyPose(backbone, deconv_head, uncertainty_head, config)
else:
    from lib.models.vit_pose import ViTPose
    model = ViTPose(backbone, deconv_head)


checkpoint_path = os.path.join(home_dir, 'checkpoints/mae_pretrain_vit_large.pth')
# checkpoint_path = os.path.join(home_dir, 'checkpoints/vitpose_large.pth')

if "mae" in checkpoint_path:
    model = model.vit_mae_init(model, checkpoint_path)
else:
    # model.init_weights(checkpoint_path)
    model = model.custom_init_weights(model, checkpoint_path)
backbone = model.backbone


for idx, data in enumerate(test_loader):
    if idx == 1:
        break
    img = data[0]
    # print(img)
    test = backbone(img)
    print(test)