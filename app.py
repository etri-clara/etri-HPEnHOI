import sys, os
import os
import torch
import torchvision.transforms as transforms

home_dir = os.path.abspath(os.getcwd()+"/HPE")
sys.path.append(os.path.join(home_dir, 'lib'))
sys.path.append(home_dir)

from lib.models.backbones.vit import ViT
from lib.models.heads import UncertaintyDeconvDepthWiseChannelHead
from lib.models.heads import TopdownHeatmapSimpleHead
from lib.dataset.coco import COCODataset
from lib.core.config import config
from lib.core.config import update_config
# from lib.models.extra.vit_large_uncertainty_config import extra
from lib.models.extra.vit_huge_uncertainty_config import extra

from lib.models.uncertainty_pose import UncertaintyPose
from lib.models.vit_pose import ViTPose

from lib.utils.vis import display_heatamp, display_keypoints, display_keypoints_with_uncertainty, coco_info, skeleton_connection_info
import cv2
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from PIL import Image


import gradio as gr
import os

### HPE #######################################################################
# Load config info
update_config(home_dir + '/experiments/coco/vit/vit_huge_uncertainty.yaml')

config.DATASET.TARGET_KEYPOINT = True
config.DATASET.TARGET_HEATMAP = True

config.TEST.USE_GT_BBOX = True

# Load model
from copy import deepcopy
## Backbone - ViT
backbone = ViT(
    img_size=extra["backbone"]["img_size"],
    patch_size=extra["backbone"]["patch_size"],
    embed_dim=extra["backbone"]["embed_dim"],
    in_channels=3,
    num_heads=extra["backbone"]["num_heads"],
    depth=extra["backbone"]["depth"],
    qkv_bias=True,
    drop_path_rate = extra["backbone"]["drop_path_rate"]
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

## Uncertainty head
uncertainty_head = UncertaintyDeconvDepthWiseChannelHead(
    extra["uncertainty_channel_head"], norm_cfg=dict(type="BN")
)
hpe_model = UncertaintyPose(backbone, deconv_head, uncertainty_head, config)
hpe_model_official = ViTPose(deepcopy(backbone), deepcopy(deconv_head))

hpe_model.eval()
hpe_model_official.eval()


def get_max_preds(heatmaps):
    """
    get predictions from score maps+
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
    assert heatmaps.ndim == 4, "batch_images should be 4-ndim"
    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


# Load weight
checkpoint_path = home_dir + "/checkpoints/vit_huge_77_1.pth.tar" #"/checkpoints/vit_huge_uncertainty_77_1_AP.pth.tar"
checkpoint_official = home_dir + "/checkpoints/vitpose-h.pth" #checkpoints/vitpose_huge.pth"

hpe_model = hpe_model.custom_init_weights(hpe_model, checkpoint_path)
hpe_model_official.init_weights(checkpoint_official)



###############################################################################

### HOI #######################################################################
import torch, detectron2
import os
import sys
import logging
import argparse
os.environ["DATASET"] = "../datasets"

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from pprint import pprint
import numpy as np
np.random.seed(1)

#home_dir = os.path.abspath(os.getcwd()+"/../")
home_dir = os.path.abspath(os.getcwd()+"/HOI")
sys.path.append(home_dir)
print(home_dir)

import warnings
warnings.filterwarnings(action='ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from torchvision import transforms
from PIL import Image

from hdecoder.BaseModel import BaseModel
from hdecoder import build_model
from hdecoder.utils.distributed import init_distributed
from hdecoder.utils.arguments import load_opt_from_config_files, load_config_dict_to_opt
from datasets.utils.vcoco_utils import get_random_images, walk_through_dir
from hdecoder.utils.visualizer import draw_hoi_results, draw_obj_attentions, draw_hoi_attention

from hdecoder.utils.arguments import load_vcoco_opt_command, load_vcoco_parser


from lib.utils.vis import display_heatamp, display_keypoints, display_keypoints_with_uncertainty, coco_info, skeleton_connection_info


cmdline_args = load_vcoco_parser()
cmdline_args.conf_files = [os.path.join(home_dir, "configs/hdecoder/vcoco_large.yaml")]
model_path = home_dir + '/checkpoints/vcoco_hdecoder_l.pt'

cmdline_args.overrides = ['WEIGHT', 'true', 'RESUME_FROM', model_path]

opt = load_vcoco_opt_command(cmdline_args)
opt = init_distributed(opt)
pretrained_pth = os.path.join(opt['RESUME_FROM'])

hoi_model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

###############################################################################

#title_markdown = ("""
# 🌋 Human Pose Estimation & Human Object Interaction & Scene Graph Generation Demo
#""")
title_markdown = ("""
# 🌋 Human Pose Estimation & Human Object Interaction Demo
""")

#  B, C, H, W = x.shape
# torch.Size([1, 3, 1750, 1914])
#torch.Size([1, 3, 192, 256])


root_dir = "/home/yongju/Github/HPE_HOI"

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

from lib.utils.transforms import get_affine_transform
from lib.utils.transforms import affine_transform
from pycocotools.coco import COCO

annFile='{}/annotations/person_keypoints_{}.json'.format("/data/coco", "val2017")
coco = COCO(annFile)

IMAGE_SIZE = [ 256, 192 ]

def box2cs(box):
    x, y, w, h = box[:4]
    return xywh2cs(x, y, w, h)


#  B, C, H, W = x.shape
# torch.Size([1, 3, 1750, 1914])
#torch.Size([1, 3, 192, 256])


root_dir = "/home/yongju/Github/HPE_HOI"

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

from lib.utils.transforms import get_affine_transform
from lib.utils.transforms import affine_transform
from pycocotools.coco import COCO

annFile='{}/annotations/person_keypoints_{}.json'.format("/data/coco", "val2017")
coco = COCO(annFile)

IMAGE_SIZE = [ 256, 192 ]

def box2cs(box):
    x, y, w, h = box[:4]
    return xywh2cs(x, y, w, h)

def xywh2cs(x, y, w, h):
    #aspect_ratio = config.MODEL.IMAGE_SIZE[0] * 1.0 / config.MODEL.IMAGE_SIZE[1]
    aspect_ratio = IMAGE_SIZE[0] * 1.0 / IMAGE_SIZE[1]
    pixel_std = 200

    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

import matplotlib.patches as patches


def plot_skeleton(original_img, predict_keypoints, uncertainty,bbox,  idx=0):

    fig,ax1  = plt.subplots() 

    uncertainty_thr = 1.5

    ## 핵심 
    center, scale = box2cs([bbox[idx + i*len(predict_keypoints)] for i in range(4)])
    trans = get_affine_transform(center=center, scale=scale, rot=0, output_size=config.MODEL.IMAGE_SIZE, inv=1)
    ## 핵심  

    preds_huge_copy = deepcopy(predict_keypoints)

    # joint 17개 
    for i in range(17):
        preds_huge_copy[idx][i, 0:2] = affine_transform(predict_keypoints[idx][i, 0:2] * 4, trans)    
        
    pred_kp = preds_huge_copy[idx] 
    # print(pred_kp)
    uncertainty = uncertainty[idx]
    x = pred_kp[:,0]
    y = pred_kp[:,1]
    for i, sk in enumerate(skeleton_connection_info['SKELETON']):
        ax1.plot(x[sk], y[sk], linewidth=2, color=skeleton_connection_info['COLOR'][i])

    ax1.plot(x[uncertainty>uncertainty_thr], y[uncertainty>uncertainty_thr],'o',markersize=6, markerfacecolor=[1,0,0], markeredgecolor='k',markeredgewidth=2)
    ax1.plot(x[uncertainty<uncertainty_thr], y[uncertainty<uncertainty_thr],'o',markersize=4, markerfacecolor=[1,1,1], markeredgecolor=[0,0,0], markeredgewidth=2)

    ax1.imshow(original_img)
    plt.axis('off')

    fig.tight_layout()

    plt.savefig("./tmp1.png", dpi=200)
    fig1 = Image.open("./tmp1.png")

    return fig1


def infer_hpe(image,coco_id):

    img_origin = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    #####

    transform = transforms.Compose([ transforms.Resize((256,192)),
                                transforms.ToTensor() ] )
    img_tensor = transform(image.convert('RGB'))
    img_tensor= torch.unsqueeze(img_tensor,0)

    cat_id = coco.getCatIds(catNms=['person'])
    ann_id = coco.getAnnIds(imgIds=[int(coco_id)], catIds=cat_id, iscrowd=None)
    anns = coco.loadAnns(ids=ann_id[0])

    anns_bbox = [ int(x) for x in anns[0]['bbox'] ]
    box = list(np.array([i for i in anns_bbox]).reshape(-1))

    idx = 0 
    center, scale = box2cs([box[idx + i*len(img_tensor)] for i in range(4)])
    trans = get_affine_transform(center=center, scale=scale, rot=0, output_size=config.MODEL.IMAGE_SIZE, inv=0)


    input_img = cv2.warpAffine(
                img_origin,
                trans,
                (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
                flags=cv2.INTER_LINEAR,
            )

    transform_tensor = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform_tensor(input_img)
    img_tensor = img_tensor.view(1,3,256,192)


    np.set_printoptions()  # 소수점 이하 자릿수 0, 지수 표현 비활성화

    with torch.no_grad():
        # pose_output = model.forward(img.cuda())
        pose_output = hpe_model.forward(img_tensor)
        # pose_output_official = model_official.forward(img)

        #g_t_preds, _1 = hpe_model.get_max_preds(heatmap.detach().cpu().numpy())

    keys_huge, uncertainty_huge, _, hm_huge = pose_output

    preds_huge, value_huge = get_max_preds(hm_huge.detach().cpu().numpy())

    # preds_huge_official, _ = get_max_preds(pose_output_official.detach().cpu().numpy())


    soft_plus = torch.nn.Softplus()

    uncertainty_map_huge = soft_plus(uncertainty_huge)

    if config.LOSS.USE_INDEXING:
        # kp_ = np.round(target.detach().cpu().numpy() / 4)
        # kp_ = np.round(keys_huge.detach().cpu().numpy())
        kp_ = np.round(preds_huge)

        x = np.clip(kp_[:, :, 0], 0, config.MODEL.HEATMAP_SIZE[0] - 1)
        y = np.clip(kp_[:, :, 1], 0, config.MODEL.HEATMAP_SIZE[1] - 1)

        # Uncertainty Map has 1 channel
        sigma_x = torch.diagonal(uncertainty_map_huge[:, 0, y, x], dim1=0, dim2=1).permute(
            1, 0
        )

        uncertainty_huge = torch.cat([sigma_x.unsqueeze(-1), sigma_x.unsqueeze(-1)], dim=-1)

    # get uncertainty 
    sigma_x = sigma_x.detach().cpu().numpy()

    image = plot_skeleton(img_origin, preds_huge, sigma_x, box,  0)

    return image


###############################################################################
from datasets.utils.vcoco_utils import valid_obj_ids, verb_classes, coco_class_list

def draw_hoi_result(image, hoi_result, title="Predicted_Result", is_save=False, save_dir_name="output"):
    #plt.figure(figsize=(8,6))
    #plt.title(title, fontsize = 40)
    #plt.imshow(image)

    fig,ax  = plt.subplots() 

    #ax = plt.gca()
    COLORS = np.random.uniform(0, 255, size=(len(valid_obj_ids)+1, 3))
    hoi_result = hoi_result[0]

    if hoi_result:
        print(f"Detect {len(hoi_result)} HOI!!, {hoi_result}")

    
    for color_id, r in enumerate(hoi_result):

        # print(f"Query: {r['object_id']-100}")
        category_id_verb = r['category_id']
        object_bbox = r['object_bbox']['bbox']
        object_id = r['object_bbox']['category_id']
        subject_id = 0
        score = r['score']
        subject_bbox = r['subject_id']['bbox']

        color = COLORS[color_id]

        center_coord_x = int((object_bbox[0] + object_bbox[2]) / 2)
        center_coord_y = int(object_bbox[1]) + 20

        sub_center_coord_x = int((subject_bbox[0] + subject_bbox[2]) / 2)
        sub_center_coord_y = int(subject_bbox[1]) + 20

        sub_xmin, sub_ymin, sub_xmax, sub_ymax = subject_bbox[0], subject_bbox[1], subject_bbox[2], subject_bbox[3]
        obj_xmin, obj_ymin, obj_xmax, obj_ymax = object_bbox[0], object_bbox[1], object_bbox[2], object_bbox[3]
        score *= 100

        result_text = f"<{coco_class_list[subject_id]}, {verb_classes[category_id_verb]}, {coco_class_list[object_id]}> ({int(score)}%)"

        if coco_class_list[object_id] is None:
           object_id = subject_id
           result_text = f"<{coco_class_list[object_id]}, {verb_classes[category_id_verb]}> ({int(score)}%)"

           ax.add_patch(plt.Rectangle(
                    (sub_xmin, sub_ymin), sub_xmax - sub_xmin, sub_ymax - sub_ymin,
                                    fill=False, color=color/256, linewidth=3))

           ax.text(sub_center_coord_x, sub_center_coord_y, result_text, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.9))

        else:
           ax.add_patch(plt.Rectangle(
                    (sub_xmin, sub_ymin), sub_xmax - sub_xmin, sub_ymax - sub_ymin,
                                    fill=False, color=color/256, linewidth=3))
           ax.add_patch(plt.Rectangle(
                    (obj_xmin, obj_ymin), obj_xmax - obj_xmin, obj_ymax - obj_ymin,
                                    fill=False, color=color/256, linewidth=3))

           ax.text(center_coord_x, center_coord_y, result_text, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.9))
    ax.imshow(image)
    plt.axis('off')

    fig.tight_layout()

    plt.savefig("./tmp2.png", dpi=200)
    fig1 = Image.open("./tmp2.png")

    return fig1

def infer_hoi(image):

    t = []
    t.append(transforms.Resize(800, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    results = []
    org_images = []

    image_ori = image.convert('RGB')  #Image.open(image_pth).convert('RGB')

    width = image_ori.size[0]
    height = image_ori.size[1]
    orig_size = (height, width)

    print("infer_hoi", width, height)

    result = hoi_model.model.hoi_inference(image_ori, orig_size, transform, thr=0.2)

    image = draw_hoi_result(image_ori, result, is_save=False)

    return image
###############################################################################

def infer_sgg(image):
    return image
###############################################################################

css = ".output-image, .input-image, .image-preview {height: 400px !important} "

with gr.Blocks(theme=gr.themes.Soft(), css=css ) as demo:

    gr.Markdown(title_markdown)

    with gr.Row(equal_height=True):
        with gr.Column(variant='pannel', scale=1, min_width=300):
            input_image = gr.Image(type="pil",height=300, interactive=False, label="Input Image")
            input_textbox = gr.Textbox(visible=False)
    with gr.Row(equal_height=True):
        with gr.Column(variant='pannel', scale=1, min_width=300):
            hpe_output_image1 = gr.Image(type="pil",height=400, interactive=False, label="HPE Output")
        with gr.Column(variant='pannel', scale=1, min_width=300):
            hoi_output_image1 = gr.Image(type="pil",height=400, interactive=False, label="HOI Output")

#        with gr.Column(variant='pannel', scale=1, min_width=300):
#            sgg_output_image1 = gr.Image(type="pil",height=400, interactive=False, label="SGG Output")

    with gr.Row(equal_height=True):
        with gr.Column(variant='pannle', scale=1):
            hpe_btn = gr.Button(value="HPE Submit")
        with gr.Column(variant='pannle', scale=1):
            hoi_btn = gr.Button(value="HOI Submit")
#        with gr.Column(variant='pannle', scale=1):
#            sgg_btn = gr.Button(value="SGG Submit")


    with gr.Row():
        gr.Examples(
        examples=[ 
                    ["/data/coco/val2017/000000066886.jpg", 66886],
                    ["/data/coco/val2017/000000079031.jpg", 79031],
                    ["/data/coco/val2017/000000579070.jpg", 579070],
                    ["/data/coco/val2017/000000458992.jpg", 458992],
                    ["/data/coco/val2017/000000004134.jpg", 4134],
                    ["/data/coco/val2017/000000008690.jpg", 8690],
                    ["/data/coco/val2017/000000128748.jpg", 128748],
                    ["/data/coco/val2017/000000140270.jpg", 140270],
                    ["/data/coco/val2017/000000496854.jpg", 496854],
                    ["/data/coco/val2017/000000513524.jpg", 513524],
                    ["/data/coco/val2017/000000578792.jpg", 578792],
                    ["/data/coco/val2017/000000581357.jpg", 581357],
                    ["/data/coco/val2017/000000559348.jpg", 59348],
                    ["/data/coco/val2017/000000521259.jpg", 521259],
                    ["/data/coco/val2017/000000514797.jpg", 514797]
                 ],
        inputs=[input_image,input_textbox],
        fn=[infer_hpe,infer_hoi, infer_sgg]
        )

    hpe_btn.click(infer_hpe, inputs=[input_image,input_textbox], outputs=[hpe_output_image1]) #,hpe_output_image2]) #, hpe_output_image3])
    hoi_btn.click(infer_hoi, inputs=[input_image], outputs=[hoi_output_image1])
#    sgg_btn.click(infer_sgg, inputs=[input_image], outputs=[sgg_output_image1])


if __name__ == "__main__":
    demo.launch(server_name="129.254.186.123")
