# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch

from lib.core.config import get_model_name
from lib.core.evaluate import accuracy
from lib.core.inference import get_final_preds
from lib.utils.transforms import flip_back
from lib.utils.vis import save_debug_images
from lib.core.inference import post_dark_udp
from lib.core.inference import get_max_preds
from lib.utils.utils import get_grad_norm

from torch.nn.utils import clip_grad
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


def train(
    config, train_loader, model, criterion, optimizer, epoch, output_dir, device=None, wdb=None, warmup_scheduler=None,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    # freeze할 module이 있으면 그 내부에 batch norm을 freeze 시킴! can set in yaml
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            result = [True if i in name else False for i in config.MODEL.FREEZE_NAME]
            if any(result):
                # print(name)
                module.eval()

    end = time.time()
    soft_plus = torch.nn.Softplus()

    end = time.time()

    # AMP setting
    if config.MODEL.USE_AMP:
        torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True)

        logger.info("Using Automatic Mixed Precision (AMP) training...")
        # Create a GradScaler object for FP16 training
        scaler = GradScaler()

    # for i, (input, target, target_weight, meta) in enumerate(train_loader):
    for i, (input, target, target_weight, heatmap, heatmap_target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        heatmap = heatmap.cuda(non_blocking=True, device=device)
        heatmap_target = heatmap_target.cuda(non_blocking=True, device=device)

        # compute output
        if config.MODEL.USE_AMP:
            with autocast():
                output = model(input)

                if config.LOSS.UNCERTAINTY:
                    pred_keys, uncertainty, unc_norm_heatmap, output = output

                    uncertainty_map = soft_plus(uncertainty)
                    # Get uncertainty in Uncertainty Map
                    if uncertainty.shape[-2:] == output.shape[-2:]:
                        if config.LOSS.USE_INDEXING:
                            # kp_ = np.round(target.detach().cpu().numpy() / 4)
                            # keypoint max version
                            if not config.MODEL.USE_EXP_KP:
                                kp_, _ = get_max_preds(output.detach().cpu().numpy())
                                kp_ = post_dark_udp(kp_, output.detach().cpu().numpy(), kernel=11)
                                pred_keys = torch.from_numpy(kp_).cuda()
                                kp_ = np.round(kp_)
                            else:
                                kp_ = np.round(pred_keys.detach().cpu().numpy())

                            x = np.clip(kp_[:, :, 0], 0, config.MODEL.HEATMAP_SIZE[0] - 1)
                            y = np.clip(kp_[:, :, 1], 0, config.MODEL.HEATMAP_SIZE[1] - 1)

                            # Uncertainty Map has 1 channel
                            sigma_x = torch.diagonal(
                                uncertainty_map[:, 0, y, x], dim1=0, dim2=1
                            ).permute(1, 0)

                            uncertainty = torch.cat(
                                [sigma_x.unsqueeze(-1), sigma_x.unsqueeze(-1)], dim=-1
                            )

                    if config.LOSS.HM_LOSS != "":
                        loss = cal_loss(
                            config,
                            criterion,
                            output,
                            heatmap,
                            heatmap_target,
                            pred_keys,
                            target,
                            target_weight,
                            uncertainty,
                            count=i,
                            wdb=wdb,
                        )
                    else:
                        loss = cal_loss(
                            config,
                            criterion,
                            0,
                            0,
                            heatmap_target,
                            pred_keys,
                            target,
                            target_weight,
                            uncertainty,
                            count=i,
                            wdb=wdb,
                        )
                else:
                    loss = cal_loss(config, criterion, output, heatmap, heatmap_target, i, wdb)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = clip_grads(model.parameters())

            if warmup_scheduler.count < config.TRAIN.WARMUP_ITERS:
                # warmup_scheduler.step()
                warmup_scheduler.count += 1
                scaler.step(warmup_scheduler)
            else:
                scaler.step(optimizer)
            
            scaler.update()
        else:
            output = model(input)

            if config.LOSS.UNCERTAINTY:
                pred_keys, uncertainty, unc_norm_heatmap, output = output

                uncertainty_map = soft_plus(uncertainty)
                # Get uncertainty in uncertainty_map
                if uncertainty.shape[-2:] == output.shape[-2:]:
                    if config.LOSS.USE_INDEXING:
                        # kp_ = np.round(target.detach().cpu().numpy() / 4)
                        # keypoint max version
                        if not config.MODEL.USE_EXP_KP:
                            kp_, _ = get_max_preds(output.detach().cpu().numpy())
                            kp_ = post_dark_udp(kp_, output.detach().cpu().numpy(), kernel=11)
                            pred_keys = torch.from_numpy(kp_).cuda()
                            kp_ = np.round(kp_)
                        else:
                            kp_ = np.round(pred_keys.detach().cpu().numpy())

                        x = np.clip(kp_[:, :, 0], 0, config.MODEL.HEATMAP_SIZE[0] - 1)
                        y = np.clip(kp_[:, :, 1], 0, config.MODEL.HEATMAP_SIZE[1] - 1)

                        # Uncertainty Map has 1 channel
                        sigma_x = torch.diagonal(
                            uncertainty_map[:, 0, y, x], dim1=0, dim2=1
                        ).permute(1, 0)

                        uncertainty = torch.cat(
                            [sigma_x.unsqueeze(-1), sigma_x.unsqueeze(-1)], dim=-1
                        )

                if config.LOSS.HM_LOSS != "":
                    loss = cal_loss(
                        config,
                        criterion,
                        output,
                        heatmap,
                        heatmap_target,
                        pred_keys,
                        target,
                        target_weight,
                        uncertainty,
                        count=i,
                        wdb=wdb,
                    )
                else:
                    loss = cal_loss(
                        config,
                        criterion,
                        0,
                        0,
                        heatmap_target,
                        pred_keys,
                        target,
                        target_weight,
                        uncertainty,
                        count=i,
                        wdb=wdb,
                    )
            else:
                loss = cal_loss(config, criterion, output, heatmap, heatmap_target, i, wdb)
            # print("grad_norm : ",get_grad_norm(model, loss))

            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grads(model.parameters())
            if warmup_scheduler.count < config.TRAIN.WARMUP_ITERS:
                warmup_scheduler.step()
                warmup_scheduler.count += 1
            else:
                optimizer.step()
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(
            output.detach().cpu().numpy(), heatmap.detach().cpu().numpy()
        )
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                "Speed {speed:.1f} samples/s\t"
                "Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "Grad_Norm {grad_norm:.5f}"
                "Accuracy {acc.val:.3f} ({acc.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    speed=input.size(0) / batch_time.val,
                    data_time=data_time,
                    loss=losses,
                    grad_norm=grad_norm,
                    acc=acc,
                )
            )
            print("lr : ",warmup_scheduler.get_lr()[-1])

            logger.info(msg)
            if config.LOSS.UNCERTAINTY:
                print("mean uncertainty : ", torch.mean(uncertainty))

            prefix = "{}_{}".format(os.path.join(output_dir, "train"), i)
            save_debug_images(config, input, meta, target, pred * 4, output, prefix)
    if wdb:
        wdb.log({"train Speed": input.size(0) / batch_time.val})
        wdb.log({"loss": losses.avg})
        wdb.log({"acc": acc.avg})
        if config.LOSS.UNCERTAINTY:
            if epoch % 2 == 0:
                wdb.log({"uncertainty_map": to_wandb_img(wdb, uncertainty_map, True)})
                wdb.log({"Img": to_wandb_img(wdb, input.permute(0, 2, 3, 1), True)})
    

def validate(
    config,
    val_loader,
    val_dataset,
    model,
    criterion,
    output_dir,
    backbone=None,
    keypoint_head=None,
    device=None,
    wdb=None,
):
    if wdb == None:
        return 0

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_preds_exp = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    bbox_ids = []
    idx = 0
    soft_plus = torch.nn.Softplus()

    with torch.no_grad():
        end = time.time()
        # for i, (input, target, target_weight, meta) in enumerate(val_loader):
        for i, (input, target, target_weight, heatmap, heatmap_target, meta) in enumerate(
            val_loader
        ):
            output_heatmap = model.forward(input)
            if config.LOSS.UNCERTAINTY:
                pred_keys, uncertainty, output_norm_heatmap, output_heatmap = output_heatmap

            uncertainty_map = soft_plus(uncertainty)

            if uncertainty.shape[-2:] == output_heatmap.shape[-2:]:
                if config.LOSS.USE_INDEXING:
                    # kp_ = np.round(target.detach().cpu().numpy() / 4)
                    # keypoint max version
                    if not config.MODEL.USE_EXP_KP:
                        kp_, _ = get_max_preds(output_heatmap.detach().cpu().numpy())
                        kp_ = post_dark_udp(kp_, output_heatmap.detach().cpu().numpy(), kernel=11)
                        pred_keys = torch.from_numpy(kp_).cuda()
                        kp_ = np.round(kp_)
                    else:
                        kp_ = np.round(pred_keys.detach().cpu().numpy())

                    x = np.clip(kp_[:, :, 0], 0, config.MODEL.HEATMAP_SIZE[0] - 1)
                    y = np.clip(kp_[:, :, 1], 0, config.MODEL.HEATMAP_SIZE[1] - 1)

                    # Uncertainty Map has 1 channel
                    sigma_x = torch.diagonal(uncertainty_map[:, 0, y, x], dim1=0, dim2=1).permute(
                        1, 0
                    )

                    uncertainty = torch.cat([sigma_x.unsqueeze(-1), sigma_x.unsqueeze(-1)], dim=-1)
            # Changed 
            if config.TEST.FLIP_TEST:
                img_flipped = input.flip(3).cuda()
                features_flipped = backbone(img_flipped)
                output_flipped_heatmap = keypoint_head.inference_model(
                    features_flipped, meta["flip_pairs"]
                )
                output_heatmap = (
                    output_heatmap + torch.from_numpy(output_flipped_heatmap.copy()).cuda()
                ) * 0.5

            heatmap = heatmap.cuda(non_blocking=True)
            heatmap_target = heatmap_target.cuda(non_blocking=True)

            num_images = input.size(0)

            if not config.TEST.USE_GT_BBOX:
                loss = 0.0
            else:
                # measure record loss
                loss = criterion(output_heatmap, heatmap, heatmap_target)
                losses.update(loss.item(), num_images)

            # measure accuracy
            _, avg_acc, cnt, pred = accuracy(output_heatmap.cpu().numpy(), heatmap.cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta["center"].numpy()
            s = meta["scale"].numpy()
            score = meta["score"].numpy()

            # Can add adaptive kp prediction /
            preds, maxvals = get_final_preds(config, output_heatmap.clone().cpu().numpy(), c, s, 11)

            # Hybrid maxvals revise
            if config.TEST.HYBRID_TEST:
                maxvals = (
                    torch.clip(
                        (maxvals[:, :, 0] / (uncertainty[:, :, 0].detach().cpu() / 1.5)), 0, 1
                    ).view(-1, 17, 1)
                ).numpy()

            all_preds[idx : idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx : idx + num_images, :, 2:3] = maxvals
            # if config.LOSS.UNCERTAINTY:
            #     # Exp_kp preds
            #     preds_exp, maxvals = get_final_preds(
            #         config,
            #         output_heatmap.clone().cpu().numpy(),
            #         c,
            #         s,
            #         11,
            #         exp_kp=pred_keys.detach().cpu().numpy(),
            #     )

            #     all_preds_exp[idx : idx + num_images, :, 0:2] = preds_exp[:, :, 0:2]
            #     all_preds_exp[idx : idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx : idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx : idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx : idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx : idx + num_images, 5] = score
            image_path.extend(meta["image"])
            bbox_ids.extend(meta["bbox_id"])
            if config.DATASET.DATASET == "posetrack":
                filenames.extend(meta["filename"])
                imgnums.extend(meta["imgnum"].numpy())

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = (
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {acc.val:.3f} ({acc.avg:.3f})".format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc
                    )
                )
                logger.info(msg)

                prefix = "{}_{}".format(os.path.join(output_dir, "val"), i)
                save_debug_images(config, input, meta, heatmap, pred * 4, output_heatmap, prefix)

        perf_indicator = 0
        if config.gpu == 0:
            name_values, perf_indicator = val_dataset.evaluate(
                all_preds, output_dir, all_boxes, image_path, bbox_ids,
            )
            # if config.LOSS.UNCERTAINTY:
            #     name_values_exp, perf_indicator_exp = val_dataset.evaluate(
            #         all_preds_exp,
            #         output_dir,
            #         all_boxes,
            #         image_path,
            #         bbox_ids,
            #     )

            if wdb:
                wdb.log({"performance (AP)": perf_indicator})
                wdb.log({"loss_valid": losses.avg})
                wdb.log({"acc_valid": acc.avg})
                # if config.LOSS.UNCERTAINTY:
                #     wdb.log({"performance usuing exp_kp (AP)": perf_indicator_exp})

            _, full_arch_name = get_model_name(config)
            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(name_value, full_arch_name)
            else:
                _print_name_value(name_values, full_arch_name)

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info("| Arch " + " ".join(["| {}".format(name) for name in names]) + " |")
    logger.info("|---" * (num_values + 1) + "|")
    logger.info(
        "| "
        + full_arch_name
        + " "
        + " ".join(["| {:.3f}".format(value) for value in values])
        + " |"
    )


def cal_loss(
    config,
    criterion: dict,
    pred_heatmap,
    gt_heatmap,
    hm_weight,
    pred_key=None,
    target_keys=None,
    target_keypoints_weight=None,
    uncertainty=None,
    count=None,
    wdb=None,
):
    loss = 0
    for k, v in criterion.items():
        if k == "heatmap":
            l = v(pred_heatmap, gt_heatmap, hm_weight, count, wdb)
            loss += l * config.LOSS.HM_LOSS_WEIGHT
        if k == "keypoint":
            l = v(pred_key, target_keys / 4, target_keypoints_weight, uncertainty, count, wdb)
            loss += l * config.LOSS.KP_LOSS_WEIGHT
    return loss


def clip_grads(params):
    grad_clip = {"max_norm": 1., "norm_type": 2}
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        # print(" : ",)
        return clip_grad.clip_grad_norm_(params, **grad_clip)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


# img is sent to wandb
def to_wandb_img(wdb, data, clamp=False):
    data = data[:8]
    if clamp:
        data = data.clamp(min=0, max=1)
    return [wdb.Image(img) for img in np.split(data.detach().cpu().numpy(), data.size(0))]
