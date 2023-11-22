# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import json_tricks as json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from lib.dataset.joints_dataset import JointsDataset
from lib.utils.nms import oks_nms


logger = logging.getLogger(__name__)


class COCODataset(JointsDataset):
    """
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
        "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    """

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = os.path.join(
            root, "person_detection_results/COCO_val2017_detections_AP_H_56_person.json"
        )
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.use_udp = cfg.TEST.USE_UDP
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.det_bbox_thr = 0.0
        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ["__background__"] + cats
        logger.info("=> classes: {}".format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]]
        )

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        self.num_joints = 17
        self.flip_pairs = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
        ]

        self.sigmas = [
            0.026,
            0.025,
            0.025,
            0.035,
            0.035,
            0.079,
            0.079,
            0.072,
            0.072,
            0.062,
            0.062,
            0.107,
            0.107,
            0.087,
            0.087,
            0.089,
            0.089,
        ]

        self.parent_ids = None
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.db = self._get_db()

        print(f"=> num_images: {self.num_images}")
        print(f"=> load {len(self.db)} samples")

    def _get_ann_file_keypoint(self):
        """self.root / annotations / person_keypoints_train2017.json"""
        prefix = "person_keypoints" if "test" not in self.image_set else "image_info"
        return os.path.join(self.root, "annotations", prefix + "_" + self.image_set + ".json")

    def _load_image_set_index(self):
        """image id: int"""
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ground truth bbox and keypoints"""
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        bbox_id = 0
        for obj in objs:
            x, y, w, h = obj["bbox"]
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj["category_id"]]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj["keypoints"]) == 0:
                continue

            joints_2d = np.zeros((self.num_joints, 2), dtype=np.float32)
            joints_2d_vis = np.zeros((self.num_joints, 2), dtype=np.float32)
            for ipt in range(self.num_joints):
                joints_2d[ipt, 0] = obj["keypoints"][ipt * 3 + 0]
                joints_2d[ipt, 1] = obj["keypoints"][ipt * 3 + 1]
                t_vis = obj["keypoints"][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_2d_vis[ipt, 0] = t_vis
                joints_2d_vis[ipt, 1] = t_vis

            # TopDownGetBboxCenterScale
            center, scale = self._box2cs(obj["clean_bbox"][:4])
            rec.append(
                {
                    "image": self.image_path_from_index(index),
                    "center": center,
                    "scale": scale,
                    "joints_2d": joints_2d,
                    "joints_2d_vis": joints_2d_vis,
                    "filename": "",
                    "bbox": obj["clean_bbox"],
                    "bbox_id": bbox_id,
                }
            )
            bbox_id = bbox_id + 1
        return rec

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image["file_name"]
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """example: images / train2017 / 000000119993.jpg"""
        file_name = "%012d.jpg" % index
        if "2014" in self.image_set:
            file_name = "COCO_%s_" % self.image_set + file_name
        prefix = "test2017" if "test" in self.image_set else self.image_set
        image_path = os.path.join(self.root, "images", prefix, file_name)

        return image_path

    def _load_coco_person_detection_results(self):
        all_boxes = None
        print(self.bbox_file)
        with open(self.bbox_file, "r") as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError("=> Load %s fail!" % self.bbox_file)

        print(f"=> Total boxes: {len(all_boxes)}")

        kpt_db = []
        bbox_id = 0
        for det_res in all_boxes:
            if det_res["category_id"] != 1:
                continue

            if det_res["image_id"] not in self.id2name:
                continue

            image_file = self.image_path_from_index(det_res["image_id"])
            box = det_res["bbox"]
            score = det_res["score"]

            if score < self.image_thre:
                continue

            center, scale = self._box2cs(box)
            joints_2d = np.zeros((self.num_joints, 2), dtype=np.float32)
            joints_2d_vis = np.ones((self.num_joints, 2), dtype=np.float32)

            kpt_db.append(
                {
                    "image": image_file,
                    "center": center,
                    "scale": scale,
                    "rotation": 0,
                    "bbox": box[:4],
                    "score": score,
                    "dataset": self.dataset_name,
                    "joints_2d": joints_2d,
                    "joints_2d_vis": joints_2d_vis,
                    "bbox_id": bbox_id,
                }
            )
            bbox_id = bbox_id + 1
        print(f"=> Total boxes after filter " f"low score@{self.det_bbox_thr}: {bbox_id}")
        return kpt_db

    # need double check this API and classes field
    def evaluate(self, preds, output_dir, all_boxes, img_path, bbox_ids):
        res_folder = os.path.join(output_dir, "results")
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(res_folder, "keypoints_%s_results.json" % self.image_set)

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append(
                {
                    "keypoints": kpt,  # preds[i]
                    "center": all_boxes[idx][0:2],  # boxes[i][0:2]
                    "scale": all_boxes[idx][2:4],  # boxes[i][2:4]
                    "area": all_boxes[idx][4],  # boxes[i][4]
                    "score": all_boxes[idx][5],  # boxes[i][5]
                    "image": int(img_path[idx][-16:-4]),  # image_id
                    "bbox_id": int(bbox_ids[idx]),  # bbox_ids[i]
                }
            )
        # image x person x (keypoints)
        kpts = defaultdict(list)

        for kpt in _kpts:
            kpts[kpt["image"]].append(kpt)

        # Changed #
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p["score"]
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p["keypoints"][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p["score"] = kpt_score * box_score

            nms = oks_nms
            keep = nms(img_kpts, 0.9, sigmas=self.sigmas)
            oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)
        if "test" not in self.image_set:
            info_str = self._do_python_keypoint_eval(res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value["AP"]
        else:
            return {"Null": 0}, 0

    def _sort_and_unique_bboxes(self, kpts, key="bbox_id"):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                "cat_id": self._class_to_coco_ind[cls],
                "cls_ind": cls_ind,
                "cls": cls,
                "ann_type": "keypoints",
                "keypoints": keypoints,
            }
            for cls_ind, cls in enumerate(self.classes)
            if not cls == "__background__"
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info("=> Writing results json to %s" % res_file)
        with open(res_file, "w") as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, "r") as f:
                for line in f:
                    content.append(line)
            content[-1] = "]"
            with open(res_file, "w") as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack["cat_id"]
        keypoints = data_pack["keypoints"]
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]["keypoints"] for k in range(len(img_kpts))])
            key_points = np.zeros((_key_points.shape[0], self.num_joints * 3), dtype=np.float32)

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    "image_id": img_kpts[k]["image"],
                    "category_id": cat_id,
                    "keypoints": list(key_points[k]),
                    "score": img_kpts[k]["score"],
                    "center": list(img_kpts[k]["center"]),
                    "scale": list(img_kpts[k]["scale"]),
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, "keypoints")
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = [
            "AP",
            "Ap .5",
            "AP .75",
            "AP (M)",
            "AP (L)",
            "AR",
            "AR .5",
            "AR .75",
            "AR (M)",
            "AR (L)",
        ]

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        eval_file = os.path.join(res_folder, "keypoints_%s_results.pkl" % self.image_set)

        with open(eval_file, "wb") as f:
            pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
        logger.info("=> coco eval results saved to %s" % eval_file)

        return info_str
