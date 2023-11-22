# HPE

## Overview 
![image](https://github.com/thinkaicho/HPE/assets/75425941/17db2113-fc0b-4541-a488-c3e2a101fefe)

## Installation
```
conda create -n vitpose_uncertainty python=3.9 -y
conda activate vitpose_uncertainty

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install -r requirements.txt
```

## File tree
```
HPE 
  .
  ├── checkpoints
  ├── data
      └── coco
          ├── annotations
          │   ├── captions_train2017.json
          │   ├── captions_val2017.json
          │   ├── instances_train2017.json
          │   ├── instances_val2017.json
          │   ├── person_keypoints_train2017.json
          │   └── person_keypoints_val2017.json
          ├── images
          │   ├── train2017
          │   ├── val2017
          │   └── val2017.tar.xz
          └── person_detection_results
          │   ├── COCO_test-dev2017_detections_AP_H_609_person.json
          │   └── COCO_val2017_detections_AP_H_56_person.json
  ├── experiments
  ├── lib
  ├── LICENSE
  ├── log
  ├── notebooks
  ├── output
  ├── README.md
  ├── requirements.txt
  ├── tools
  └── wandb
```

## Training
Multi GPU 
```$python3 tools/train_vit.py --cfg experiments/coco/vit/vit_large_uncertainty.yaml --weight {model path} --wandb --gpus 0 1 2 3```

## Trained Model
| Model | AP | Download |
| ------ | ------ | -----|
| ViTPose_huge + uncertainty  | 77.1 | [model](https://drive.google.com/file/d/1YzmJt5aI35a6mbufDkHT3ahvOAKipqcY/view?usp=sharing) |
