"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
"""

import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from fcclip import add_maskformer2_config, add_fcclip_config
from predictor import VisualizationDemo
from PIL import Image
import torch
from pathlib import Path
from tqdm import tqdm
import glob
# constants
WINDOW_NAME = "fc-clip demo"


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="fcclip demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument('--root_folder', help='root folder of dataset',
                        default='/data/DSEC')

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser




if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    image_paths = glob.glob(args.root_folder + "/*/*/images_aligned/left/*.png", recursive=True)
    image_paths = sorted(image_paths)


    for i_path in tqdm(image_paths, desc="Generating pseudo labels"):
        path = Path(i_path)
        city_name = i_path.split('/')[-4]    
        train_or_test = i_path.split('/')[-5] 
        image_name = path.name      
        save_path = args.root_folder / train_or_test / city_name / "pl_fcclip_rgb" / "left" / image_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img = read_image(path, format="BGR")

        predictions, visualized_output = demo.run_on_image(img)
        user_to_dsec = torch.tensor([
                0,
                1, 1,
                2,
                3, 3, 3, 3, 3,
                4, 4, 4,
                5,
                6,
                7, 7, 7, 7, 7,
                8,8,8,8,8,8,8,8,8,8,8,8,8,
                9,
                10,10,10
            ], device=predictions["sem_seg"].device)

        sem_seg = predictions["sem_seg"]  
        H, W = sem_seg.shape[1:]

        num_dsec_classes = 11
        dsec_logits = torch.full((num_dsec_classes, H, W), float('-inf'), device=sem_seg.device)

        for user_idx in range(len(user_to_dsec)):
            dsec_idx = user_to_dsec[user_idx]
            dsec_logits[dsec_idx] = torch.maximum(dsec_logits[dsec_idx], sem_seg[user_idx])

        pseudo_mask = dsec_logits.argmax(0)  
        pseudo_np = pseudo_mask.cpu().numpy().astype(np.uint8)

        Image.fromarray(pseudo_np).save(save_path)