import io
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

from mmpose.apis import inference_top_down_pose_model, init_pose_model
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import argparse
from typing import List
from skimage.measure import label
import io
from PIL import Image
import argparse


def main():
    parser = argparse.ArgumentParser(description="Segment & Track People")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seqs", type=str, required=True)
    parser.add_argument("--cams", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=-1)

    args = parser.parse_args()

    data_dir: str = args.data_dir
    seqs: List[str] = args.seqs.split(",")
    cams: List[str] = args.cams.split(",")
    device: str = args.device

    VIT_DIR = "/private/home/taoshaf/Documents/python/ViTPose"
    VITPOSE_CKPT_PTH = (
        "/private/home/taoshaf/checkpoints/vitpose/vitpose_h_wholebody/wholebody.pth"
    )
    VITPOSE_CFG = os.path.join(
        VIT_DIR,
        "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py",
    )
    detector = init_pose_model(VITPOSE_CFG, VITPOSE_CKPT_PTH, device=device)

    def estimate_pose(seq: str, cam: str):
        print("-----------------------------------------------")
        print(f"Estimate {seq}/images/{cam}.")
        save_dir = "/private/home/taoshaf/data/arctic"
        img_dir = os.path.join(data_dir, "images", seq, cam)
        bbox_dir = os.path.join(save_dir, seq, "bboxes", cam)
        keypoint_dir = os.path.join(save_dir, seq, "keypoints", "vitpose", cam)

        if not os.path.exists(keypoint_dir):
            try:
                os.makedirs(keypoint_dir)
            except:
                pass

        image_idxs = sorted([int(img_idx[:-4]) for img_idx in os.listdir(bbox_dir)])
        for image_idx in tqdm(image_idxs):
            img_file = os.path.join(img_dir, f"{image_idx:05}.jpg")
            bbox_file = os.path.join(bbox_dir, f"{image_idx:05}.pkl")

            if not os.path.exists(img_file) or not os.path.exists(bbox_file):
                continue

            img = cv2.imread(img_file)
            bbox = pd.read_pickle(bbox_file)
            result, _ = inference_top_down_pose_model(
                detector,
                img,
                person_results=[{"bbox": bbox}],
                format="xyxy",
            )
            keypoints = {1: result[0]["keypoints"]}
            keypoint_file = os.path.join(keypoint_dir, f"{image_idx:06}.pkl")
            with open(keypoint_file, "wb") as f:
                pickle.dump(keypoints, f)

    for seq in seqs:
        for cam in cams:
            estimate_pose(seq, cam)


if __name__ == "__main__":
    main()
