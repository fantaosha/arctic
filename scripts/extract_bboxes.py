import os
import json
import numpy as np
import einops
import pickle
import argparse
from typing import List
from tqdm import tqdm


def extract_bboxes(seq: str):
    data_dir = "/large_experiments/3po/data/arctic/arctic/data/arctic_data/data"
    save_dir = "/private/home/taoshaf/data/arctic/bboxes"
    mocap_dir = os.path.join(data_dir, "mocap_npy")

    print(f"Extract bounding boxes for {seq}.")

    with open(
        "/large_experiments/3po/data/arctic/arctic/data/arctic_data/data/meta/misc.json"
    ) as f:
        meta_data = json.load(f)

    subject = seq.split("/")[0]
    offset = meta_data[subject]["ioi_offset"]
    cams_int = np.array(meta_data[subject]["intris_mat"])
    cams_int = np.concatenate(
        [
            cams_int[:, 0, 0][:, None],
            cams_int[:, 1, 1][:, None],
            cams_int[:, 0, 2][:, None],
            cams_int[:, 1, 2][:, None],
        ],
        axis=-1,
    )
    cams_ext = np.array(meta_data[subject]["world2cam"])[:, :3]
    cams_ext[:, :3, 3] *= 100

    mocap_file = os.path.join(mocap_dir, f"{seq.replace('/', '_')}.npy")
    mocap_data = np.load(mocap_file, allow_pickle=True)[()]
    markers = mocap_data["subject"]["points"] / 10
    markers_cam = (
        einops.einsum(cams_ext[:, :, :3], markers, "n i j, b k j->b n k i")
        + cams_ext[:, :, 3][None, :, None]
    )

    markers_2d = (markers_cam[:, :, :, :2] / markers_cam[:, :, :, [2]]) * cams_int[
        :, None, :2
    ] + cams_int[:, None, 2:]

    cams = [f"{cam_idx}" for cam_idx in range(1, 9)]
    img_sizes = np.array(meta_data[subject]["image_size"][1:])

    bboxes = (
        np.concatenate(
            [
                markers_2d.min(axis=-2).clip([0, 0], img_sizes[None]),
                markers_2d.max(axis=-2).clip([0, 0], img_sizes[None]),
            ],
            axis=-1,
        )
        .round()
        .astype(int)
    )
    bbox_centers = (bboxes[..., :2] + bboxes[..., 2:]) / 2
    bbox_scales = bboxes[..., 2:] - bboxes[..., :2]
    bboxes = (
        np.concatenate(
            [
                (bbox_centers - 0.55 * bbox_scales).clip([0, 0], img_sizes[None]),
                (bbox_centers + 0.55 * bbox_scales).clip([0, 0], img_sizes[None]),
            ],
            axis=-1,
        )
        .round()
        .astype(int)
    )

    bbox_dir = os.path.join(save_dir, seq)
    for cam_idx, cam in enumerate(tqdm(cams)):
        bbox_cam_dir = os.path.join(bbox_dir, cam)
        if not os.path.exists(bbox_cam_dir):
            os.makedirs(bbox_cam_dir)
        num_frames = bboxes.shape[0]
        for frame_idx in tqdm(range(num_frames)):
            img_idx = frame_idx + offset
            with open(os.path.join(bbox_cam_dir, f"{img_idx:05}.pkl"), "wb") as f:
                pickle.dump(bboxes[frame_idx, cam_idx], f)


def main():
    parser = argparse.ArgumentParser(description="Extract Bounding Boxes")
    parser.add_argument("--seqs", type=str, required=True)

    args = parser.parse_args()
    seqs: List[str] = args.seqs.split(",")

    for seq in seqs:
        extract_bboxes(seq)


if __name__ == "__main__":
    main()
