import os

os.environ["OMP_NUM_THREADS"] = "1"

from tqdm import tqdm
import sys
from pathlib import Path
import argparse
import numpy as np

import torch
import json
import pandas as pd
import pickle
import shutil
from typing import List, Optional


import subprocess


sys.path.append(Path(os.path.abspath(__file__)).parents[1].__str__())


from mesh_fitting.optim.model.final_atlas import FinalATLAS as ATLAS
from mesh_fitting.atlas.atlas import FinalATLAS as OrigATLAS
from mesh_fitting.utils.pipeline import get_skel_params_in_cam


def anno_frames(
    model: ATLAS,
    data_dir: str,
    result_dir: str,
    keypoint: str,
    anno_dir: str,
    result_file: str,
    rescale: float,
    seq: str,
    cams: List[str],
    device: str,
    start: int,
    end: int,
    num_groups: int,
    group: int,
):
    dtype = torch.float32

    for cam in cams:
        anno_frames_for_single_cam(
            model,
            data_dir=data_dir,
            result_dir=result_dir,
            keypoint=keypoint,
            anno_dir=anno_dir,
            result_file=result_file,
            rescale=rescale,
            seq=seq,
            cam=cam,
            cams=cams,
            device=device,
            dtype=dtype,
            start=start,
            end=end,
            num_groups=num_groups,
            group=group,
        )


def anno_frames_for_single_cam(
    model: ATLAS,
    data_dir: str,
    result_dir: str,
    keypoint: str,
    anno_dir: str,
    result_file: str,
    rescale: float,
    seq: str,
    cam: str,
    cams: List[str],
    device: str,
    dtype: torch.Type,
    start: int,
    end: int,
    num_groups: int,
    group: int,
):
    atlas_dir = "/private/home/taoshaf/data/atlas/v4.6"

    data_dir: str = os.path.join(data_dir, seq)
    result_dir: str = os.path.join(result_dir, seq, "results")

    result_file = os.path.join(result_dir, "final_atlas", keypoint, result_file)
    results = pd.read_pickle(result_file)["all"]

    cam_idxs = {cam: cam_idx for cam_idx, cam in enumerate(cams)}
    cam_idx = cam_idxs[cam]

    bbox_dir = os.path.join(data_dir, "bboxes", cam)

    param_mapping = pd.read_pickle(
        os.path.join(atlas_dir, "optim2orig_param_mapping.pkl")
    ).to(device=device, dtype=dtype)

    anno_cam_dir = os.path.join(anno_dir, keypoint, seq, cam)
    if not os.path.exists(anno_cam_dir):
        try:
            os.makedirs(anno_cam_dir)
        except Exception as e:
            pass

    # img_dst_dir = os.path.join(anno_dir, "imgs", seq)
    # if not os.path.exists(img_dst_dir):
    #     img_src_dir = os.path.join(data_dir, "images")
    #     os.symlink(img_src_dir, img_dst_dir)

    frame_idxs = list(results.keys())
    num_frames = len(frame_idxs)
    if end == -1:
        end = num_frames
    start = max(0, start)
    end = min(num_frames, end)

    num_frames_per_group = max(0, (end - start + num_groups - 1) // num_groups)
    first = start + num_frames_per_group * group
    last = min(end, num_frames_per_group * (group + 1))

    frame_idxs = np.array(frame_idxs[first:last])

    cams_ext = torch.tensor(
        np.array([results[frame_idx]["camera_ext"] for frame_idx in frame_idxs]),
        device=device,
        dtype=dtype,
    )[:, cam_idx]
    cams_int = (
        torch.tensor(
            np.array([results[frame_idx]["camera_int"] for frame_idx in frame_idxs]),
            device=device,
            dtype=dtype,
        )
        * rescale
    )[:, cam_idx]
    skel_params = torch.tensor(
        np.array([results[frame_idx]["skel_params"] for frame_idx in frame_idxs]),
        device=device,
        dtype=dtype,
    )
    shape_params = torch.tensor(
        np.array([results[frame_idx]["shape_params"][:16] for frame_idx in frame_idxs]),
        device=device,
        dtype=dtype,
    )
    opt_params = torch.tensor(
        np.array(
            [
                results[frame_idx]["opt_params"]["skel_params"]
                for frame_idx in frame_idxs
            ]
        ),
        device=device,
        dtype=dtype,
    )

    skel_params = get_skel_params_in_cam(model, skel_params, cams_ext)
    opt_params = get_skel_params_in_cam(model, opt_params, cams_ext)
    cams_t = skel_params[:, :3].clone()
    skel_params[:, :3] = 0

    keypoints_2d = []
    keypoints_3d = []
    for batch_start in range(0, len(frame_idxs), 100):
        batch_end = min(batch_start + 100, len(frame_idxs))
        cam_int = cams_int[batch_start:batch_end, None]
        kpts_3d = model.forward(
            skel_params[batch_start:batch_end],
            shape_params[batch_start:batch_end],
            return_keypoints=True,
        )["keypoints"][:, :70]
        kpts_3d_cam = kpts_3d + cams_t[batch_start:batch_end, None]
        kpts_3d_cam[:, :, [1, 2]] *= -1
        kpts_2d = (kpts_3d_cam[..., :2] / kpts_3d_cam[..., 2:]) * cam_int[
            ..., :2
        ] + cam_int[..., 2:]

        keypoints_2d.append(
            torch.concat([kpts_2d, kpts_2d.new_ones(*kpts_2d.shape[:-1], 1)], dim=-1)
            .cpu()
            .numpy()
        )
        keypoints_3d.append(
            torch.concat(
                [kpts_3d / 100, kpts_3d.new_ones(*kpts_3d.shape[:-1], 1)], dim=-1
            )
            .cpu()
            .numpy()
        )
    keypoints_2d = np.concatenate(keypoints_2d, axis=0)
    keypoints_3d = np.concatenate(keypoints_3d, axis=0)

    orig_skel_params = skel_params @ param_mapping.transpose(-2, -1)
    orig_skel_params[:, 68:122] = 0

    orig_root_t = orig_skel_params[:, :3] / 100
    orig_root_r = orig_skel_params[:, 3:6]
    orig_body_params = orig_skel_params[:, 6:-68]
    orig_hand_params = torch.cat(
        [opt_params[..., 98:114], opt_params[..., 82:98]], dim=-1
    )
    orig_scale_params = opt_params[:, -38:].clone()
    orig_shape_params = shape_params.clone()
    orig_cams_t = cams_t.clone() / 100

    orig_root_t = orig_root_t.cpu().numpy()
    orig_root_r = orig_root_r.cpu().numpy()
    orig_body_params = orig_body_params.cpu().numpy()
    orig_hand_params = orig_hand_params.cpu().numpy()
    orig_scale_params = orig_scale_params.cpu().numpy()
    orig_shape_params = orig_shape_params.cpu().numpy()
    orig_cams_t = orig_cams_t.cpu().numpy()

    for batch_idx in tqdm(range(len(frame_idxs))):
        frame_idx = frame_idxs[batch_idx]
        bbox_file = os.path.join(bbox_dir, f"{frame_idx:05}.pkl")
        if not os.path.exists(bbox_file):
            continue

        bbox = np.round(pd.read_pickle(bbox_file) * rescale).astype(int)

        if min(bbox[2:] - bbox[:2]) < 15:
            continue

        body_kpts_2d = keypoints_2d[batch_idx][:17, :2]
        num_body_kpts_in_bbox = np.all(
            (body_kpts_2d[:, :2] >= bbox[:2]) * (body_kpts_2d[:, :2] <= bbox[2:]),
            axis=-1,
        ).sum()
        # if num_body_kpts_in_bbox <= 6:
        #     continue

        cam_int = cams_int[batch_idx].cpu().numpy()
        cam_int *= rescale
        intrinsics = np.eye(3, 3, dtype=np.float32)
        intrinsics[0, 0] = cam_int[0]
        intrinsics[1, 1] = cam_int[1]
        intrinsics[:2, 2] = cam_int[2:]

        atlas_params = {}
        atlas_params["global_trans"] = orig_root_t[batch_idx]
        atlas_params["global_rot"] = orig_root_r[batch_idx]
        atlas_params["body_pose_params"] = orig_body_params[batch_idx]
        atlas_params["hand_pose_params"] = orig_hand_params[batch_idx]
        atlas_params["scale_params"] = orig_scale_params[batch_idx]
        atlas_params["shape_params"] = orig_shape_params[batch_idx][:16]
        atlas_params["expr_params"] = torch.zeros(10, dtype=dtype).numpy()

        anno = {}
        anno["person_id"] = 0
        anno["keypoints_2d"] = keypoints_2d[batch_idx]
        anno["keypoints_3d"] = keypoints_3d[batch_idx]
        anno["atlas_params"] = atlas_params
        anno["atlas_valid"] = True
        anno["atlas_version"] = "trinity_v4.6"
        anno["bbox"] = bbox
        anno["bbox_format"] = "xyxy"
        anno["bbox_score"] = 1.0
        anno["center"] = 0.5 * (bbox[:2] + bbox[2:])
        anno["scale"] = bbox[2:] - bbox[:2]
        anno["metadata"] = {
            "cam_trans": orig_cams_t[batch_idx],
            "cam_int": intrinsics,
            "loss": 0,
        }
        annos = [anno]

        # file = os.path.join(anno_cam_dir, f"{seq}-{cam}-{frame_idx:06}.pkl")
        file = os.path.join(anno_cam_dir, f"{frame_idx:06}.pkl")
        with open(file, "wb") as f:
            pickle.dump(annos, f)


def main():
    parser = argparse.ArgumentParser(description="Annotate ARCTIC")
    parser.add_argument("--seqs", type=str, required=True)
    parser.add_argument("--cams", type=str, default="")
    parser.add_argument(
        "--data_dir", type=str, default="/private/home/taoshaf/data/arctic/"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/private/home/taoshaf/data/arctic/",
    )
    parser.add_argument(
        "--anno_dir",
        type=str,
        default="/private/home/taoshaf/data/annotation/arctic/annos/full",
    )
    parser.add_argument(
        "--result_file", type=str, default="init_single_frame_results.pkl"
    )
    parser.add_argument("--rescale", type=float, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--keypoint", type=str, default="sparse")
    parser.add_argument("--num_groups", type=int, default=12)
    parser.add_argument("--group", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)

    args = parser.parse_args()

    device = args.device
    cam_idxs: Optional[str] = args.cams
    seqs: List[str] = args.seqs.split(",")
    data_dir: str = args.data_dir
    keypoint: str = args.keypoint
    anno_dir: str = args.anno_dir
    result_dir: str = args.result_dir
    result_file: str = args.result_file
    num_groups: int = args.num_groups
    group: int = args.group
    start: int = args.start
    end: int = args.end
    rescale: float = args.rescale

    outputs = {0: subprocess.DEVNULL, 1: None}

    dtype = torch.float32

    atlas_dir = "/private/home/taoshaf/data/atlas/v4.6"

    num_shape_comps = 16
    num_scale_comps = 38
    num_hand_comps = 16
    num_expr_comps = 10

    # Load in ATLAS
    model = ATLAS(
        atlas_dir,
        num_shape_comps,
        num_scale_comps,
        num_hand_comps,
        num_expr_comps,
        load_keypoint_mapping=True,
        lod="lod3",
        verbose=True,
    ).to(device=device, dtype=dtype)

    if group >= 0:
        for seq in seqs:
            cam_idxs: Optional[str] = args.cams
            cams = sorted(
                [
                    cam
                    for cam in os.listdir(os.path.join(args.data_dir, seq, "bboxes"))
                    if cam.isdigit()
                ]
            )
            num_cams = len(cams)

            if cam_idxs != "":
                cam_idxs: List[str] = [int(idx) for idx in cam_idxs.split(",")]
                cams = [
                    cams[cam_idx]
                    for cam_idx in cam_idxs
                    if cam_idx >= 0 and cam_idx < len(cams)
                ]
            else:
                cam_idxs = np.array(num_cams)
            print(f"Annotate {seq} for {", ".join(cams)}")

            # try:
            anno_frames(
                model,
                data_dir,
                result_dir,
                keypoint,
                anno_dir,
                result_file,
                rescale,
                seq,
                cams,
                device,
                start,
                end,
                num_groups,
                group,
            )
            # except Exception as e:
            #     print(f"An error occurred: {e}")

    else:
        cmd = ["python", f"{os.path.abspath(__file__)}"]
        arg_vars = vars(args)
        cmd_args = sum(
            [[f"--{key}", f"{val}"] for key, val in arg_vars.items() if key != "group"],
            [],
        )
        processes = [
            subprocess.Popen(
                cmd + cmd_args + ["--group", f"{group}"], stdout=outputs[group == 0]
            )
            for group in range(num_groups)
        ]
        try:
            for process in processes:
                process.wait()
        except Exception as e:
            print(f"An error occurred: {e}")
            for process in processes:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
