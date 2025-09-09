import os

os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import io
import pickle
from typing import List
import argparse

from typing import List, Optional

import subprocess


def convert_dense_keypoints(
    data_dir: str,
    save_dir: str,
    seq: str,
    cams: List[str],
):
    for cam in cams:
        convert_dense_keypoints_for_single_cam(data_dir, save_dir, seq, cam)


def convert_dense_keypoints_for_single_cam(
    data_dir: str,
    save_dir: str,
    seq: str,
    cam: str,
):
    src_dir: str = os.path.join(data_dir, seq)
    dst_dir: str = os.path.join(save_dir, seq)

    dst_dir: str = os.path.join(dst_dir, "keypoints", "dense", cam)
    if not os.path.exists(dst_dir):
        try:
            os.makedirs(dst_dir)
        except Exception as e:
            print(e)

    kpts = {}
    src_cam_dir = os.path.join(src_dir, cam)
    if not os.path.exists(src_cam_dir):
        print(f"Failed to process {seq}-{cam}.")
        return
    kpt_files = [
        os.path.join(src_cam_dir, kpt_file)
        for kpt_file in sorted(os.listdir(os.path.join(src_cam_dir)))
        if kpt_file.endswith(".pkl")
    ]
    for kpt_file in tqdm(kpt_files):
        frame_idx = int(kpt_file.split(".pkl")[0][-6:])
        kpt = pd.read_pickle(kpt_file)
        if frame_idx not in kpts:
            kpts[frame_idx] = {}
        kpts[frame_idx][1] = np.concatenate([kpt["joints2d"], kpt["conf"]], axis=-1)

    dst_frame_idxs = [
        int(file.split(".pkl")[0])
        for file in os.listdir(os.path.join(dst_dir))
        if file.endswith(".pkl")
    ]

    for frame_idx in dst_frame_idxs:
        if frame_idx not in kpts:
            os.remove(os.path.join(dst_dir, f"{frame_idx:06}.pkl"))

    for frame_idx, kpt in kpts.items():
        with open(os.path.join(dst_dir, f"{frame_idx:06}.pkl"), "wb") as f:
            pickle.dump(kpt, f)

    print(f"{seq}-{cam} has been processed.")


def main():
    parser = argparse.ArgumentParser(description="Convert Dense Keypoints")
    parser.add_argument("--seqs", type=str, required=True)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/checkpoint/taoshaf/results/CondDenseDetection/arctic/",
    )
    parser.add_argument(
        "--save_dir", type=str, default="/private/home/taoshaf/data/arctic/"
    )
    parser.add_argument("--num_jobs", type=int, default=8)

    args = parser.parse_args()

    seqs: List[str] = args.seqs.split(",")
    data_dir: str = args.data_dir
    save_dir: str = args.save_dir
    num_seqs = len(seqs)
    num_jobs: int = min(args.num_jobs, num_seqs)

    if num_jobs == 1:
        for seq in seqs:
            cams = sorted(
                [
                    cam
                    for cam in os.listdir(os.path.join(args.save_dir, seq, "bboxes"))
                    if cam.isdigit()
                ]
            )

            print(f"Convert dense keypoints for {seq} with {", ".join(cams)}")

            try:
                convert_dense_keypoints(
                    data_dir,
                    save_dir,
                    seq,
                    cams,
                )
            except Exception as e:
                print(f"An error occurred: {e}")

    else:
        num_seqs_per_job = num_seqs // num_jobs
        seqs_for_jobs = [
            num_seqs_per_job + (job < (num_seqs - num_seqs_per_job * num_jobs))
            for job in range(num_jobs)
        ]
        seq_offsets = np.cumsum([0] + seqs_for_jobs)

        cmd = ["python", f"{os.path.abspath(__file__)}"]
        arg_vars = vars(args)
        cmd_args = sum(
            [
                [f"--{key}", f"{val}"]
                for key, val in arg_vars.items()
                if key != "num_jobs" and key != "seqs"
            ],
            [],
        )
        processes = [
            subprocess.Popen(
                cmd
                + cmd_args
                + [
                    "--num_jobs",
                    "1",
                    "--seqs",
                    ",".join(seqs[seq_offsets[job] : seq_offsets[job + 1]]),
                ],
            )
            for job in range(num_jobs)
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
