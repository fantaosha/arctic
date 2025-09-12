import os
import hydra
from omegaconf import DictConfig
from mesh_fitting.optim.model import FinalATLAS
import torch
import numpy as np
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import pickle
from mesh_fitting.utils.pipeline import (
    load_keypoints,
    relax,
    ransac,
    ransac_hand,
    smooth,
    get_min_keypoint_2d_scores,
)
from mesh_fitting.optim.kinematics import normalize_rotmat, transform_root_pose
from mesh_fitting.optim.model.utils import lod_mapping
import json
from theseus import SO3
from linear_blend_skinning_cuda.quaternions import Quaternion
from arctic.configs.markers import *
from arctic.utils.pipeline import optimize_frames, optimize_stages


def estimate_root_pose(
    model: FinalATLAS,
    keypoints_3d: torch.Tensor,
    keypoints_3d_ref: torch.Tensor,
):
    b = keypoints_3d.shape[0]
    device = keypoints_3d.device
    dtype = keypoints_3d.dtype
    skel_params = torch.zeros(b, model.num_skel_params).to(device=device, dtype=dtype)
    shape_params = torch.zeros(b, model.num_shape_comps).to(device=device, dtype=dtype)

    _ = model(
        skel_params,
        shape_params,
        do_pcblend=False,
        return_keypoints=False,
        return_joint_coords=False,
        return_joint_params=False,
        return_scales=False,
    )

    joints_t = model.lbs_fn.out_joint_state_t.to(dtype)
    joints_r = model.lbs_fn.out_joint_state_r.to(dtype)

    X = (keypoints_3d_ref - joints_t[..., None, 1, :]) @ joints_r[..., 1, :, :]
    Y = keypoints_3d
    Xmean = X.mean(dim=-2)
    Ymean = Y.mean(dim=-2)
    X = X - Xmean[..., None, :]
    Y = Y - Ymean[..., None, :]
    R = SO3.normalize((Y.transpose(-2, -1) @ X).view(-1, 3, 3))
    rot = joints_r[..., 1, :, :].view(-1, 3, 3).transpose(-2, -1) @ R

    root_pose = skel_params.new_empty(b, 6)
    root_pose[:, 0:3] = (
        Ymean
        - torch.einsum("b...ij,b...j->bi", R, Xmean)
        - model.lbs_fn.joint_offset[1]
    )
    root_pose[:, 3:6] = Quaternion.batchToXYZ(
        Quaternion.batchQuatFromMatrix(rot.to(torch.float32))
    )
    return root_pose.reshape(b, 6).to(dtype)


def load_keypoints(cfg: DictConfig, data_dir: str, cameras: List[str]):
    keypoint_dir: str = os.path.join(data_dir, "keypoints")
    if cfg.keypoint_model is not None:
        keypoint_dir: str = os.path.join(keypoint_dir, cfg.keypoint_model)
    keypoint_data = {}
    for cam in cameras:
        keypoint_cam_dir = os.path.join(keypoint_dir, cam)
        if not os.path.exists(keypoint_cam_dir):
            keypoint_data[cam] = {}
            continue
        image_idxs = sorted(
            [
                int(file[:-4])
                for file in os.listdir(keypoint_cam_dir)
                if file.endswith(".pkl")
            ]
        )
        keypoint_data[cam] = {
            image_idx: pd.read_pickle(
                os.path.join(keypoint_cam_dir, f"{image_idx:06}.pkl")
            )[1]
            for image_idx in image_idxs
        }
    return keypoint_data


def init_single_frame_fitting(
    cfg: DictConfig,
    frame_idxs: np.ndarray,
    cameras: List[int],
    model: FinalATLAS,
    cams_ext: torch.Tensor,
    cams_int: torch.Tensor,
    keypoint_sparse_2d_info: Dict[str, torch.Tensor],
    keypoint_sparse_2d_pixels: torch.Tensor,
    keypoint_sparse_2d_scores: torch.Tensor,
    keypoint_sparse_2d_masks: torch.Tensor,
    keypoint_dense_2d_info: Dict[str, torch.Tensor],
    keypoint_dense_2d_pixels: torch.Tensor,
    keypoint_dense_2d_scores: torch.Tensor,
    keypoint_dense_3d_info: Dict[str, torch.Tensor],
    keypoint_dense_3d_poses: torch.Tensor,
    keypoint_dense_3d_scores: torch.Tensor,
):
    device = keypoint_sparse_2d_pixels.device
    dtype = keypoint_sparse_2d_pixels.dtype

    num_frames = keypoint_sparse_2d_pixels.shape[0]
    skel_params = torch.zeros(
        num_frames, model.num_skel_params, device=device, dtype=dtype
    )
    shape_params = torch.zeros(
        num_frames, model.num_shape_comps, device=device, dtype=dtype
    )

    torso_idxs = [MARKERS.index(marker) for marker in TORSO_MARKERS]
    skel_params[..., :6] = estimate_root_pose(
        model,
        keypoint_dense_3d_poses[..., torso_idxs, :],
        keypoint_dense_3d_info["mean"][..., torso_idxs, :],
    )

    if cfg.init_file is not None:
        data_dir: str = os.path.join(cfg.paths.data_dir, cfg.dataset, cfg.sequence)
        init_dir: str = os.path.join(
            data_dir, "results", cfg.get("init_folder", cfg.result_folder)
        )
        init_file: str = os.path.join(init_dir, cfg.init_file)
        inits = pd.read_pickle(init_file)["all"]
        for batch_idx, frame_idx in enumerate(frame_idxs):
            init = inits[frame_idx]
            skel_params[batch_idx] = torch.from_numpy(init["skel_params"]).to(
                device, dtype
            )
            shape_params[batch_idx] = torch.from_numpy(init["shape_params"]).to(
                device, dtype
            )
            cams_ext[batch_idx] = torch.from_numpy(init["camera_ext"]).to(device, dtype)
            cams_int[batch_idx] = torch.from_numpy(init["camera_int"]).to(device, dtype)

    keypoint_sparse_2d_scores = keypoint_sparse_2d_scores[..., None]
    keypoint_sparse_2d_masks = keypoint_sparse_2d_masks[..., None]
    keypoint_dense_2d_scores = keypoint_dense_2d_scores[..., None]
    keypoint_dense_3d_scores = keypoint_dense_3d_scores[..., None]

    results = {"all": {}}
    for start in tqdm(range(0, num_frames, cfg.batch_size)):
        end = min(start + cfg.batch_size, num_frames)
        results = optimize_frames(
            cfg.optim,
            frame_idxs=frame_idxs[start:end],
            model=model,
            cams_ext=cams_ext[start:end],
            cams_int=cams_int[start:end],
            shape_params=shape_params[start:end],
            skel_params=skel_params[start:end],
            opt_params=None,
            opt_states=None,
            keypoint_sparse_2d_info=keypoint_sparse_2d_info,
            keypoint_sparse_2d_pixels=keypoint_sparse_2d_pixels[start:end],
            keypoint_sparse_2d_scores=keypoint_sparse_2d_scores[start:end],
            keypoint_sparse_2d_masks=keypoint_sparse_2d_masks[start:end],
            keypoint_dense_2d_info=keypoint_dense_2d_info,
            keypoint_dense_2d_pixels=keypoint_dense_2d_pixels[start:end],
            keypoint_dense_2d_scores=(keypoint_dense_2d_scores[start:end]),
            keypoint_dense_3d_info=keypoint_dense_3d_info,
            keypoint_dense_3d_poses=keypoint_dense_3d_poses[start:end],
            keypoint_dense_3d_scores=keypoint_dense_3d_scores[start:end],
            results=results,
        )

    return results


def single_frame_fitting(
    cfg: DictConfig,
    frame_idxs: np.ndarray,
    cameras: List[int],
    model: FinalATLAS,
    cams_ext: torch.Tensor,
    cams_int: torch.Tensor,
    keypoint_sparse_2d_info: Dict[str, torch.Tensor],
    keypoint_sparse_2d_pixels: torch.Tensor,
    keypoint_sparse_2d_scores: torch.Tensor,
    keypoint_sparse_2d_masks: torch.Tensor,
    keypoint_dense_2d_info: Dict[str, torch.Tensor],
    keypoint_dense_2d_pixels: torch.Tensor,
    keypoint_dense_2d_scores: torch.Tensor,
    keypoint_dense_3d_info: Dict[str, torch.Tensor],
    keypoint_dense_3d_poses: torch.Tensor,
    keypoint_dense_3d_scores: torch.Tensor,
):
    device = keypoint_sparse_2d_pixels.device
    dtype = keypoint_sparse_2d_pixels.dtype

    data_dir: str = os.path.join(cfg.paths.data_dir, cfg.dataset, cfg.sequence)
    result_dir: str = os.path.join(data_dir, "results", cfg.result_folder)
    init_guess = pd.read_pickle(os.path.join(result_dir, cfg.init_file))["all"]

    assert np.all(np.array(list(init_guess.keys())) == np.array(frame_idxs))

    cams_ext = np.array(
        [init_guess[frame_idx]["camera_ext"] for frame_idx in frame_idxs]
    )
    cams_int = np.array(
        [init_guess[frame_idx]["camera_int"] for frame_idx in frame_idxs]
    )
    shape_params = np.array(
        [init_guess[frame_idx]["shape_params"] for frame_idx in frame_idxs]
    )
    skel_params = np.array(
        [init_guess[frame_idx]["skel_params"] for frame_idx in frame_idxs]
    )
    opt_params = {
        key: np.array(
            [init_guess[frame_idx]["opt_params"][key] for frame_idx in frame_idxs]
        )
        for key in init_guess[frame_idxs[0]]["opt_params"]
    }
    opt_param_transforms = np.array(
        [init_guess[frame_idx]["param_transform"] for frame_idx in frame_idxs]
    )

    assert (opt_param_transforms == opt_param_transforms[0]).all()

    cams_ext = torch.from_numpy(cams_ext).to(device=device, dtype=dtype)
    cams_int = torch.from_numpy(cams_int).to(device=device, dtype=dtype)
    shape_params = torch.from_numpy(shape_params).to(device=device, dtype=dtype)
    skel_params = torch.from_numpy(skel_params).to(device=device, dtype=dtype)
    opt_params = {
        key: torch.from_numpy(val).to(device=device, dtype=dtype)
        for key, val in opt_params.items()
    }
    opt_states = {"param_transform": opt_param_transforms[0]}
    param_transforms = {}
    for param, tf_name in opt_states["param_transform"].items():
        if param == "cameras_ext":
            num_cams = cams_ext.shape[-3]
            param_transforms["cameras_ext"] = ParamPoseTransform(
                size=num_cams,
                param_idxs={"cameras_ext": np.arange(num_cams)},
                name="cam_ext",
            )
        elif param == "cameras_int":
            raise NotImplementedError
        elif param == "shape_params":
            param_transforms["shape_params"] = model.param_shape_transform
        elif param == "skel_params":
            param_transforms["skel_params"] = getattr(
                model, f"param_{tf_name}_transform"
            )
        else:
            raise NotImplementedError

    skel_params[:, model.scale_idxs] = skel_params[:, model.scale_idxs].mean(
        dim=0, keepdim=True
    )
    shape_params[:] = shape_params[:].mean(dim=0, keepdim=True)
    opt_params["shape_params"][:] = opt_params["shape_params"].mean(dim=0, keepdim=True)
    opt_params["skel_params"][
        :, param_transforms["skel_params"].param_idxs["scale"]
    ] = opt_params["skel_params"][
        :, param_transforms["skel_params"].param_idxs["scale"]
    ].mean(
        dim=0, keepdim=True
    )

    # Estimate Shape ans Scale Parameters
    num_frames = len(frame_idxs)

    init_frame_range = np.arange(num_frames)[
        int(num_frames * 0.05) : -int(num_frames * 0.05) : (
            num_frames + cfg.batch_size - 1
        )
        // cfg.batch_size
    ]

    opt_info = {}
    opt_info["keypoint_sparse_2d"] = keypoint_sparse_2d_info
    opt_info["keypoint_dense_2d"] = keypoint_dense_2d_info
    opt_info["keypoint_dense_3d"] = keypoint_dense_3d_info

    variables_shared = {
        "cameras_ext": cams_ext[init_frame_range],
        "cameras_int": cams_int[init_frame_range],
        "shape_params": shape_params[init_frame_range],
        "skel_params": skel_params[init_frame_range],
    }
    opt_params_shared = {key: val[init_frame_range] for key, val in opt_params.items()}
    opt_data_shared = {}
    opt_data_shared["keypoint_sparse_2d"] = {
        "keypoint_2d_pixels": keypoint_sparse_2d_pixels[init_frame_range],
        "keypoint_2d_scores": keypoint_sparse_2d_scores[init_frame_range],
        "keypoint_2d_masks": keypoint_sparse_2d_masks[init_frame_range],
    }
    opt_data_shared["keypoint_dense_2d"] = {
        "keypoint_2d_pixels": keypoint_dense_2d_pixels[init_frame_range],
        "keypoint_2d_scores": keypoint_dense_2d_scores[init_frame_range],
    }
    opt_data_shared["keypoint_dense_3d"] = {
        "keypoint_3d_poses": keypoint_dense_3d_poses[init_frame_range],
        "keypoint_3d_scores": keypoint_dense_3d_scores[init_frame_range],
    }

    variables_shared, opt_params_shared, opt_states_shared = optimize_stages(
        cfg.init_optim.shared_params,
        model=model,
        variables=variables_shared,
        opt_params=opt_params_shared,
        opt_info=opt_info,
        opt_states=opt_states,
        opt_data=opt_data_shared,
    )


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(cfg: DictConfig):
    device = cfg.device
    dtype = torch.float64
    num_markers = 73

    arctic_dir = "/large_experiments/3po/data/arctic/arctic/data/arctic_data/data"
    split_file = os.path.join(arctic_dir, "splits", "p1_train.npy")
    seq_info = np.load(split_file, allow_pickle=True)[()]
    with open(
        "/large_experiments/3po/data/arctic/arctic/data/arctic_data/data/meta/misc.json"
    ) as f:
        meta_data = json.load(f)

    # Load keypoint info
    # Sparse 2D keypoint info
    keypoint_sparse_2d_file = os.path.join(cfg.model.model_data_dir, "keypoints.pkl")
    keypoint_sparse_2d_info = pd.read_pickle(keypoint_sparse_2d_file)
    keypoint_sparse_2d_joint_mapping = keypoint_sparse_2d_info["joint"]
    keypoint_sparse_2d_dirs = {
        kpt: torch.tensor(
            kpt_dir[:16],
            device=device,
            dtype=dtype,
        )[: cfg.model.num_shape_comps]
        for kpt, kpt_dir in keypoint_sparse_2d_info["dirs"].items()
    }
    keypoint_sparse_2d_mean = {
        kpt: torch.tensor(kpt_mean, device=device, dtype=dtype)
        for kpt, kpt_mean in keypoint_sparse_2d_info["mean"].items()
    }
    keypoint_sparse_2d_idxs = cfg.keypoints.sparse.keypoint_idxs

    keypoint_sparse_2d_info: Dict[str, torch.Tensor] = {
        "idxs": keypoint_sparse_2d_idxs,
        "joint": keypoint_sparse_2d_joint_mapping,
        "dirs": keypoint_sparse_2d_dirs,
        "mean": keypoint_sparse_2d_mean,
    }

    # Dense 2D keypoint info
    lod_dense_2d = lod_mapping(
        cfg.model.model_data_dir,
        16,
        cfg.model.num_expr_comps,
        lod="lod_595",
    )

    keypoint_dense_2d_info: Dict[str, torch.Tensor] = {
        "weights": lod_dense_2d["skin_weights"].to(device=device, dtype=dtype),
        "joints": lod_dense_2d["skin_indices"].numpy(),
        "dirs": lod_dense_2d["shape_comps"].to(device=device, dtype=dtype),
        "mean": lod_dense_2d["shape_mean"].to(device=device, dtype=dtype),
    }
    keypoint_dense_2d_info["dirs"][16:, :] = 0

    # Dense 3D keypoint info
    keypoint_dense_3d_info = pd.read_pickle(
        "/private/home/taoshaf/Documents/projects/arctic/data/markers.pkl"
    )
    keypoint_dense_3d_info: Dict[str, torch.Tensor] = {
        "weights": keypoint_dense_3d_info["skin_weights"].to(
            device=device, dtype=dtype
        ),
        "joints": keypoint_dense_3d_info["skin_indices"].numpy(),
        "dirs": torch.cat(
            [
                keypoint_dense_3d_info["shape_comps"].to(device=device, dtype=dtype),
                torch.zeros(num_markers * 3, num_markers, 3).to(
                    device=device, dtype=dtype
                ),
            ],
            dim=0,
        ),
        "mean": keypoint_dense_3d_info["shape_mean"].to(device=device, dtype=dtype),
    }
    for marker_idx in range(num_markers):
        keypoint_dense_3d_info["dirs"][
            16 + 3 * marker_idx : 19 + 3 * marker_idx, marker_idx
        ] = 4 * torch.eye(3)

    model: FinalATLAS = hydra.utils.instantiate(cfg.model).to(
        device=device, dtype=dtype
    )
    model.shape_comps[16:] = 0

    seq: str = cfg.sequence
    subject: str = seq.split("/")[0]
    seq_dir = os.path.join("/private/home/taoshaf/data/arctic", seq)

    offset: int = meta_data[subject]["ioi_offset"]

    # Load Cameras
    cams_int = torch.from_numpy(np.array(meta_data[subject]["intris_mat"])).to(
        device=device, dtype=dtype
    )
    cams_int = torch.concat(
        [
            cams_int[:, 0, 0][:, None],
            cams_int[:, 1, 1][:, None],
            cams_int[:, 0, 2][:, None],
            cams_int[:, 1, 2][:, None],
        ],
        dim=-1,
    )
    cams_ext = torch.from_numpy(np.array(meta_data[subject]["world2cam"])).to(
        device=device, dtype=dtype
    )[:, :3]
    cams_ext[:, :3, 3] *= 100

    num_cams = cams_ext.shape[0]
    if cfg.cameras is None:
        cameras: List[str] = [f"{cam_idx}" for cam_idx in range(1, num_cams + 1)]
    else:
        cameras: List[str] = cfg.cameras

    mocap_path = os.path.join(arctic_dir, "mocap_npy")
    mocap_file = os.path.join(mocap_path, f"{seq.replace('/', '_')}.npy")
    mocap_data = np.load(mocap_file, allow_pickle=True)[()]

    # Load dense 3D keypoint data
    keypoint_dense_3d_poses = torch.from_numpy(
        mocap_data["subject"]["points"].copy() / 10
    ).to(device=device, dtype=dtype)
    keypoint_dense_3d_scores = torch.ones(
        [keypoint_dense_3d_poses.shape[0], keypoint_dense_3d_poses.shape[1]]
    ).to(device=device, dtype=dtype)

    # Load sparse 2D keypoint data
    keypoint_sparse_2d_data = load_keypoints(cfg.keypoints.sparse, seq_dir, cameras)
    image_idxs = np.array(
        sorted(
            list(
                set.union(
                    *[set(keypoint_sparse_2d_data[cam].keys()) for cam in cameras]
                )
            )
        )
    )
    image_idxs = image_idxs[5:]
    image_idxs = image_idxs[image_idxs >= offset]
    assert np.all(np.diff(image_idxs) == 1)
    mocap_idxs = image_idxs - offset

    # Load dense 2D keypoint data
    try:
        keypoint_dense_2d_data = load_keypoints(cfg.keypoints.dense, seq_dir, cameras)
    except:
        keypoint_dense_2d_data = None

    if keypoint_dense_2d_data is not None:
        for cam, kpts_dense_cam in keypoint_dense_2d_data.items():
            for frame_idx in kpts_dense_cam:
                if frame_idx not in keypoint_sparse_2d_data[cam]:
                    kpts_dense_cam[frame_idx][:, 2] = 0

    num_frames = len(mocap_idxs)
    cams_ext = cams_ext[None].repeat(num_frames, 1, 1, 1)
    cams_int = cams_int[None].repeat(num_frames, 1, 1)

    # Process dense 3D keypoints
    keypoint_dense_3d_poses = keypoint_dense_3d_poses[mocap_idxs]
    keypoint_dense_3d_scores = keypoint_dense_3d_scores[mocap_idxs]

    # Process sparse 2D keypoints
    keypoint_sparse_2d_data = np.stack(
        [
            [
                keypoint_sparse_2d_data[cam].get(image_idx, np.zeros([133, 3]))
                for image_idx in image_idxs
            ]
            for cam in cameras
        ],
        axis=1,
    )
    keypoint_sparse_2d_pixels = torch.tensor(
        keypoint_sparse_2d_data[..., :2], device=device, dtype=dtype
    )
    keypoint_sparse_2d_init_scores = torch.tensor(
        keypoint_sparse_2d_data[..., 2], device=device, dtype=dtype
    )

    # Filter and smooth sparse 2D Keypoints
    keypoint_sparse_2d_min_scores = torch.tensor(
        get_min_keypoint_2d_scores(cfg.keypoints.sparse), device=device, dtype=dtype
    )
    keypoint_sparse_idxs = cfg.keypoints.sparse.keypoint_idxs
    keypoint_sparse_2d_score_flags = 1
    keypoint_sparse_2d_scores = relax(
        cfg.filters.relax,
        keypoint_idxs=keypoint_sparse_idxs,
        keypoint_2d_raw_scores=keypoint_sparse_2d_init_scores
        * keypoint_sparse_2d_score_flags,
        keypoint_2d_min_scores=keypoint_sparse_2d_min_scores,
    )

    keypoint_sparse_2d_scales = (
        (
            keypoint_sparse_2d_pixels[..., :, :].max(dim=-2, keepdim=True)[0]
            - keypoint_sparse_2d_pixels[..., :, :].min(dim=-2, keepdim=True)[0]
        )
        .expand(*keypoint_sparse_2d_pixels.shape)
        .clone()
    ).clamp(1)
    keypoint_sparse_2d_masks = torch.zeros_like(keypoint_sparse_2d_scores, dtype=bool)
    keypoint_sparse_3d_poses = keypoint_sparse_2d_pixels.new_zeros(
        [keypoint_sparse_2d_pixels.shape[0], keypoint_sparse_2d_pixels.shape[-2], 3]
    )
    keypoint_sparse_3d_masks = keypoint_sparse_2d_pixels.new_zeros(
        [keypoint_sparse_2d_pixels.shape[0], keypoint_sparse_2d_pixels.shape[-2]],
        dtype=bool,
    )

    (
        keypoint_sparse_2d_pixels,
        keypoint_sparse_2d_scores,
        keypoint_sparse_2d_masks,
        keypoint_sparse_3d_poses,
        keypoint_sparse_3d_masks,
    ) = ransac(
        cfg.filters.ransac,
        keypoint_idxs=keypoint_sparse_idxs,
        keypoint_2d_pixels=keypoint_sparse_2d_pixels,
        keypoint_2d_scores=keypoint_sparse_2d_scores,
        keypoint_2d_scales=keypoint_sparse_2d_scales,
        keypoint_2d_masks=keypoint_sparse_2d_masks,
        keypoint_3d_poses=keypoint_sparse_3d_poses,
        keypoint_3d_masks=keypoint_sparse_3d_masks,
        cams_ext=cams_ext,
        cams_int=cams_int,
    )

    (
        keypoint_sparse_2d_pixels,
        keypoint_sparse_2d_scores,
        keypoint_sparse_2d_masks,
        keypoint_sparse_3d_poses,
        keypoint_sparse_3d_masks,
    ) = ransac_hand(
        cfg.filters.ransac_hand,
        keypoint_idxs=keypoint_sparse_idxs,
        keypoint_2d_pixels=keypoint_sparse_2d_pixels,
        keypoint_2d_scores=keypoint_sparse_2d_scores,
        keypoint_2d_scales=keypoint_sparse_2d_scales,
        keypoint_2d_masks=keypoint_sparse_2d_masks,
        keypoint_3d_poses=keypoint_sparse_3d_poses,
        keypoint_3d_masks=keypoint_sparse_3d_masks,
        cams_ext=cams_ext,
        cams_int=cams_int,
    )

    keypoint_sparse_2d_pixels, keypoint_sparse_3d_poses = smooth(
        cfg.filters.smooth,
        keypoint_idxs=keypoint_sparse_idxs,
        keypoint_2d_pixels=keypoint_sparse_2d_pixels,
        keypoint_2d_scores=keypoint_sparse_2d_scores,
        keypoint_2d_masks=keypoint_sparse_2d_masks,
        keypoint_3d_poses=keypoint_sparse_3d_poses,
        keypoint_3d_masks=keypoint_sparse_3d_masks,
        cams_ext=cams_ext,
        cams_int=cams_int,
    )

    (
        keypoint_sparse_2d_pixels,
        keypoint_sparse_2d_scores,
        keypoint_sparse_2d_masks,
        keypoint_sparse_3d_poses,
        keypoint_sparse_3d_masks,
    ) = ransac(
        cfg.filters.ransac,
        keypoint_idxs=keypoint_sparse_idxs,
        keypoint_2d_pixels=keypoint_sparse_2d_pixels,
        keypoint_2d_scores=keypoint_sparse_2d_scores,
        keypoint_2d_scales=keypoint_sparse_2d_scales,
        keypoint_2d_masks=keypoint_sparse_2d_masks,
        keypoint_3d_poses=keypoint_sparse_3d_poses,
        keypoint_3d_masks=keypoint_sparse_3d_masks,
        cams_ext=cams_ext,
        cams_int=cams_int,
    )

    l_hand_arctic_keypoint_idxs = [
        "left_wrist",
        "left_index_third_joint",
        "left_index_second_joint",
        "left_index_first_joint",
        "left_middle_third_joint",
        "left_middle_second_joint",
        "left_middle_first_joint",
        "left_pinky_third_joint",
        "left_pinky_second_joint",
        "left_pinky_first_joint",
        "left_ring_third_joint",
        "left_ring_second_joint",
        "left_ring_first_joint",
        "left_thumb_third_joint",
        "left_thumb_second_joint",
        "left_thumb_first_joint",
        "left_thumb_tip",
        "left_index_tip",
        "left_middle_tip",
        "left_ring_tip",
        "left_pinky_tip",
    ]

    r_hand_arctic_keypoint_idxs = [
        "right_wrist",
        "right_index_third_joint",
        "right_index_second_joint",
        "right_index_first_joint",
        "right_middle_third_joint",
        "right_middle_second_joint",
        "right_middle_first_joint",
        "right_pinky_third_joint",
        "right_pinky_second_joint",
        "right_pinky_first_joint",
        "right_ring_third_joint",
        "right_ring_second_joint",
        "right_ring_first_joint",
        "right_thumb_third_joint",
        "right_thumb_second_joint",
        "right_thumb_first_joint",
        "right_thumb_tip",
        "right_index_tip",
        "right_middle_tip",
        "right_ring_tip",
        "right_pinky_tip",
    ]

    l_hand_vitpose_keypoint_idxs = [
        cfg.keypoints.sparse.keypoint_idxs[keypoint]
        for keypoint in l_hand_arctic_keypoint_idxs
    ]
    r_hand_vitpose_keypoint_idxs = [
        cfg.keypoints.sparse.keypoint_idxs[keypoint]
        for keypoint in r_hand_arctic_keypoint_idxs
    ]
    keypoint_sparse_2d_pixels[:, :, l_hand_vitpose_keypoint_idxs] = torch.from_numpy(
        seq_info["data_dict"][seq]["2d"]["joints.left"][mocap_idxs, 1:-1]
    ).to(device=device, dtype=dtype)
    keypoint_sparse_2d_pixels[:, :, r_hand_vitpose_keypoint_idxs] = torch.from_numpy(
        seq_info["data_dict"][seq]["2d"]["joints.right"][mocap_idxs, 1:-1]
    ).to(device=device, dtype=dtype)
    keypoint_sparse_2d_scores[:, :, l_hand_vitpose_keypoint_idxs] = 1
    keypoint_sparse_2d_scores[:, :, r_hand_vitpose_keypoint_idxs] = 1

    # Process 2D dense keypoints
    if keypoint_dense_2d_data is not None:
        keypoint_dense_2d_data = np.stack(
            [
                [
                    keypoint_dense_2d_data[cam].get(
                        image_idx, np.zeros([cfg.keypoints.dense.num_keypoints, 3])
                    )
                    for image_idx in image_idxs
                ]
                for cam in cameras
            ],
            axis=1,
        )
    else:
        keypoint_dense_2d_data = np.zeros(
            [len(image_idxs), len(cameras), cfg.keypoints.dense.num_keypoints, 3]
        )
    keypoint_dense_2d_pixels = torch.tensor(
        keypoint_dense_2d_data[..., :-1], device=device, dtype=dtype
    )

    keypoint_dense_2d_scores = (
        np.exp(-keypoint_dense_2d_data[..., -1])
        / np.exp(-keypoint_dense_2d_data[..., -1]).max(axis=-1, keepdims=True)
        * (keypoint_dense_2d_data[..., 2] > 0)
    )
    keypoint_dense_2d_scores = torch.from_numpy(keypoint_dense_2d_scores).to(
        device=device,
        dtype=dtype,
    )
    keypoint_dense_2d_scores *= (
        keypoint_dense_2d_scores >= cfg.keypoints.dense.min_keypoint_2d_score
    )

    if cfg.problem == "init_single_frame_fitting":
        results = init_single_frame_fitting(
            cfg=cfg,
            frame_idxs=image_idxs,
            cameras=cameras,
            model=model,
            cams_ext=cams_ext,
            cams_int=cams_int,
            keypoint_sparse_2d_info=keypoint_sparse_2d_info,
            keypoint_sparse_2d_pixels=keypoint_sparse_2d_pixels,
            keypoint_sparse_2d_scores=keypoint_sparse_2d_scores,
            keypoint_sparse_2d_masks=keypoint_sparse_2d_masks,
            keypoint_dense_2d_info=keypoint_dense_2d_info,
            keypoint_dense_2d_pixels=keypoint_dense_2d_pixels,
            keypoint_dense_2d_scores=keypoint_dense_2d_scores,
            keypoint_dense_3d_info=keypoint_dense_3d_info,
            keypoint_dense_3d_poses=keypoint_dense_3d_poses,
            keypoint_dense_3d_scores=keypoint_dense_3d_scores,
        )

    result_dir: str = os.path.join(seq_dir, "results", cfg.result_folder)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = os.path.join(result_dir, cfg.result_file)
    with open(result_file, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
