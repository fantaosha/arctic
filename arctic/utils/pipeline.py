from omegaconf import DictConfig
import os
from typing import List, Dict, Optional, Any, Union
import pandas as pd
import torch
import numpy as np
from mesh_fitting.optim.model import FinalATLAS
from mesh_fitting.utils.pipeline import optimize_stages


def optimize_frames(
    cfg: DictConfig,
    frame_idxs: np.ndarray,
    model: FinalATLAS,
    cams_ext: torch.Tensor,
    cams_int: torch.Tensor,
    skel_params: torch.Tensor,
    shape_params: torch.Tensor,
    opt_params: Optional[torch.Tensor],
    opt_states: Optional[Dict[str, Any]],
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
    results: Dict[str, Dict[int, np.ndarray]],
):
    variables = {
        "cameras_ext": cams_ext,
        "cameras_int": cams_int,
        "skel_params": skel_params,
        "shape_params": shape_params,
    }

    opt_info = {}
    opt_info["keypoint_sparse_2d"] = keypoint_sparse_2d_info
    opt_info["keypoint_dense_2d"] = keypoint_dense_2d_info
    opt_info["keypoint_dense_3d"] = keypoint_dense_3d_info

    opt_data = {}
    opt_data["keypoint_sparse_2d"] = {
        "keypoint_2d_pixels": keypoint_sparse_2d_pixels,
        "keypoint_2d_scores": keypoint_sparse_2d_scores,
        "keypoint_2d_masks": keypoint_sparse_2d_masks,
    }
    opt_data["keypoint_dense_2d"] = {
        "keypoint_2d_pixels": keypoint_dense_2d_pixels,
        "keypoint_2d_scores": keypoint_dense_2d_scores,
    }
    opt_data["keypoint_dense_3d"] = {
        "keypoint_3d_poses": keypoint_dense_3d_poses,
        "keypoint_3d_scores": keypoint_dense_3d_scores,
    }

    for stage, cfg_stage in cfg.items():
        variables, opt_params, opt_states = optimize_stages(
            cfg_stage,
            model=model,
            variables=variables,
            opt_params=opt_params,
            opt_states=opt_states,
            opt_info=opt_info,
            opt_data=opt_data,
        )

    ret_vars = {
        var_name: var_data.cpu().numpy() for var_name, var_data in variables.items()
    }
    ret_params = {
        param_name: param_data.cpu().numpy()
        for param_name, param_data in opt_params.items()
    }

    results["all"].update(
        {
            frame_idx: {
                "camera_ext": ret_vars["cameras_ext"][n],
                "camera_int": ret_vars["cameras_int"][n],
                "skel_params": ret_vars["skel_params"][n],
                "shape_params": ret_vars["shape_params"][n],
                "opt_params": {key: val[n] for key, val in ret_params.items()},
                "param_transform": opt_states["param_transform"],
            }
            for n, frame_idx in enumerate(frame_idxs)
        }
    )

    return results
