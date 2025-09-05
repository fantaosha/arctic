import torch
from arctic.configs.markers import *
from mesh_fitting.optim.model import FinalATLAS as ATLAS
from mesh_fitting.optim.factors import get_full_keypoint_3d_factors
import numpy as np
from typing import Optional
from mesh_fitting.loss import Loss


def get_torso_full_keypoint_3d_factors(
    model: ATLAS,
    keypoint_dirs: torch.Tensor,
    keypoint_mean: torch.Tensor,
    keypoint_3d_poses: torch.Tensor,
    keypoint_3d_confidence: torch.Tensor,
    joint_indices: np.ndarray,
    joint_weights: torch.Tensor,
    weight: float = 1.0,
    loss_fn: Optional[Loss] = None,
    idx: Optional[int] = 0,
    *args,
    **kwargs,
):
    dtype = keypoint_mean.dtype
    joint_state_t0 = model.lbs_fn.joint_state_t_zero.to(dtype)
    joint_state_r0 = model.lbs_fn.joint_state_r_zero.to(dtype)
    joint_state_s0 = model.lbs_fn.joint_state_s_zero.to(dtype)

    keypoint_idxs = np.array([MARKERS.index(marker) for marker in TORSO_MARKERS])

    factors = {
        f"torso_full_keypoint_3d_factors_{idx}": get_full_keypoint_3d_factors(
            keypoint_3d_data=keypoint_3d_poses[..., keypoint_idxs, :],
            keypoint_3d_confidence=keypoint_3d_confidence[..., keypoint_idxs, :],
            keypoint_dirs=keypoint_dirs[:, keypoint_idxs],
            keypoint_mean=keypoint_mean[keypoint_idxs],
            joint_indices=joint_indices[keypoint_idxs],
            joint_weights=joint_weights[keypoint_idxs],
            joint_state_t0=joint_state_t0,
            joint_state_r0=joint_state_r0,
            joint_state_s0=joint_state_s0,
            weight=weight,
            loss_fn=loss_fn,
        )
    }
    return factors


def get_arm_full_keypoint_3d_factors(
    model: ATLAS,
    keypoint_dirs: torch.Tensor,
    keypoint_mean: torch.Tensor,
    keypoint_3d_poses: torch.Tensor,
    keypoint_3d_confidence: torch.Tensor,
    joint_indices: np.ndarray,
    joint_weights: torch.Tensor,
    weight: float = 1.0,
    loss_fn: Optional[Loss] = None,
    idx: Optional[int] = 0,
    *args,
    **kwargs,
):
    dtype = keypoint_mean.dtype
    joint_state_t0 = model.lbs_fn.joint_state_t_zero.to(dtype)
    joint_state_r0 = model.lbs_fn.joint_state_r_zero.to(dtype)
    joint_state_s0 = model.lbs_fn.joint_state_s_zero.to(dtype)

    keypoint_idxs = np.array(
        [MARKERS.index(marker) for marker in L_ARM_MARKERS + R_ARM_MARKERS]
    )

    factors = {
        f"arm_full_keypoint_3d_factors_{idx}": get_full_keypoint_3d_factors(
            keypoint_3d_data=keypoint_3d_poses[..., keypoint_idxs, :],
            keypoint_3d_confidence=keypoint_3d_confidence[..., keypoint_idxs, :],
            keypoint_dirs=keypoint_dirs[:, keypoint_idxs],
            keypoint_mean=keypoint_mean[keypoint_idxs],
            joint_indices=joint_indices[keypoint_idxs],
            joint_weights=joint_weights[keypoint_idxs],
            joint_state_t0=joint_state_t0,
            joint_state_r0=joint_state_r0,
            joint_state_s0=joint_state_s0,
            weight=weight,
            loss_fn=loss_fn,
        )
    }
    return factors


def get_leg_full_keypoint_3d_factors(
    model: ATLAS,
    keypoint_dirs: torch.Tensor,
    keypoint_mean: torch.Tensor,
    keypoint_3d_poses: torch.Tensor,
    keypoint_3d_confidence: torch.Tensor,
    joint_indices: np.ndarray,
    joint_weights: torch.Tensor,
    weight: float = 1.0,
    loss_fn: Optional[Loss] = None,
    idx: Optional[int] = 0,
    *args,
    **kwargs,
):
    dtype = keypoint_mean.dtype
    joint_state_t0 = model.lbs_fn.joint_state_t_zero.to(dtype)
    joint_state_r0 = model.lbs_fn.joint_state_r_zero.to(dtype)
    joint_state_s0 = model.lbs_fn.joint_state_s_zero.to(dtype)

    keypoint_idxs = np.array(
        [
            MARKERS.index(marker)
            for marker in L_LEG_MARKERS
            + R_LEG_MARKERS
            + L_FOOT_MARKERS
            + R_FOOT_MARKERS
        ]
    )

    factors = {
        f"leg_full_keypoint_3d_factors_{idx}": get_full_keypoint_3d_factors(
            keypoint_3d_data=keypoint_3d_poses[..., keypoint_idxs, :],
            keypoint_3d_confidence=keypoint_3d_confidence[..., keypoint_idxs, :],
            keypoint_dirs=keypoint_dirs[:, keypoint_idxs],
            keypoint_mean=keypoint_mean[keypoint_idxs],
            joint_indices=joint_indices[keypoint_idxs],
            joint_weights=joint_weights[keypoint_idxs],
            joint_state_t0=joint_state_t0,
            joint_state_r0=joint_state_r0,
            joint_state_s0=joint_state_s0,
            weight=weight,
            loss_fn=loss_fn,
        )
    }
    return factors


def get_body_full_keypoint_3d_factors(
    model: ATLAS,
    keypoint_dirs: torch.Tensor,
    keypoint_mean: torch.Tensor,
    keypoint_3d_poses: torch.Tensor,
    keypoint_3d_confidence: torch.Tensor,
    joint_indices: np.ndarray,
    joint_weights: torch.Tensor,
    weight: float = 1.0,
    loss_fn: Optional[Loss] = None,
    idx: Optional[int] = 0,
    *args,
    **kwargs,
):
    dtype = keypoint_mean.dtype
    joint_state_t0 = model.lbs_fn.joint_state_t_zero.to(dtype)
    joint_state_r0 = model.lbs_fn.joint_state_r_zero.to(dtype)
    joint_state_s0 = model.lbs_fn.joint_state_s_zero.to(dtype)

    keypoint_idxs = np.array([MARKERS.index(marker) for marker in BODY_MARKERS])

    factors = {
        f"body_full_keypoint_3d_factors_{idx}": get_full_keypoint_3d_factors(
            keypoint_3d_data=keypoint_3d_poses[..., keypoint_idxs, :],
            keypoint_3d_confidence=keypoint_3d_confidence[..., keypoint_idxs, :],
            keypoint_dirs=keypoint_dirs[:, keypoint_idxs],
            keypoint_mean=keypoint_mean[keypoint_idxs],
            joint_indices=joint_indices[keypoint_idxs],
            joint_weights=joint_weights[keypoint_idxs],
            joint_state_t0=joint_state_t0,
            joint_state_r0=joint_state_r0,
            joint_state_s0=joint_state_s0,
            weight=weight,
            loss_fn=loss_fn,
        )
    }
    return factors


def get_hand_full_keypoint_3d_factors(
    model: ATLAS,
    keypoint_dirs: torch.Tensor,
    keypoint_mean: torch.Tensor,
    keypoint_3d_poses: torch.Tensor,
    keypoint_3d_confidence: torch.Tensor,
    joint_indices: np.ndarray,
    joint_weights: torch.Tensor,
    weight: float = 1.0,
    loss_fn: Optional[Loss] = None,
    idx: Optional[int] = 0,
    *args,
    **kwargs,
):
    dtype = keypoint_mean.dtype
    joint_state_t0 = model.lbs_fn.joint_state_t_zero.to(dtype)
    joint_state_r0 = model.lbs_fn.joint_state_r_zero.to(dtype)
    joint_state_s0 = model.lbs_fn.joint_state_s_zero.to(dtype)

    keypoint_idxs = np.array(
        [MARKERS.index(marker) for marker in L_HAND_MARKERS + R_HAND_MARKERS]
    )

    factors = {
        f"hand_full_keypoint_3d_factors_{idx}": get_full_keypoint_3d_factors(
            keypoint_3d_data=keypoint_3d_poses[..., keypoint_idxs, :],
            keypoint_3d_confidence=keypoint_3d_confidence[..., keypoint_idxs, :],
            keypoint_dirs=keypoint_dirs[:, keypoint_idxs],
            keypoint_mean=keypoint_mean[keypoint_idxs],
            joint_indices=joint_indices[keypoint_idxs],
            joint_weights=joint_weights[keypoint_idxs],
            joint_state_t0=joint_state_t0,
            joint_state_r0=joint_state_r0,
            joint_state_s0=joint_state_s0,
            weight=weight,
            loss_fn=loss_fn,
        )
    }
    return factors
