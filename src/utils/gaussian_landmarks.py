"""Utilities for building appearance-aware landmarks from a 3DGS submap.

The first implementation deliberately uses a single reference view.  It
projects optimized Gaussian centres into that view, associates at most one
visible/depth-consistent Gaussian with each DINO patch, and attaches the patch
descriptor to the selected 3D centre.  This keeps the coordinate path easy to
validate before adding multi-view descriptor aggregation.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch


def _as_tensor(value, *, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype)


def build_single_view_gaussian_landmarks(
    gaussian_model,
    reference_depth: np.ndarray,
    intrinsics: np.ndarray,
    reference_c2w: np.ndarray,
    patch_tokens: torch.Tensor,
    patch_grid_shape: Tuple[int, int],
    min_opacity: float = 0.05,
    depth_tolerance_m: float = 0.05,
    max_landmarks: int = 2048,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Attach reference-view patch descriptors to selected Gaussian centres.

    Returned points are expressed in the reference camera frame.  Therefore
    they can be paired directly with RGB-D points lifted from another camera
    and passed to the existing source-to-target RANSAC estimator.
    """
    if not 0.0 <= float(min_opacity) <= 1.0:
        raise ValueError("min_opacity must be in [0, 1].")
    if float(depth_tolerance_m) <= 0.0:
        raise ValueError("depth_tolerance_m must be positive.")
    if int(max_landmarks) < 3:
        raise ValueError("max_landmarks must be at least 3.")

    depth = np.asarray(reference_depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"reference_depth must be HxW, received {depth.shape}.")
    height, width = depth.shape

    grid_h, grid_w = (int(patch_grid_shape[0]), int(patch_grid_shape[1]))
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("patch_grid_shape must contain positive dimensions.")
    if patch_tokens.ndim != 2 or patch_tokens.shape[0] != grid_h * grid_w:
        raise ValueError(
            "patch token count does not match the supplied patch grid: "
            f"{tuple(patch_tokens.shape)} versus {grid_h}x{grid_w}."
        )

    xyz_world = gaussian_model.get_xyz().detach()
    opacity = gaussian_model.get_opacity().detach().reshape(-1)
    if xyz_world.ndim != 2 or xyz_world.shape[1] != 3:
        raise ValueError("Gaussian centres must have shape [N, 3].")
    if opacity.shape[0] != xyz_world.shape[0]:
        raise ValueError("Gaussian opacity and centre arrays have different lengths.")

    diagnostics = {
        "num_gaussians_total": int(xyz_world.shape[0]),
        "num_gaussians_opacity_valid": 0,
        "num_gaussians_in_view": 0,
        "num_gaussians_depth_consistent": 0,
        "num_gaussian_landmarks": 0,
        "mean_gaussian_landmark_opacity": None,
        "mean_gaussian_depth_residual_m": None,
    }
    if xyz_world.shape[0] == 0:
        empty_desc = patch_tokens.new_empty((0, patch_tokens.shape[1]))
        return empty_desc, np.empty((0, 3)), np.empty(0, np.int64), diagnostics

    device, dtype = xyz_world.device, xyz_world.dtype
    K = _as_tensor(intrinsics, device=device, dtype=dtype)
    c2w = _as_tensor(reference_c2w, device=device, dtype=dtype)
    if K.shape != (3, 3) or c2w.shape != (4, 4):
        raise ValueError("intrinsics and reference_c2w must be 3x3 and 4x4.")

    world_to_camera = torch.linalg.inv(c2w)
    xyz_h = torch.cat(
        (xyz_world, torch.ones((xyz_world.shape[0], 1), device=device, dtype=dtype)),
        dim=1,
    )
    xyz_camera = (world_to_camera @ xyz_h.T).T[:, :3]
    z = xyz_camera[:, 2]

    finite = torch.isfinite(xyz_camera).all(dim=1) & torch.isfinite(opacity)
    opacity_valid = finite & (opacity >= float(min_opacity))
    diagnostics["num_gaussians_opacity_valid"] = int(opacity_valid.sum().item())

    safe_z = torch.where(z.abs() > 1e-12, z, torch.ones_like(z))
    u = K[0, 0] * xyz_camera[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * xyz_camera[:, 1] / safe_z + K[1, 2]
    ui = torch.round(u).long()
    vi = torch.round(v).long()
    in_view = (
        opacity_valid
        & (z > 0.0)
        & (ui >= 0)
        & (ui < width)
        & (vi >= 0)
        & (vi < height)
    )
    diagnostics["num_gaussians_in_view"] = int(in_view.sum().item())
    if not bool(in_view.any()):
        empty_desc = patch_tokens.new_empty((0, patch_tokens.shape[1]))
        return empty_desc, np.empty((0, 3)), np.empty(0, np.int64), diagnostics

    depth_tensor = _as_tensor(depth, device=device, dtype=dtype)
    sampled_depth = torch.zeros_like(z)
    sampled_depth[in_view] = depth_tensor[vi[in_view], ui[in_view]]
    depth_residual = torch.abs(sampled_depth - z)
    depth_consistent = (
        in_view
        & torch.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
        & (depth_residual <= float(depth_tolerance_m))
    )
    diagnostics["num_gaussians_depth_consistent"] = int(
        depth_consistent.sum().item()
    )
    candidate_indices = torch.nonzero(depth_consistent).squeeze(1)
    if candidate_indices.numel() == 0:
        empty_desc = patch_tokens.new_empty((0, patch_tokens.shape[1]))
        return empty_desc, np.empty((0, 3)), np.empty(0, np.int64), diagnostics

    patch_rows = torch.clamp(
        torch.floor(v[candidate_indices] * grid_h / height).long(), 0, grid_h - 1
    )
    patch_cols = torch.clamp(
        torch.floor(u[candidate_indices] * grid_w / width).long(), 0, grid_w - 1
    )
    patch_ids = patch_rows * grid_w + patch_cols

    # Pick the closest-to-rendered-depth Gaussian in each patch; opacity is a
    # deterministic tie breaker.  One landmark per patch prevents duplicated
    # descriptors from destabilising mutual-nearest-neighbour matching.
    candidate_cpu = candidate_indices.detach().cpu().numpy()
    patch_cpu = patch_ids.detach().cpu().numpy()
    residual_cpu = depth_residual[candidate_indices].detach().cpu().numpy()
    opacity_cpu = opacity[candidate_indices].detach().cpu().numpy()
    best_by_patch = {}
    for local_index, patch_id in enumerate(patch_cpu.tolist()):
        score = (float(residual_cpu[local_index]), -float(opacity_cpu[local_index]))
        previous = best_by_patch.get(int(patch_id))
        if previous is None or score < previous[0]:
            best_by_patch[int(patch_id)] = (score, local_index)

    selected_local = [item[1] for item in best_by_patch.values()]
    selected_local.sort(
        key=lambda index: (float(residual_cpu[index]), -float(opacity_cpu[index]))
    )
    selected_local = selected_local[: int(max_landmarks)]
    selected_gaussian = candidate_cpu[selected_local]
    selected_patch = patch_cpu[selected_local].astype(np.int64, copy=False)

    selected_gaussian_t = torch.as_tensor(
        selected_gaussian, device=device, dtype=torch.long
    )
    descriptor_indices = torch.as_tensor(
        selected_patch, device=patch_tokens.device, dtype=torch.long
    )
    descriptors = patch_tokens[descriptor_indices]
    points_camera = xyz_camera[selected_gaussian_t].detach().cpu().numpy()

    diagnostics["num_gaussian_landmarks"] = int(len(selected_local))
    diagnostics["mean_gaussian_landmark_opacity"] = float(
        opacity[selected_gaussian_t].mean().item()
    )
    diagnostics["mean_gaussian_depth_residual_m"] = float(
        depth_residual[selected_gaussian_t].mean().item()
    )
    return descriptors, points_camera, selected_patch, diagnostics
