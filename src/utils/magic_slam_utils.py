""" This module contains utility functions for key components of MAGiC-SLAM pipeline. """
import random

import numpy as np
import open3d as o3d
import roma
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from src.entities.gaussian_model import GaussianModel
from src.entities.losses import l1_loss, ssim
from src.utils import utils
from src.utils.utils import find_submap, np2ptcloud, coarse_registration

from argparse import ArgumentParser
from src.entities.arguments import OptimizationParams

import copy
import time

class Registration(object):
    """ A class to store the registration information between two submaps. """
    def __init__(self, source_agent_id: int, source_frame_id: int, target_agent_id: int, target_frame_id: int) -> None:
        self.source_agent_id = source_agent_id
        self.source_frame_id = source_frame_id
        self.target_agent_id = target_agent_id
        self.target_frame_id = target_frame_id
        self.source_transformation = np.eye(4)
        self.target_transformation = np.eye(4)
        self.init_transformation = np.eye(4)
        self.transformation = np.eye(4)
        self.inlier_rmse = 100.0
        self.fitness = 0.0
        self.registration_method_requested = None
        self.coarse_method_used = None
        self.fallback_used = False
        self.coarse_registration_failed = False
        self.num_source_features = None
        self.num_target_features = None
        self.num_feature_matches = None
        self.num_mutual_matches = None
        self.num_confidence_matches = None
        self.num_depth_valid_correspondences = None
        self.num_geometry_consistent_correspondences = None
        self.num_correspondences = None
        self.num_ransac_inliers = None
        self.ransac_inlier_ratio = None
        self.correspondence_retention_ratio = None
        self.mean_match_similarity = None
        self.mean_match_margin = None
        self.match_similarity_p10 = None
        self.match_similarity_p50 = None
        self.match_similarity_p90 = None
        self.match_margin_p10 = None
        self.match_margin_p50 = None
        self.match_margin_p90 = None
        self.mean_compatibility_support = None
        self.dino_min_similarity = None
        self.dino_min_margin = None
        self.dino_keep_top_fraction = None
        self.dino_depth_window_radius = None
        self.dino_depth_max_mad_m = None
        self.dino_compatibility_threshold_m = None
        self.dino_min_compatibility_support = None
        self.dino_preprocess_mode = None
        self.source_patch_input_height = None
        self.source_patch_input_width = None
        self.source_patch_grid_height = None
        self.source_patch_grid_width = None
        self.target_patch_input_height = None
        self.target_patch_input_width = None
        self.target_patch_grid_height = None
        self.target_patch_grid_width = None
        self.num_gaussians_total = None
        self.num_gaussians_opacity_valid = None
        self.num_gaussians_in_view = None
        self.num_gaussians_depth_consistent = None
        self.num_gaussian_landmarks = None
        self.mean_gaussian_landmark_opacity = None
        self.mean_gaussian_depth_residual_m = None
        self.gaussian_min_opacity = None
        self.gaussian_depth_tolerance_m = None
        self.gaussian_max_landmarks = None
        self.gaussian_refit_inliers = None
        self.coarse_registration_time = None
        self.icp_registration_time = None
        self.coarse_transformation = None


def refine_map(gaussian_model, agents_datasets: dict, agents_keyframe_ids: dict, agents_c2ws: dict, iterations=3000):
    """ Refine a Gaussian model using keyframes from agents' datasets: Section 3.4
    Args:
        gaussian_model: The Gaussian model.
        agents_datasets: The agents' datasets (agent_id: str -> dataset : Dataset)
        agents_keyframe_ids: The agents' keyframe IDs (agent_id: str -> keyframe_ids : np.ndarray)
        agents_c2ws: The agents' camera-to-world matrices (agent_id: str -> c2ws : np.ndarray)
        iterations: The number of iterations to refine the map.
    Returns:
        gaussian_model: The refined Gaussian model.
    """
    print("Refining map")
    for iteration in tqdm(range(iterations)):
        agent_id = random.choice(list(agents_datasets.keys()))
        sample_id = random.randint(0, len(agents_keyframe_ids[agent_id]) - 1)
        render_data = agents_datasets[agent_id].get_render_frame(
            agents_keyframe_ids[agent_id][sample_id],
            np.linalg.inv(agents_c2ws[agent_id][sample_id]))
        gt_color, gt_depth = render_data["gt_color"], render_data["gt_depth"]

        render_dict = utils.render_gaussian_model(gaussian_model, render_data["render_settings"])
        color_loss = 0.8 * l1_loss(render_dict["color"], gt_color) + 0.2 * (1.0 - ssim(render_dict["color"], gt_color))

        depth_lss = l1_loss(render_dict["depth"], gt_depth)

        loss = color_loss + depth_lss

        loss.backward()

        with torch.no_grad():
            radii = render_dict["radii"]
            visibility_filter = radii > 0
            gaussian_model.max_radii2D[visibility_filter] = torch.max(
                gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])

            if iteration > iterations * 0.2 and iteration < iterations * 0.8:
                prune_mask = (gaussian_model.get_opacity() < 0.005).squeeze()
                gaussian_model.prune_points(prune_mask)

            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad(set_to_none=True)
            # gaussian_model.update_learning_rate(iteration)
    return gaussian_model


def merge_submaps(agents_submaps: dict, agents_kf_ids: dict, agents_opt_kf_c2ws: dict, opt_args):
    """ Merge submaps from agents in a coarse manner: Section 3.4
    Args:
        agents_submaps: A dictionary of agent submaps.
        agents_kf_ids: A dictionary of agent keyframe IDs.
        agents_opt_kf_c2ws: A dictionary of agent optimized camera-to-world matrices.
        opt_args: The optimization arguments.
    Returns:
        merged_map: The merged Gaussian model.
    """
    merged_map = GaussianModel(0)
    merged_map.training_setup(opt_args)
    device = "cuda"

    print("Merging submaps")
    for agent_id in tqdm(sorted(agents_submaps.keys())):

        for _, submap in enumerate(agents_submaps[agent_id][::-1]):

            xyz = submap["gaussian_model_params"]["xyz"].to(device)
            rotations = submap["gaussian_model_params"]["rotation"].to(device)
            features_dc = submap["gaussian_model_params"]["features_dc"].to(device)
            features_rest = submap["gaussian_model_params"]["features_rest"].to(device)
            opacity = submap["gaussian_model_params"]["opacity"].to(device)
            scaling = submap["gaussian_model_params"]["scaling"].to(device)

            kf_mask = agents_kf_ids[agent_id] == submap["submap_start_frame_id"]
            submap_opt_c2w = agents_opt_kf_c2ws[agent_id][kf_mask][0]
            submap_c2w = submap["submap_c2ws"][0]
            delta = utils.np2torch(submap_opt_c2w @ np.linalg.inv(submap_c2w), device=device)

            opt_xyz = xyz @ delta[:3, :3].T + delta[:3, 3]

            rotation_matrices = roma.unitquat_to_rotmat(rotations)
            opt_rotations = delta[:3, :3][None] @ rotation_matrices
            opt_rotations = roma.rotmat_to_unitquat(opt_rotations)

            merged_map.densification_postfix(
                opt_xyz,
                features_dc,
                features_rest,
                opacity,
                scaling,
                opt_rotations)

    return merged_map


def apply_pose_correction(submap_optimized_poses: dict, agents_submaps: dict) -> dict:
    """ Applies the pose correction to the keyframes of the agents.
        The pose graph consists of the first poses of each sub-map.
        After optimization, the correction between the optimized pose of a sub-map
        is applied to all the keyframe poses within that sub-map.
    Args:
        submap_optimized_poses: The optimized poses of the submaps.
        agents_submaps: The submaps of the agents.
    Returns:
        agents_corrected_kf_poses: A dictionary containing the corrected keyframe poses of the agents
    """
    agents_corrected_kf_poses = {}
    for agent_id in sorted(agents_submaps.keys()):
        opt_kf_poses = []
        for i, submap in enumerate(agents_submaps[agent_id]):
            submap_kf_ids = submap["keyframe_ids"] - submap["submap_start_frame_id"]
            delta = submap_optimized_poses[agent_id][i] @ np.linalg.inv(submap["submap_c2ws"][0])
            for pose in submap["submap_c2ws"][submap_kf_ids]:
                opt_kf_poses.append(delta @ pose)
        agents_corrected_kf_poses[agent_id] = np.array(opt_kf_poses)
    return agents_corrected_kf_poses


def register_submaps(agents_submaps: dict, registration: Registration):
    """ Register two submaps using ICP.
    Args:
        agents_submaps: A dictionary of agent submaps.
        registration: The registration object.
    Returns:
        registration: The registration object with the transformation and fitness updated.
    """

    source_cloud = agents_submaps[registration.source_agent_id]
    source_submap = find_submap(registration.source_frame_id, agents_submaps[registration.source_agent_id])
    target_submap = find_submap(registration.target_frame_id, agents_submaps[registration.target_agent_id])

    source_cloud = np2ptcloud(source_submap["point_cloud"][:, :3], source_submap["point_cloud"][:, 3:] / 255.0)
    target_cloud = np2ptcloud(target_submap["point_cloud"][:, :3], target_submap["point_cloud"][:, 3:] / 255.0)

    source_cloud.estimate_normals()
    target_cloud.estimate_normals()

    distance_threshold = 0.005
    fine_alignment = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, distance_threshold, registration.init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500,
                                                          relative_fitness=1e-9, relative_rmse=1e-9))

    registration.transformation = fine_alignment.transformation
    registration.fitness = fine_alignment.fitness
    registration.inlier_rmse = fine_alignment.inlier_rmse

    return registration


def register_agents_submaps(agents_submaps: dict, registrations: list,
                            registration_method, max_threads: int = 5) -> list:
    """ Register the submaps of the agents in parallel.
    Args:
        agents_submaps: A dictionary of agent submaps with (agent_id: int -> submaps: list)
        registrations: A list of Registration objects.
        registration_method: The registration method to use.
        max_threads: The maximum number of threads to use.
    """
    registrations = Parallel(n_jobs=max_threads)(delayed(registration_method)(
        agents_submaps, registration) for registration in registrations)
    return registrations

def register_submaps_depth(agents_submaps: dict, registration: Registration,
                           initial_transformation_unknown: bool = True,
                           registration_method: str = "fpfh",
                           feature_extractor=None,
                           fallback_to_fpfh: bool = True,
                           registration_options: dict = None):
    """ Register two submaps using ICP.
    Args:
        agents_submaps: A dictionary of agent submaps.
        registration: The registration object.
        initial_transformation_unknown: If True, use a coarse registration as the
            initial guess for inter-agent loops. If False, the odometry-based init_transformation
            set by detect_loops() is used directly (assumes a known relative pose between agents).
        registration_method: "fpfh", "dinov2", "gaussian_landmark", "sift",
            "orb", or "akaze"
            for the coarse registration used when initial_transformation_unknown is True.
        feature_extractor: DINOv2 feature extractor, required by "dinov2" and
            "gaussian_landmark".
        fallback_to_fpfh: If True, use FPFH when a visual method produces too
            few valid correspondences. Disable this for pure extractor benchmarks.
        registration_options: Shared RANSAC/correspondence settings plus sparse
            local-feature settings. Defaults preserve the original DINOv2 path.
    Returns:
        registration: The registration object with the transformation and fitness updated.
    """

    source_submap = find_submap(registration.source_frame_id, agents_submaps[registration.source_agent_id])
    target_submap = find_submap(registration.target_frame_id, agents_submaps[registration.target_agent_id])

    source_color, source_depth = utils.get_rgbd(source_submap)
    target_color, target_depth = utils.get_rgbd(target_submap)
    source_cloud = utils.rgbd2ptcloud(source_color, source_depth, source_submap["intrinsics"], np.eye(4))
    target_cloud = utils.rgbd2ptcloud(target_color, target_depth, target_submap["intrinsics"], np.eye(4))

    supported_methods = {
        "fpfh", "dinov2", "gaussian_landmark", "sift", "orb", "akaze"
    }
    registration_method = registration_method.lower()
    if registration_method not in supported_methods:
        raise ValueError(
            f"Unknown registration_method={registration_method!r}; "
            f"expected one of {sorted(supported_methods)}")
    registration.registration_method_requested = registration_method
    registration_options = registration_options or {}
    min_correspondences = registration_options.get("min_correspondences", 8)
    ransac_distance_threshold = registration_options.get(
        "ransac_distance_threshold", 0.05)

    if source_submap['agent_id'] != target_submap['agent_id'] and initial_transformation_unknown:
        coarse_start = time.perf_counter()
        transform = None
        diagnostics = {}
        if registration_method == "dinov2":
            if feature_extractor is None or not hasattr(feature_extractor, "extract_patch_tokens"):
                raise ValueError("DINOv2 registration requires an extractor with extract_patch_tokens().")
            transform, diagnostics = utils.coarse_registration_dinov2(
                source_color, source_depth, target_color, target_depth,
                source_submap["intrinsics"], target_submap["intrinsics"], feature_extractor,
                distance_threshold=ransac_distance_threshold,
                min_correspondences=min_correspondences,
                return_diagnostics=True,
                min_similarity=registration_options.get(
                    "dino_min_similarity", -1.0),
                min_margin=registration_options.get("dino_min_margin", -1.0),
                keep_top_fraction=registration_options.get(
                    "dino_keep_top_fraction", 1.0),
                depth_window_radius=registration_options.get(
                    "dino_depth_window_radius", 0),
                depth_max_mad_m=registration_options.get(
                    "dino_depth_max_mad_m"),
                compatibility_threshold_m=registration_options.get(
                    "dino_compatibility_threshold_m", 0.0),
                min_compatibility_support=registration_options.get(
                    "dino_min_compatibility_support", 0))
        elif registration_method == "gaussian_landmark":
            if (feature_extractor is None or
                    not hasattr(feature_extractor, "extract_patch_tokens")):
                raise ValueError(
                    "Gaussian-landmark registration requires an extractor "
                    "with extract_patch_tokens().")
            target_gaussian = utils.load_3dgs(target_submap)
            try:
                transform, diagnostics = (
                    utils.coarse_registration_gaussian_landmark(
                        source_color=source_color,
                        source_depth=source_depth,
                        target_color=target_color,
                        target_depth=target_depth,
                        source_intrinsics=source_submap["intrinsics"],
                        target_intrinsics=target_submap["intrinsics"],
                        target_gaussian_model=target_gaussian,
                        target_c2w=np.asarray(target_submap["submap_c2ws"][0]),
                        feature_extractor=feature_extractor,
                        distance_threshold=ransac_distance_threshold,
                        min_correspondences=min_correspondences,
                        return_diagnostics=True,
                        min_similarity=registration_options.get(
                            "dino_min_similarity", -1.0),
                        min_margin=registration_options.get(
                            "dino_min_margin", -1.0),
                        keep_top_fraction=registration_options.get(
                            "dino_keep_top_fraction", 1.0),
                        depth_window_radius=registration_options.get(
                            "dino_depth_window_radius", 0),
                        depth_max_mad_m=registration_options.get(
                            "dino_depth_max_mad_m"),
                        compatibility_threshold_m=registration_options.get(
                            "dino_compatibility_threshold_m", 0.0),
                        min_compatibility_support=registration_options.get(
                            "dino_min_compatibility_support", 0),
                        gaussian_min_opacity=registration_options.get(
                            "gaussian_min_opacity", 0.05),
                        gaussian_depth_tolerance_m=registration_options.get(
                            "gaussian_depth_tolerance_m", 0.05),
                        gaussian_max_landmarks=registration_options.get(
                            "gaussian_max_landmarks", 2048),
                        gaussian_refit_inliers=registration_options.get(
                            "gaussian_refit_inliers", True),
                    )
                )
            finally:
                del target_gaussian
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        elif registration_method in {"sift", "orb", "akaze"}:
            transform, diagnostics = utils.coarse_registration_local_features(
                source_color, source_depth, target_color, target_depth,
                source_submap["intrinsics"], target_submap["intrinsics"],
                method=registration_method,
                distance_threshold=ransac_distance_threshold,
                min_correspondences=min_correspondences,
                max_keypoints=registration_options.get("max_keypoints", 4096),
                ratio_threshold=registration_options.get("ratio_threshold", 0.8))

        for field, value in diagnostics.items():
            setattr(registration, field, value)
        min_ransac_inliers = registration_options.get("min_ransac_inliers", 3)
        if (transform is not None and diagnostics and
                diagnostics["num_ransac_inliers"] < min_ransac_inliers):
            transform = None
        if diagnostics and registration_method in {
                "dinov2", "gaussian_landmark"}:
            print(f"[registration] {registration_method} correspondence stages: "
                  f"{diagnostics['num_mutual_matches']} mutual -> "
                  f"{diagnostics['num_confidence_matches']} confidence -> "
                  f"{diagnostics['num_depth_valid_correspondences']} depth -> "
                  f"{diagnostics['num_geometry_consistent_correspondences']} geometry -> "
                  f"{diagnostics['num_ransac_inliers']} RANSAC inliers "
                  f"(agent {source_submap['agent_id']} <-> {target_submap['agent_id']})")
        elif diagnostics:
            print(f"[registration] {registration_method} attempt: "
                  f"{diagnostics['num_feature_matches']} feature matches, "
                  f"{diagnostics['num_correspondences']} depth-valid correspondences, "
                  f"{diagnostics['num_ransac_inliers']} RANSAC inliers "
                  f"(agent {source_submap['agent_id']} <-> {target_submap['agent_id']})")
        if transform is None:
            if registration_method == "fpfh" or fallback_to_fpfh:
                transform = utils.coarse_registration(source_cloud, target_cloud)
                registration.coarse_method_used = "fpfh"
                registration.fallback_used = registration_method != "fpfh"
            else:
                registration.coarse_registration_failed = True
                registration.coarse_registration_time = time.perf_counter() - coarse_start
                return registration
        else:
            registration.coarse_method_used = registration_method
        registration.coarse_registration_time = time.perf_counter() - coarse_start
        registration.init_transformation = transform
        registration.coarse_transformation = np.asarray(transform).copy()

    source_cloud.estimate_normals()
    target_cloud.estimate_normals()

    distance_threshold = 0.005
    icp_start = time.perf_counter()
    fine_alignment = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, distance_threshold, registration.init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500,
                                                          relative_fitness=1e-9, relative_rmse=1e-9))
    registration.icp_registration_time = time.perf_counter() - icp_start

    # draw_registration_result(source_cloud, target_cloud, fine_alignment.transformation)

    registration.transformation = fine_alignment.transformation
    registration.fitness = fine_alignment.fitness
    registration.inlier_rmse = fine_alignment.inlier_rmse
    # registration.info_matrix = get_information_matrix(
    #    source_cloud, target_cloud, distance_threshold, registration.transformation, fine_alignment.fitness, threshold=0.5)

    return registration

def move_to_device(obj, device="cuda"):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    elif isinstance(obj, set):
        return {move_to_device(v, device) for v in obj}
    else:
        return obj # Leave non-tensor types unchanged

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def get_information_matrix(source_cloud, target_cloud, distance_threshold, transformation, fitness, threshold=0.5):
    """ Compute the information matrix for the registration.
    Args:
        source_cloud: The source point cloud.
        target_cloud: The target point cloud.
        transformation: The transformation matrix.
    Returns:
        information_matrix: The information matrix.
    """
    if not isinstance(source_cloud, o3d.t.geometry.PointCloud):
        source_cloud_t = o3d.t.geometry.PointCloud.from_legacy(source_cloud)
    else:
        source_cloud_t = source_cloud

    if not isinstance(target_cloud, o3d.t.geometry.PointCloud):
        target_cloud_t = o3d.t.geometry.PointCloud.from_legacy(target_cloud)
    else:
        target_cloud_t = target_cloud
    # Now get_information_matrix will work
    if fitness > threshold:
        info_matrix = o3d.t.pipelines.registration.get_information_matrix(
            source_cloud_t, target_cloud_t, distance_threshold, transformation)
    else:
        info_matrix = o3d.core.Tensor.zeros((6, 6), dtype=o3d.core.float64)
    
    # Reorder from Open3D [tx,ty,tz,rx,ry,rz] to GTSAM [rx,ry,rz,tx,ty,tz]
    reorder_indices = [3, 4, 5, 0, 1, 2]
    info_matrix = info_matrix[np.ix_(reorder_indices, reorder_indices)].numpy()

    return info_matrix

def register_agents_submaps_depth(agents_submaps: dict, registrations: list,
                            registration_method, max_threads: int = 1)-> list:
    """ Args:
        agents_submaps: A dictionary of agent submaps with (agent_id: int -> submaps: list)
        registrations: A list of Registration objects.
        registration_method: The registration method to use.
        intrinsics: The camera intrinsics to use for depth registration.
    """
    if max_threads < 2:
        registrations = [registration_method(agents_submaps, registration) for registration in registrations]
    else:
       registrations = Parallel(n_jobs=max_threads)(delayed(registration_method)(
           agents_submaps, registration) for registration in registrations)
    return registrations
