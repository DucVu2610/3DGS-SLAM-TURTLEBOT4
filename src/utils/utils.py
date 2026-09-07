""" This module contains utility functions used in various parts of the pipeline. """
import copy
import os
import random

import numpy as np
import open3d as o3d
import torch
from gaussian_rasterizer import (GaussianRasterizationSettings,
                                 GaussianRasterizer)
from argparse import ArgumentParser
from src.entities.arguments import OptimizationParams
from src.entities.gaussian_model import GaussianModel
import cv2
from typing import Dict, Any

import time
def find_submap(frame_id: int, submaps: dict) -> dict:
    """ Finds the submap that starts with the given frame ID.
    Args:
        frame_id: The frame ID to search for.
        submaps: The dictionary of submaps to search in.
    Returns:
        The submap that contains the given frame ID.
    """
    for submap in submaps:
        if submap["submap_start_frame_id"] <= frame_id < submap["submap_end_frame_id"]:
            return submap
    return None


def setup_seed(seed: int) -> None:
    """ Sets the seed for generating random numbers to ensure reproducibility across multiple runs.
    Args:
        seed: The seed value to set for random number generators in torch, numpy, and random.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """ Converts a PyTorch tensor to a NumPy ndarray.
    Args:
        tensor: The PyTorch tensor to convert.
    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.clone().detach().cpu().numpy()


def np2torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Converts a NumPy ndarray to a PyTorch tensor.
    Args:
        array: The NumPy ndarray to convert.
        device: The device to which the tensor is sent. Defaults to 'cpu'.

    Returns:
        A PyTorch tensor with the same data as the input array.
    """
    return torch.from_numpy(array).float().to(device)


def np2ptcloud(pts: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    """converts numpy array to point cloud
    Args:
        pts (ndarray): point cloud
    Returns:
        (PointCloud): resulting point cloud
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    return cloud


def get_render_settings(w, h, intrinsics, w2c, near=0.01, far=100, sh_degree=0):
    """
    Constructs and returns a GaussianRasterizationSettings object for rendering,
    configured with given camera parameters.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        intrinsic (array): 3*3, Intrinsic camera matrix.
        w2c (array): World to camera transformation matrix.
        near (float, optional): The near plane for the camera. Defaults to 0.01.
        far (float, optional): The far plane for the camera. Defaults to 100.

    Returns:
        GaussianRasterizationSettings: Configured settings for Gaussian rasterization.
    """
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1,
                                                  1], intrinsics[0, 2], intrinsics[1, 2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    viewmatrix = w2c.transpose(0, 1)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far /
                                    (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]], device='cuda').float().transpose(0, 1)
    full_proj_matrix = viewmatrix.unsqueeze(
        0).bmm(opengl_proj.unsqueeze(0)).squeeze(0)
    return GaussianRasterizationSettings(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], device='cuda').float(),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_matrix,
        sh_degree=sh_degree,
        campos=cam_center,
        prefiltered=False,
        debug=False)


def render_gaussian_model(gaussian_model, render_settings,
                          override_means_3d=None, override_means_2d=None,
                          override_scales=None, override_rotations=None,
                          override_opacities=None, override_colors=None):
    """
    Renders a Gaussian model with specified rendering settings, allowing for
    optional overrides of various model parameters.

    Args:
        gaussian_model: A Gaussian model object that provides methods to get
            various properties like xyz coordinates, opacity, features, etc.
        render_settings: Configuration settings for the GaussianRasterizer.
        override_means_3d (Optional): If provided, these values will override
            the 3D mean values from the Gaussian model.
        override_means_2d (Optional): If provided, these values will override
            the 2D mean values. Defaults to zeros if not provided.
        override_scales (Optional): If provided, these values will override the
            scale values from the Gaussian model.
        override_rotations (Optional): If provided, these values will override
            the rotation values from the Gaussian model.
        override_opacities (Optional): If provided, these values will override
            the opacity values from the Gaussian model.
        override_colors (Optional): If provided, these values will override the
            color values from the Gaussian model.
    Returns:
        A dictionary containing the rendered color, depth, radii, and 2D means
        of the Gaussian model. The keys of this dictionary are 'color', 'depth',
        'radii', and 'means2D', each mapping to their respective rendered values.
    """
    renderer = GaussianRasterizer(raster_settings=render_settings)

    if override_means_3d is None:
        means3D = gaussian_model.get_xyz()
    else:
        means3D = override_means_3d

    if override_means_2d is None:
        means2D = torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        means2D.retain_grad()
    else:
        means2D = override_means_2d

    if override_opacities is None:
        opacities = gaussian_model.get_opacity()
    else:
        opacities = override_opacities

    shs, colors_precomp = None, None
    if override_colors is not None:
        colors_precomp = override_colors
    else:
        shs = gaussian_model.get_features()

    render_args = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "colors_precomp": colors_precomp,
        "shs": shs,
        "scales": gaussian_model.get_scaling() if override_scales is None else override_scales,
        "rotations": gaussian_model.get_rotation() if override_rotations is None else override_rotations,
        "cov3D_precomp": None
    }
    color, depth, alpha, radii = renderer(**render_args)

    return {"color": color, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}


def rgbd2ptcloud(img, depth, intrinsics, pose=np.eye(4)):
    """converts rgbd image to point cloud
    Args:
        img (ndarray): rgb image
        depth (fcndarray): depth map
        intrinsics (ndarray): intrinsics matrix
    Returns:
        (PointCloud): resulting point cloud
    """
    height, width, _ = img.shape
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(img)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
        depth_trunc=100,
    )
    intrinsics = o3d.open3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx=intrinsics[0][0],
        fy=intrinsics[1][1],
        cx=intrinsics[0][2],
        cy=intrinsics[1][2])
    return o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_img, intrinsics, extrinsic=pose, project_valid_depth_only=True)


def ptcloud2numpy(ptcloud: o3d.geometry.PointCloud) -> np.ndarray:
    """converts point cloud to numpy array
    Args:
        ptcloud (PointCloud): point cloud
    Returns:
        (ndarray): resulting numpy array
    """
    if ptcloud.has_colors():
        return np.hstack((np.asarray(ptcloud.points), np.asarray(ptcloud.colors)))
    return np.asarray(ptcloud.points)


def depth2ptcloud(depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray = np.eye(4)) -> o3d.geometry.PointCloud:
    """Converts a depth map to a point cloud.
    Args:
        depth (ndarray): Depth map.
        intrinsics (ndarray): Intrinsics matrix.
        pose (ndarray): Pose matrix. Defaults to identity matrix.
    Returns:
        PointCloud: Resulting point cloud.
    """
    height, width = depth.shape
    depth_img = o3d.geometry.Image(np.ascontiguousarray(depth))
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx=intrinsics[0][0],
        fy=intrinsics[1][1],
        cx=intrinsics[0][2],
        cy=intrinsics[1][2])
    return o3d.geometry.PointCloud.create_from_depth_image(depth_img, intrinsics, extrinsic=pose)

def clone_obj(obj):
    """Deep copy an object, while detaching and cloning all tensors.
    Args:
        obj: The object to clone.
    Returns:
        clone_obj: The cloned object
    """
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        if hasattr(clone_obj.__class__, attr) and isinstance(getattr(clone_obj.__class__, attr), property):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj


def torch2np_decorator(func):
    """A decorator that creates the directory specified in the function's 'directory' keyword
       argument before calling the function.
    Args:
        func: The function to be decorated.
    Returns:
        The wrapper function.
    """
    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(torch2np(arg))
            elif isinstance(arg, dict):
                new_arg = {}
                for k, v in arg.items():
                    new_arg[k] = torch2np(v) if isinstance(v, torch.Tensor) else v
                new_args.append(new_arg)
            else:
                new_args.append(arg)

        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                new_kwargs[k] = torch2np(v)
            elif isinstance(v, dict):
                new_kwargs[k] = {}
                for k1, v1 in v.items():
                    new_kwargs[k][k1] = torch2np(v1) if isinstance(v1, torch.Tensor) else v1
            else:
                new_kwargs[k] = v
        return func(*new_args, *new_kwargs)
    return wrapper

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

def load_3dgs(submap: dict) -> 'GaussianModel':
    """Loads a Gaussian model from a submap dictionary."""
    opt_args = OptimizationParams(ArgumentParser(description="Training script parameters"))
    gaussian = GaussianModel(0)
    gaussian.training_setup(opt_args)
    gaussian.restore_from_params(move_to_device(submap["gaussian_model_params"]), opt_args)
    return gaussian

def render_rgbd_from_3dgs(submap):
    """Renders RGB-D images from a 3D Gaussian model.
    Args:
        gaussian_model (GaussianModel): The 3D Gaussian model to render from.
    Returns:
        dict: A dictionary containing the rendered RGB and depth images.
    """
    # Implement the rendering logic here
    render_settings = get_render_settings(submap["width"], submap["height"], submap["intrinsics"], np.linalg.inv(submap["submap_c2ws"][0]))
    gaussian = load_3dgs(submap)
    render = render_gaussian_model(gaussian, render_settings)
    color = render["color"].squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
    color = (color * 255.0 / color.max()).astype(np.uint8)
    depth = render["depth"].squeeze(0).cpu().detach().numpy()
    return color, depth

def get_rgbd(submap):
    if "start_rgb" in submap and "start_depth" in submap:
        return submap["start_rgb"], submap["start_depth"]
    elif "gaussian_model_params" in submap:
        return render_rgbd_from_3dgs(submap)
    else:
        raise ValueError("No RGB-D data available in the submap.")

def get_pcd_from_rgbd(submap):
    color, depth = get_rgbd(submap)
    return rgbd2ptcloud(color, depth, intrinsics=submap["intrinsics"], pose=np.eye(4))

def coarse_registration(source_cloud, target_cloud) -> np.ndarray:
    """ Coarse registration of point clouds using FPFH features and RANSAC.
    Args:
        source_cloud: The source point cloud.
        target_cloud: The target point cloud.
    Returns:
        The transformation matrix that aligns the source cloud to the target cloud.
    """
    voxel_size = 0.02

    start_time = time.time()
    source_cloud = source_cloud.voxel_down_sample(voxel_size)
    target_cloud = target_cloud.voxel_down_sample(voxel_size)
    end_time = time.time()
    print(f"Downsampling took {end_time - start_time:.2f} seconds")

    # Estimate normals for the point clouds
    start_time = time.time()
    source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    end_time = time.time()
    print(f"Normal estimation took {end_time - start_time:.2f} seconds")

    # Compute FPFH features
    start_time = time.time()
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_cloud, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_cloud, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
    end_time = time.time()
    print(f"FPFH feature computation took {end_time - start_time:.2f} seconds")

    # Global registration with RANSAC
    start_time = time.time()
    coarse_alignment = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_cloud, target_cloud, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    end_time = time.time()
    print(f"Coarse registration took {end_time - start_time:.2f} seconds")
    return coarse_alignment.transformation

def unproject(u, v, depth, intrinsics):
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]
    return [(u - cx) * depth / fx, (v - cy) * depth / fy, depth]

def patch_idx_to_pixel(patch_idx, grid_w, patch_h_px, patch_w_px):
    row, col = patch_idx // grid_w, patch_idx % grid_w
    return int((col + 0.5) * patch_w_px), int((row + 0.5) * patch_h_px)


def _ransac_from_correspondences(src_pts, tgt_pts, distance_threshold):
    """Estimate a rigid transform from paired 3D points and report inliers."""
    src_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(src_pts, dtype=np.float64)))
    tgt_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(tgt_pts, dtype=np.float64)))
    corres = o3d.utility.Vector2iVector(
        np.column_stack((np.arange(len(src_pts)), np.arange(len(src_pts)))).astype(np.int32))
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd, tgt_pcd, corres, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
    return result.transformation, len(result.correspondence_set)


def _valid_depth_point(u, v, depth, intrinsics):
    """Return the back-projected point at a feature location, or None."""
    u, v = int(round(u)), int(round(v))
    if v < 0 or u < 0 or v >= depth.shape[0] or u >= depth.shape[1]:
        return None
    value = float(depth[v, u])
    if not np.isfinite(value) or value <= 0:
        return None
    return unproject(u, v, value, intrinsics)


def _create_local_feature_detector(method, max_keypoints):
    """Create an OpenCV local feature detector and its descriptor norm."""
    method = method.lower()
    if method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("This OpenCV build does not provide SIFT_create().")
        return cv2.SIFT_create(nfeatures=max_keypoints), cv2.NORM_L2
    if method == "orb":
        return cv2.ORB_create(nfeatures=max_keypoints), cv2.NORM_HAMMING
    if method == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    raise ValueError(f"Unsupported local feature registration method: {method}")


def coarse_registration_local_features(source_color, source_depth, target_color, target_depth,
                                       source_intrinsics, target_intrinsics, method,
                                       distance_threshold=0.05, min_correspondences=8,
                                       max_keypoints=4096, ratio_threshold=0.8):
    """Sparse image features -> mutual ratio-test matches -> 3D RANSAC.

    Supported methods are SIFT, ORB, and AKAZE. The returned diagnostics use
    the same field names as the DINOv2 path so experiment results can be
    compared without method-specific parsing.
    """
    detector, norm_type = _create_local_feature_detector(method, max_keypoints)
    source_gray = cv2.cvtColor(source_color, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_color, cv2.COLOR_RGB2GRAY)
    src_kp, src_desc = detector.detectAndCompute(source_gray, None)
    tgt_kp, tgt_desc = detector.detectAndCompute(target_gray, None)

    diagnostics = {
        "num_source_features": len(src_kp),
        "num_target_features": len(tgt_kp),
        "num_feature_matches": 0,
        "num_correspondences": 0,
        "num_ransac_inliers": 0,
    }
    if src_desc is None or tgt_desc is None or len(src_desc) < 2 or len(tgt_desc) < 2:
        return None, diagnostics

    matcher = cv2.BFMatcher(norm_type)

    def ratio_matches(query_desc, train_desc):
        accepted = {}
        for pair in matcher.knnMatch(query_desc, train_desc, k=2):
            if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance:
                accepted[pair[0].queryIdx] = pair[0].trainIdx
        return accepted

    source_to_target = ratio_matches(src_desc, tgt_desc)
    target_to_source = ratio_matches(tgt_desc, src_desc)
    matches = [(source_idx, target_idx)
               for source_idx, target_idx in source_to_target.items()
               if target_to_source.get(target_idx) == source_idx]
    diagnostics["num_feature_matches"] = len(matches)

    src_pts, tgt_pts = [], []
    for source_idx, target_idx in matches:
        source_xy = src_kp[source_idx].pt
        target_xy = tgt_kp[target_idx].pt
        source_point = _valid_depth_point(*source_xy, source_depth, source_intrinsics)
        target_point = _valid_depth_point(*target_xy, target_depth, target_intrinsics)
        if source_point is None or target_point is None:
            continue
        src_pts.append(source_point)
        tgt_pts.append(target_point)

    diagnostics["num_correspondences"] = len(src_pts)
    if len(src_pts) < min_correspondences:
        return None, diagnostics

    transform, num_inliers = _ransac_from_correspondences(
        src_pts, tgt_pts, distance_threshold)
    diagnostics["num_ransac_inliers"] = num_inliers
    return transform, diagnostics


def coarse_registration_dinov2(source_color, source_depth, target_color, target_depth,
                                source_intrinsics, target_intrinsics,
                                feature_extractor, distance_threshold=0.05,
                                min_correspondences=8, return_diagnostics=False):
    """ DINOv2 patch mutual-NN correspondences -> 3D lift via depth -> RANSAC
        (Umeyama/Kabsch on known correspondences).
    Returns:
        (transform, num_correspondences) -- transform is a 4x4 np.ndarray or
        None if there were not enough valid correspondences; num_correspondences
        is always an int so the caller can log why it failed. """
    from PIL import Image
    src_tok, (gh, gw), (ph, pw) = feature_extractor.extract_patch_tokens(Image.fromarray(source_color))
    tgt_tok, _, _ = feature_extractor.extract_patch_tokens(Image.fromarray(target_color))

    diagnostics = {
        "num_source_features": int(src_tok.shape[0]),
        "num_target_features": int(tgt_tok.shape[0]),
        "num_feature_matches": 0,
        "num_correspondences": 0,
        "num_ransac_inliers": 0,
    }

    sim = src_tok @ tgt_tok.T
    s2t = sim.argmax(dim=1)
    t2s = sim.argmax(dim=0)
    mutual = t2s[s2t] == torch.arange(s2t.shape[0], device=sim.device)
    matched_src = torch.nonzero(mutual).squeeze(1)
    diagnostics["num_feature_matches"] = int(matched_src.numel())
    if matched_src.numel() < min_correspondences:
        return (None, diagnostics) if return_diagnostics else (None, int(matched_src.numel()))
    matched_tgt = s2t[matched_src]

    src_pts, tgt_pts = [], []
    for si, ti in zip(matched_src.tolist(), matched_tgt.tolist()):
        u_s, v_s = patch_idx_to_pixel(si, gw, ph, pw)
        u_t, v_t = patch_idx_to_pixel(ti, gw, ph, pw)
        source_point = _valid_depth_point(u_s, v_s, source_depth, source_intrinsics)
        target_point = _valid_depth_point(u_t, v_t, target_depth, target_intrinsics)
        if source_point is None or target_point is None:
            continue
        src_pts.append(source_point)
        tgt_pts.append(target_point)

    diagnostics["num_correspondences"] = len(src_pts)
    if len(src_pts) < min_correspondences:
        return (None, diagnostics) if return_diagnostics else (None, len(src_pts))

    transform, num_inliers = _ransac_from_correspondences(
        src_pts, tgt_pts, distance_threshold)
    diagnostics["num_ransac_inliers"] = num_inliers
    return (transform, diagnostics) if return_diagnostics else (transform, len(src_pts))

def geodesic_rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """ Sign-ambiguity-free rotation error in degrees between two 3x3
        rotation matrices, via the trace of the relative rotation.
        Use this instead of comparing quaternions directly with an L2
        norm -- q and -q represent the same rotation but can have a
        large L2 distance, which silently corrupts quaternion-diff-based
        error metrics. """
    R_err = R_a.T @ R_b
    cos_angle = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def tensor_to_jpeg_bytes_cv2(tensor: torch.Tensor, quality: int = 95) -> bytes:
    """
    Convert RGB tensor to JPEG bytes using OpenCV (fastest method).
    Args:
        tensor: PyTorch tensor of shape (3, H, W) with values in [0, 1]
        quality: JPEG quality (0-100)
    Returns:
        bytes: JPEG-encoded image data
    """
    # Move to CPU if on GPU
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    
    # Convert from (3, H, W) to (H, W, 3) and scale to 0-255
    img_array = tensor.permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Encode as JPEG in memory
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded_img = cv2.imencode('.jpg', img_bgr, encode_params)
    
    if not success:
        raise RuntimeError("Failed to encode image as JPEG")
    
    return encoded_img.tobytes()

def depth_to_compressed_bytes(depth_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Compress depth tensor using PNG encoding in memory.
    Args:
        depth_tensor: PyTorch tensor of shape (H, W) with depth values
    Returns:
        dict: Contains compressed bytes and reconstruction parameters
    """
    if depth_tensor.device.type == 'cuda':
        depth_tensor = depth_tensor.cpu()
    
    depth_array = depth_tensor.numpy()
    
    # Store original range for reconstruction
    depth_min, depth_max = float(depth_array.min()), float(depth_array.max())
    
    # Normalize to 16-bit range for better precision
    if depth_max > depth_min:
        depth_normalized = ((depth_array - depth_min) / (depth_max - depth_min) * 65535).astype(np.uint16)
    else:
        depth_normalized = np.zeros_like(depth_array, dtype=np.uint16)
    
    # Encode as PNG in memory
    success, encoded_depth = cv2.imencode('.png', depth_normalized)
    
    if not success:
        raise RuntimeError("Failed to encode depth as PNG")
    
    return {
        'data': encoded_depth.tobytes(),
        'min_depth': depth_min,
        'max_depth': depth_max,
        'shape': depth_array.shape
    }

# Decoding functions for loading data back
def jpeg_bytes_to_tensor_cv2(jpeg_bytes: bytes) -> torch.Tensor:
    """
    Convert JPEG bytes back to RGB tensor using OpenCV.
    Returns:
        torch.Tensor: RGB tensor of shape (3, H, W) with values in [0, 1]
    """
    # Decode JPEG from bytes
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise RuntimeError("Failed to decode JPEG bytes")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize
    tensor = torch.from_numpy(img_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    
    return tensor

def compressed_bytes_to_depth_tensor(depth_data: Dict[str, Any]) -> torch.Tensor:
    """
    Convert compressed depth bytes back to depth tensor.
    Args:
        depth_data: dict with 'data', 'min_depth', 'max_depth', 'shape'
    Returns:
        torch.Tensor: Depth tensor with original depth values
    """
    # Decode PNG from bytes
    nparr = np.frombuffer(depth_data['data'], np.uint8)
    depth_normalized = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if depth_normalized is None:
        raise RuntimeError("Failed to decode PNG depth bytes")
    
    # Reconstruct original depth values
    depth_min = depth_data['min_depth']
    depth_max = depth_data['max_depth']
    
    if depth_max > depth_min:
        depth_array = (depth_normalized.astype(np.float32) / 65535.0) * (depth_max - depth_min) + depth_min
    else:
        depth_array = np.full(depth_data['shape'], depth_min, dtype=np.float32)
    
    return torch.from_numpy(depth_array)