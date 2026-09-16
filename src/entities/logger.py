""" This module includes the Logger class, which is responsible for logging for both Mapper and the Tracker """
import csv
import pickle
from io import BytesIO
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image

from src.utils.io_utils import get_gpu_usage_by_id, save_dict_to_json
from src.utils.tracking_eval import align_trajectories, compute_ate, pose_error
from src.utils.utils import find_submap, geodesic_rotation_error_deg, torch2np_decorator
from src.utils.vis_utils import plot_trajectory


class Logger(object):

    def __init__(self, output_path: Union[Path, str], agent_id=0, use_wandb=False) -> None:
        self.output_path = Path(output_path)
        self.agent_id = agent_id
        self.use_wandb = use_wandb

    def log_tracking_iteration(self, frame_id, cur_pose, gt_quat, gt_trans, total_loss,
                               color_loss, depth_loss, iter, num_iters,
                               print_output=False, print_wandb=False) -> None:
        """ Logs tracking iteration metrics including pose error, losses, and optionally reports to Weights & Biases.
        Logs the error between the current pose estimate and ground truth quaternion and translation,
        as well as various loss metrics. Can output to wandb if enabled and specified, and print to console.
        Args:
            frame_id: Identifier for the current frame.
            cur_pose: The current estimated pose as a tensor (quaternion + translation).
            gt_quat: Ground truth quaternion.
            gt_trans: Ground truth translation.
            total_loss: Total computed loss for the current iteration.
            color_loss: Computed color loss for the current iteration.
            depth_loss: Computed depth loss for the current iteration.
            iter: The current iteration number.
            num_iters: The total number of iterations planned.
            wandb_output: Whether to output the log to wandb.
            print_output: Whether to print the log output.
        """
        prefix = f"Agent_{self.agent_id}_tracking"
        quad_err = torch.abs(cur_pose[:4] - gt_quat).mean().item() * 100  # in degrees
        trans_err = torch.abs(cur_pose[4:] - gt_trans).mean().item() * 100  # in cm
        if self.use_wandb and print_wandb:
            wandb.log(
                {
                    f"{prefix}/idx": frame_id,
                    f"{prefix}/cam_quad_err": quad_err,
                    f"{prefix}/cam_position_err": trans_err,
                    f"{prefix}/total_loss": total_loss.item(),
                    f"{prefix}/color_loss": color_loss.item(),
                    f"{prefix}/depth_loss": depth_loss.item(),
                    f"{prefix}/num_iters": num_iters,
                })
        if iter == num_iters - 1:
            msg = f"frame_id: {frame_id}, cam_quad_err: {quad_err:.5f}, cam_trans_err: {trans_err:.5f} "
        else:
            msg = f"iter: {iter}, total_loss: {total_loss.item():.5f}, color_loss: {color_loss.item():.5f}, depth_loss: {depth_loss.item():.5f} "
        msg = msg + f", cam_quad_err: {quad_err:.5f}, cam_trans_err: {trans_err:.5f}"
        if print_output:
            print(msg, flush=True)

    def log_mapping_iteration(self, **kwargs) -> None:
        """ Logs mapping iteration metrics including the number of new points, model size, and optimization times,
        and optionally reports to Weights & Biases (wandb).
        Args:
            frame_id: Identifier for the current frame.
            new_pts_num: The number of new points added in the current mapping iteration.
            model_size: The total size of the model after the current mapping iteration.
            iter_opt_time: Time taken per optimization iteration.
            opt_dict: A dictionary containing optimization metrics such as PSNR, color loss, and depth loss.
        """
        prefix = f"Agent_{self.agent_id}_mapping"
        if self.use_wandb:
            wandb_log_data = {}
            for key, value in kwargs.items():
                wandb_log_data[f"{prefix}/{key}"] = value
            wandb.log(wandb_log_data)

    @torch2np_decorator
    def log_tracking_results(self, est_c2w, gt_c2w, artifact_name: str):
        (self.output_path / "traj_eval").mkdir(exist_ok=True, parents=True)

        est_t, gt_t = est_c2w[:, :3, 3], gt_c2w[:, :3, 3]
        est_t_aligned = align_trajectories(est_t, gt_t)
        ate = compute_ate(est_t, gt_t)
        ate_aligned = compute_ate(est_t_aligned, gt_t)
        ate_rmse, ate_rmse_aligned = ate["rmse"] * 100, ate_aligned["rmse"] * 100

        save_dict_to_json({
            "ate_rmse": ate_rmse,
            "ate_rmse_aligned": ate_rmse_aligned
        }, "ate.json", directory=self.output_path / "traj_eval")

        ax = plot_trajectory(
            est_t, label=f"ate-rmse: {round(ate_rmse, 3)} cm", color="orange")
        ax = plot_trajectory(
            est_t_aligned, ax, label=f"ate-rsme (aligned): {round(ate_rmse_aligned, 3)} cm", color="lightskyblue")
        ax = plot_trajectory(
            gt_t, ax, label="GT", color="green")

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        plt.close()
        buf.seek(0)
        plot_image = Image.open(buf)
        plot_image.save(str(self.output_path / "traj_eval" / f"{artifact_name}"))
        if self.use_wandb:
            prefix = f"Agent_{self.agent_id}_trajectory/all_eval"
            wandb.log({prefix: [wandb.Image(plot_image)]})

    def log_loops(self, loops, artifact_name: str):
        (self.output_path).mkdir(exist_ok=True, parents=True)
        with open(self.output_path / artifact_name, "wb") as f:
            pickle.dump(loops, f)

    def log_registration_metrics(self, agents_datasets: dict, loops: list,
                                 fitness_threshold: float,
                                 inlier_rmse_threshold: float,
                                 rotation_threshold_deg: float = 5.0,
                                 translation_threshold_cm: float = 10.0) -> None:
        """Write paper-ready per-loop metrics and a compact JSON summary."""
        rows = []
        for loop in loops:
            source_pose = np.asarray(
                agents_datasets[loop.source_agent_id].poses[loop.source_frame_id])
            target_pose = np.asarray(
                agents_datasets[loop.target_agent_id].poses[loop.target_frame_id])
            reference = np.linalg.inv(target_pose) @ source_pose
            rotation_error = geodesic_rotation_error_deg(
                loop.transformation[:3, :3], reference[:3, :3])
            translation_error = float(np.linalg.norm(
                loop.transformation[:3, 3] - reference[:3, 3]) * 100.0)
            success = bool(rotation_error < rotation_threshold_deg and
                           translation_error < translation_threshold_cm)
            passed_filter = bool(loop.fitness > fitness_threshold and
                                 loop.inlier_rmse < inlier_rmse_threshold)

            coarse_transformation = getattr(loop, "coarse_transformation", None)
            coarse_rotation_error = None
            coarse_translation_error = None
            if coarse_transformation is not None:
                coarse_transformation = np.asarray(coarse_transformation)
                coarse_rotation_error = geodesic_rotation_error_deg(
                    coarse_transformation[:3, :3], reference[:3, :3])
                coarse_translation_error = float(np.linalg.norm(
                    coarse_transformation[:3, 3] - reference[:3, 3]) * 100.0)

            rows.append({
                "source_agent_id": loop.source_agent_id,
                "source_frame_id": loop.source_frame_id,
                "target_agent_id": loop.target_agent_id,
                "target_frame_id": loop.target_frame_id,
                "registration_method_requested": getattr(
                    loop, "registration_method_requested", None),
                "coarse_method_used": getattr(loop, "coarse_method_used", None),
                "fallback_used": bool(getattr(loop, "fallback_used", False)),
                "coarse_registration_failed": bool(getattr(
                    loop, "coarse_registration_failed", False)),
                "num_source_features": getattr(loop, "num_source_features", None),
                "num_target_features": getattr(loop, "num_target_features", None),
                "num_feature_matches": getattr(loop, "num_feature_matches", None),
                "num_mutual_matches": getattr(loop, "num_mutual_matches", None),
                "num_confidence_matches": getattr(
                    loop, "num_confidence_matches", None),
                "num_depth_valid_correspondences": getattr(
                    loop, "num_depth_valid_correspondences",
                    getattr(loop, "num_correspondences", None)),
                "num_geometry_consistent_correspondences": getattr(
                    loop, "num_geometry_consistent_correspondences", None),
                "num_correspondences": getattr(loop, "num_correspondences", None),
                "num_ransac_inliers": getattr(loop, "num_ransac_inliers", None),
                "ransac_inlier_ratio": getattr(
                    loop, "ransac_inlier_ratio", None),
                "correspondence_retention_ratio": getattr(
                    loop, "correspondence_retention_ratio", None),
                "mean_match_similarity": getattr(
                    loop, "mean_match_similarity", None),
                "mean_match_margin": getattr(loop, "mean_match_margin", None),
                "match_similarity_p10": getattr(loop, "match_similarity_p10", None),
                "match_similarity_p50": getattr(loop, "match_similarity_p50", None),
                "match_similarity_p90": getattr(loop, "match_similarity_p90", None),
                "match_margin_p10": getattr(loop, "match_margin_p10", None),
                "match_margin_p50": getattr(loop, "match_margin_p50", None),
                "match_margin_p90": getattr(loop, "match_margin_p90", None),
                "mean_compatibility_support": getattr(
                    loop, "mean_compatibility_support", None),
                "dino_min_similarity": getattr(loop, "dino_min_similarity", None),
                "dino_min_margin": getattr(loop, "dino_min_margin", None),
                "dino_keep_top_fraction": getattr(
                    loop, "dino_keep_top_fraction", None),
                "dino_depth_window_radius": getattr(
                    loop, "dino_depth_window_radius", None),
                "dino_depth_max_mad_m": getattr(
                    loop, "dino_depth_max_mad_m", None),
                "dino_compatibility_threshold_m": getattr(
                    loop, "dino_compatibility_threshold_m", None),
                "dino_min_compatibility_support": getattr(
                    loop, "dino_min_compatibility_support", None),
                "dino_preprocess_mode": getattr(
                    loop, "dino_preprocess_mode", None),
                "source_patch_input_height": getattr(
                    loop, "source_patch_input_height", None),
                "source_patch_input_width": getattr(
                    loop, "source_patch_input_width", None),
                "source_patch_grid_height": getattr(
                    loop, "source_patch_grid_height", None),
                "source_patch_grid_width": getattr(
                    loop, "source_patch_grid_width", None),
                "target_patch_input_height": getattr(
                    loop, "target_patch_input_height", None),
                "target_patch_input_width": getattr(
                    loop, "target_patch_input_width", None),
                "target_patch_grid_height": getattr(
                    loop, "target_patch_grid_height", None),
                "target_patch_grid_width": getattr(
                    loop, "target_patch_grid_width", None),
                "num_gaussians_total": getattr(
                    loop, "num_gaussians_total", None),
                "num_gaussians_opacity_valid": getattr(
                    loop, "num_gaussians_opacity_valid", None),
                "num_gaussians_in_view": getattr(
                    loop, "num_gaussians_in_view", None),
                "num_gaussians_depth_consistent": getattr(
                    loop, "num_gaussians_depth_consistent", None),
                "num_gaussian_landmarks": getattr(
                    loop, "num_gaussian_landmarks", None),
                "mean_gaussian_landmark_opacity": getattr(
                    loop, "mean_gaussian_landmark_opacity", None),
                "mean_gaussian_depth_residual_m": getattr(
                    loop, "mean_gaussian_depth_residual_m", None),
                "gaussian_min_opacity": getattr(
                    loop, "gaussian_min_opacity", None),
                "gaussian_depth_tolerance_m": getattr(
                    loop, "gaussian_depth_tolerance_m", None),
                "gaussian_max_landmarks": getattr(
                    loop, "gaussian_max_landmarks", None),
                "gaussian_refit_inliers": getattr(
                    loop, "gaussian_refit_inliers", None),
                "coarse_registration_time_s": getattr(
                    loop, "coarse_registration_time", None),
                "icp_registration_time_s": getattr(
                    loop, "icp_registration_time", None),
                "coarse_rotation_error_deg": coarse_rotation_error,
                "coarse_translation_error_cm": coarse_translation_error,
                "fitness": float(loop.fitness),
                "inlier_rmse": float(loop.inlier_rmse),
                "rotation_error_deg": float(rotation_error),
                "translation_error_cm": translation_error,
                "icp_rotation_improvement_deg": (
                    coarse_rotation_error - float(rotation_error)
                    if coarse_rotation_error is not None else None),
                "icp_translation_improvement_cm": (
                    coarse_translation_error - translation_error
                    if coarse_translation_error is not None else None),
                "success": success,
                "passed_filter": passed_filter,
                "false_negative": success and not passed_filter,
                "false_positive": passed_filter and not success,
            })

        csv_path = self.output_path / "registration_metrics.csv"
        if rows:
            with open(csv_path, "w", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        def mean_of(field):
            values = [row[field] for row in rows if row[field] is not None]
            return float(np.mean(values)) if values else None

        successful = sum(row["success"] for row in rows)
        summary = {
            "num_inter_agent_loops": len(rows),
            "num_successful": successful,
            "success_rate_percent": 100.0 * successful / len(rows) if rows else 0.0,
            "mean_rotation_error_deg": mean_of("rotation_error_deg"),
            "mean_translation_error_cm": mean_of("translation_error_cm"),
            "mean_fitness": mean_of("fitness"),
            "mean_inlier_rmse": mean_of("inlier_rmse"),
            "mean_feature_matches": mean_of("num_feature_matches"),
            "mean_mutual_matches": mean_of("num_mutual_matches"),
            "mean_confidence_matches": mean_of("num_confidence_matches"),
            "mean_depth_valid_correspondences": mean_of(
                "num_depth_valid_correspondences"),
            "mean_geometry_consistent_correspondences": mean_of(
                "num_geometry_consistent_correspondences"),
            "mean_final_correspondences": mean_of("num_correspondences"),
            "mean_ransac_inliers": mean_of("num_ransac_inliers"),
            "mean_ransac_inlier_ratio": mean_of("ransac_inlier_ratio"),
            "mean_correspondence_retention_ratio": mean_of(
                "correspondence_retention_ratio"),
            "mean_match_similarity": mean_of("mean_match_similarity"),
            "mean_match_margin": mean_of("mean_match_margin"),
            "mean_compatibility_support": mean_of(
                "mean_compatibility_support"),
            "mean_gaussians_total": mean_of("num_gaussians_total"),
            "mean_gaussians_opacity_valid": mean_of(
                "num_gaussians_opacity_valid"),
            "mean_gaussians_in_view": mean_of("num_gaussians_in_view"),
            "mean_gaussians_depth_consistent": mean_of(
                "num_gaussians_depth_consistent"),
            "mean_gaussian_landmarks": mean_of("num_gaussian_landmarks"),
            "mean_gaussian_landmark_opacity": mean_of(
                "mean_gaussian_landmark_opacity"),
            "mean_gaussian_depth_residual_m": mean_of(
                "mean_gaussian_depth_residual_m"),
            "mean_coarse_rotation_error_deg": mean_of(
                "coarse_rotation_error_deg"),
            "mean_coarse_translation_error_cm": mean_of(
                "coarse_translation_error_cm"),
            "mean_icp_rotation_improvement_deg": mean_of(
                "icp_rotation_improvement_deg"),
            "mean_icp_translation_improvement_cm": mean_of(
                "icp_translation_improvement_cm"),
            "mean_coarse_registration_time_s": mean_of("coarse_registration_time_s"),
            "mean_icp_registration_time_s": mean_of("icp_registration_time_s"),
            "num_passed_filter": sum(row["passed_filter"] for row in rows),
            "num_coarse_failures": sum(row["coarse_registration_failed"] for row in rows),
            "num_fpfh_fallbacks": sum(row["fallback_used"] for row in rows),
            "num_false_negatives": sum(row["false_negative"] for row in rows),
            "num_false_positives": sum(row["false_positive"] for row in rows),
            "rotation_success_threshold_deg": rotation_threshold_deg,
            "translation_success_threshold_cm": translation_threshold_cm,
            "fitness_threshold": fitness_threshold,
            "inlier_rmse_threshold": inlier_rmse_threshold,
        }
        save_dict_to_json(summary, "registration_summary.json", directory=self.output_path)

    def log_loops_quality(self, agents_submaps: dict, agents_datasets: dict, loops: list):
        output_file = open(str(self.output_path / "loops_quality.txt"), "w")
        good_loops = []
        errors_t, errors_q = [], []
        for i, loop in enumerate(loops):
            sa, sf, ta, tf = loop.source_agent_id, loop.source_frame_id, loop.target_agent_id, loop.target_frame_id
            source_sub, target_sub = find_submap(sf, agents_submaps[sa]), find_submap(tf, agents_submaps[ta])

            rel_pose = loop.transformation
            rel_pose_est = np.linalg.inv(target_sub["submap_c2ws"][0]) @ source_sub["submap_c2ws"][0]
            gt_rel_pose = np.linalg.inv(agents_datasets[ta][tf][-1]) @ agents_datasets[sa][sf][-1]

            t_reg_err, q_reg_err = pose_error(rel_pose, gt_rel_pose)
            t_est_err, q_est_err = pose_error(rel_pose_est, gt_rel_pose)

            if t_reg_err > t_est_err or q_reg_err > q_est_err:
                errors_t.append(t_reg_err)
                errors_q.append(q_reg_err)
                txt = f"{i} {loop.inlier_rmse:.3f} t_reg_err: {t_reg_err:.3f}, t_est_err: {t_est_err:.3f}" + \
                    f"q_reg_err: {q_reg_err: .3f}, q_est_err: {q_est_err:.3f} \n"
                output_file.write(txt)
                continue
            good_loops.append(loop)
        if len(errors_t) > 0:
            errors_t, errors_q = np.array(errors_t), np.array(errors_q)
            txt = f"Mean t_reg_err: {np.mean(errors_t):.3f}, Mean q_reg_err: {np.mean(errors_q):.3f}"
            output_file.write(txt)
        txt = f"Number of good loops: {len(good_loops)} out of {len(loops)}"
        output_file.write(txt)
        output_file.close()

    def vis_mapping_iteration(self, frame_id, iter, color, depth, gt_color, gt_depth, seeding_mask=None) -> None:
        """
        Visualization of depth, color images and save to file.

        Args:
            frame_id (int): current frame index.
            iter (int): the iteration number.
            save_rendered_image (bool): whether to save the rgb image in separate folder
            img_dir (str): the directory to save the visualization.
            seeding_mask: used in mapper when adding gaussians, if not none.
        """
        (self.output_path / "mapping_vis").mkdir(exist_ok=True, parents=True)
        gt_depth_np = gt_depth.cpu().numpy()
        gt_color_np = gt_color.cpu().numpy()

        depth_np = depth.detach().cpu().numpy()
        color = torch.round(color * 255.0) / 255.0
        color_np = color.detach().cpu().numpy()
        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        # make errors >=5cm noticeable
        depth_residual = np.clip(depth_residual, 0.0, 0.05)

        color_residual = np.abs(gt_color_np - color_np)
        color_residual[np.squeeze(gt_depth_np == 0.0)] = 0.0

        # Determine Aspect Ratio and Figure Size
        aspect_ratio = color.shape[1] / color.shape[0]
        fig_height = 8
        # Adjust the multiplier as needed for better spacing
        fig_width = fig_height * aspect_ratio * 1.2

        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        axs[0, 0].imshow(gt_depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 0].set_title('Input Depth', fontsize=16)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 1].set_title('Rendered Depth', fontsize=16)
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, cmap="plasma")
        axs[0, 2].set_title('Depth Residual', fontsize=16)
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np, cmap="plasma")
        axs[1, 0].set_title('Input RGB', fontsize=16)
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap="plasma")
        axs[1, 1].set_title('Rendered RGB', fontsize=16)
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        if seeding_mask is not None:
            axs[1, 2].imshow(seeding_mask, cmap="gray")
            axs[1, 2].set_title('Densification Mask', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])
        else:
            axs[1, 2].imshow(color_residual, cmap="plasma")
            axs[1, 2].set_title('RGB Residual', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])

        for ax in axs.flatten():
            ax.axis('off')
        fig.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust top margin
        fig_name = str(self.output_path / "mapping_vis" / f'{frame_id:04d}_{iter:04d}.jpg')
        fig_title = f"Mapper Color/Depth at frame {frame_id:04d} iters {iter:04d}"
        plt.suptitle(fig_title, y=0.98, fontsize=20)
        plt.savefig(fig_name, dpi=250, bbox_inches='tight')
        plt.clf()
        plt.close()
        if self.use_wandb:
            prefix = f"Agent_{self.agent_id}_mapping_vis/"
            log_title = prefix + f'{frame_id:04d}_{iter:04d}'
            wandb.log({log_title: [wandb.Image(fig_name)]})
        print(f"Saved rendering vis of color/depth at {frame_id:04d}_{iter:04d}.jpg")

    def log_gpu_usage(self, gpu_id: int) -> None:
        """ Logs the GPU memory usage to a JSON file.
        Args:
            gpu_id: The GPU ID to log the memory usage for.
        """
        memory_used, memory_total = get_gpu_usage_by_id(gpu_id)
        prefix = f"GPU_{gpu_id}"
        gpu_dict = {
            f"{prefix}/memory_used": memory_used,
            f"{prefix}/memory_total": memory_total
        }
        save_dict_to_json(gpu_dict, "system_stats.json", directory=self.output_path)
        if self.use_wandb:
            wandb.log(gpu_dict)
