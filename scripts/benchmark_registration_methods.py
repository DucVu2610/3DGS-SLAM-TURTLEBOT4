#!/usr/bin/env python3
"""Benchmark coarse inter-agent registration methods on cached submaps.

This reuses one run's loop candidates and Gaussian submap checkpoints so every
method sees the same registration queries. It therefore isolates registration
from mapping, keyframe selection, and loop retrieval.
"""
import argparse
import csv
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.entities.datasets import get_dataset
from src.entities.logger import Logger
from src.entities.loop_detection.feature_extractors import get_feature_extractor
from src.utils.magic_slam_utils import Registration, register_submaps_depth
from src.utils.utils import setup_seed


SUPPORTED_METHODS = ("fpfh", "dinov2", "sift", "orb", "akaze")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Completed run containing config.yaml, loops.pkl, and agent_*/submaps")
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS,
                        default=list(SUPPORTED_METHODS))
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <run-dir>/registration_benchmark")
    parser.add_argument("--dinov2-weights",
                        help="Override loop_detection.weights_path for DINOv2")
    parser.add_argument("--allow-fpfh-fallback", action="store_true",
                        help="Enable production fallback; leave off for pure extractor comparison")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="RANSAC seeds; use several values for paper statistics")
    return parser.parse_args()


def load_run(run_dir):
    with open(run_dir / "config.yaml", "r") as input_file:
        config = yaml.safe_load(input_file)
    with open(run_dir / "loops.pkl", "rb") as input_file:
        all_loops = pickle.load(input_file)

    inter_loops = [loop for loop in all_loops
                   if loop.source_agent_id != loop.target_agent_id]
    if not inter_loops:
        raise RuntimeError(f"No inter-agent loops found in {run_dir / 'loops.pkl'}")

    agents_submaps = {}
    for agent_id in config["data"]["agent_ids"]:
        checkpoint_paths = sorted((run_dir / f"agent_{agent_id}" / "submaps").glob("*.ckpt"))
        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No submap checkpoints found for agent {agent_id} in {run_dir}")
        agents_submaps[agent_id] = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in checkpoint_paths
        ]
    return config, agents_submaps, inter_loops


def load_datasets(config):
    input_root = Path(config["data"]["input_path"])
    agent_paths = sorted(path for path in input_root.glob("*") if not path.name.startswith("."))
    agent_ids = config["data"]["agent_ids"]
    if len(agent_paths) < len(agent_ids):
        raise FileNotFoundError(
            f"Expected at least {len(agent_ids)} agent directories under {input_root}")

    datasets = {}
    dataset_class = get_dataset(config["dataset_name"])
    for index, agent_id in enumerate(agent_ids):
        dataset_config = {
            **config["data"],
            **config["cam"],
            **config["submap"],
            "input_path": str(agent_paths[index]),
        }
        datasets[agent_id] = dataset_class(dataset_config)
    return datasets


def fresh_registration(loop):
    return Registration(
        loop.source_agent_id,
        loop.source_frame_id,
        loop.target_agent_id,
        loop.target_frame_id)


def run_method(method, config, agents_submaps, candidate_loops, datasets,
               output_dir, feature_extractor, allow_fpfh_fallback, seed):
    setup_seed(seed)
    method_dir = output_dir / method / f"seed_{seed}"
    method_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(method_dir)

    start = time.perf_counter()
    registrations = []
    for candidate in candidate_loops:
        registration = fresh_registration(candidate)
        registration = register_submaps_depth(
            agents_submaps,
            registration,
            initial_transformation_unknown=True,
            registration_method=method,
            feature_extractor=feature_extractor if method == "dinov2" else None,
            fallback_to_fpfh=allow_fpfh_fallback,
            registration_options=config["submap"].get("registration_options", {}))
        registrations.append(registration)
    wall_time = time.perf_counter() - start

    fitness_threshold = config["loop_detection"]["fitness_threshold"]
    inlier_rmse_threshold = config["loop_detection"]["inlier_rmse_threshold"]
    filtered = [registration for registration in registrations
                if registration.fitness > fitness_threshold and
                registration.inlier_rmse < inlier_rmse_threshold]
    logger.log_loops(registrations, "loops.pkl")
    logger.log_loops(filtered, "filtered_loops.pkl")
    logger.log_registration_metrics(
        datasets, registrations, fitness_threshold, inlier_rmse_threshold)

    summary_path = method_dir / "registration_summary.json"
    with open(summary_path, "r") as input_file:
        summary = json.load(input_file)
    summary.update({
        "method": method,
        "benchmark_wall_time_s": wall_time,
        "allow_fpfh_fallback": allow_fpfh_fallback,
        "seed": seed,
    })
    with open(summary_path, "w") as output_file:
        json.dump(summary, output_file, indent=2)
    return summary


def write_comparison(output_dir, summaries):
    with open(output_dir / "comparison.json", "w") as output_file:
        json.dump(summaries, output_file, indent=2)
    if summaries:
        with open(output_dir / "comparison.csv", "w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    aggregate_fields = (
        "success_rate_percent",
        "mean_rotation_error_deg",
        "mean_translation_error_cm",
        "mean_fitness",
        "mean_feature_matches",
        "mean_depth_valid_correspondences",
        "mean_ransac_inliers",
        "mean_coarse_registration_time_s",
        "mean_icp_registration_time_s",
        "num_passed_filter",
        "num_coarse_failures",
        "num_fpfh_fallbacks",
        "num_false_negatives",
        "num_false_positives",
    )
    aggregate_rows = []
    for method in dict.fromkeys(summary["method"] for summary in summaries):
        method_runs = [summary for summary in summaries if summary["method"] == method]
        row = {"method": method, "num_seeds": len(method_runs)}
        for field in aggregate_fields:
            values = [run[field] for run in method_runs if run.get(field) is not None]
            row[f"{field}_mean"] = statistics.mean(values) if values else None
            row[f"{field}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(row)

    with open(output_dir / "comparison_aggregate.json", "w") as output_file:
        json.dump(aggregate_rows, output_file, indent=2)
    if aggregate_rows:
        with open(output_dir / "comparison_aggregate.csv", "w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "registration_benchmark").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config, agents_submaps, candidate_loops = load_run(run_dir)
    datasets = load_datasets(config)

    feature_extractor = None
    if "dinov2" in args.methods:
        extractor_config = dict(config["loop_detection"])
        extractor_config["feature_extractor_name"] = "dino"
        if args.dinov2_weights:
            extractor_config["weights_path"] = args.dinov2_weights
        feature_extractor = get_feature_extractor(extractor_config)

    summaries = []
    for method in args.methods:
        for seed in args.seeds:
            print(f"\n=== Benchmarking {method}, seed={seed}, "
                  f"{len(candidate_loops)} inter-agent loop(s) ===")
            summaries.append(run_method(
                method, config, agents_submaps, candidate_loops, datasets,
                output_dir, feature_extractor, args.allow_fpfh_fallback, seed))
    write_comparison(output_dir, summaries)
    print(f"\nComparison written to {output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
