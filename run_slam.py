import argparse

from src.entities.magic_slam import MAGiCSLAM
from src.utils.io_utils import load_config
from src.utils.utils import setup_seed


def get_args():
    parser = argparse.ArgumentParser(description='Arguments to SLAM pipeline')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--input_path', default="")
    parser.add_argument('--output_path', default="")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--group_name', type=str)
    parser.add_argument(
        '--registration_method',
        choices=[
            'fpfh', 'dinov2', 'gaussian_landmark',
            'sift', 'orb', 'akaze'],
        help='Override submap.registration_method for a comparable experiment run')
    parser.add_argument(
        '--registration_device', choices=['cpu', 'cuda'],
        help='Device for DINO patch extraction during inter-agent registration')
    parser.add_argument(
        '--registration_weights_path',
        help='DINO weights used by dinov2/gaussian_landmark registration')
    parser.add_argument(
        '--disable_fpfh_fallback', action='store_true',
        help='Do not fall back to FPFH when semantic coarse registration fails')
    return parser.parse_args()


def update_config_with_args(config, args):
    if args.input_path:
        config["data"]["input_path"] = args.input_path
    if args.output_path:
        config["data"]["output_path"] = args.output_path
    # Seed 0 is a valid and commonly used benchmark seed.
    if args.seed is not None:
        config["seed"] = args.seed
    if args.multi_gpu:
        config["multi_gpu"] = True
    if args.use_wandb:
        config["use_wandb"] = True
    if args.wandb_entity:
        config["wandb_entity"] = args.wandb_entity
    if args.wandb_offline:
        config["wandb_offline"] = True
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    if args.group_name:
        config["group_name"] = args.group_name
    if args.registration_method:
        config["submap"]["registration_method"] = args.registration_method
    if args.registration_device:
        config["submap"]["registration_device"] = args.registration_device
    if args.registration_weights_path:
        config["submap"]["registration_weights_path"] = (
            args.registration_weights_path)
    if args.disable_fpfh_fallback:
        config["submap"]["fallback_to_fpfh"] = False
    return config


def validate_registration_config(config):
    """Fail early when the selected method cannot run in the SLAM path."""
    submap_config = config.setdefault("submap", {})
    method = submap_config.get("registration_method", "fpfh").lower()
    supported = {
        "fpfh", "dinov2", "gaussian_landmark", "sift", "orb", "akaze"
    }
    if method not in supported:
        raise ValueError(
            f"Unsupported submap.registration_method={method!r}; "
            f"expected one of {sorted(supported)}")

    anchor_data = submap_config.get("anchor_data")
    if method != "fpfh" and anchor_data not in {"depth", "render_depth"}:
        raise ValueError(
            f"registration_method={method!r} requires anchor_data='depth' "
            "or 'render_depth'.")

    submap_config.setdefault("initial_transformation_unknown", True)
    submap_config.setdefault("PGO_GTSAM", True)
    submap_config.setdefault("fallback_to_fpfh", True)
    submap_config.setdefault("registration_options", {})
    return config


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_path)
    config = update_config_with_args(config, args)
    config = validate_registration_config(config)

    setup_seed(config["seed"])
    coga_slam = MAGiCSLAM(config)
    coga_slam.run()
