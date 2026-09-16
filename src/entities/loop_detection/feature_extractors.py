""" This module contains the feature extractors for the loop detection module.
    We implement NetVLAD feature extractor to compare it with ours.
    For NetVLAD we use the same hyper parameters as in CP-SLAM.
"""
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.entities.loop_detection.netvlad import NetVLAD


class BaseFeatureExtractor(object):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.weights_path = config["weights_path"]
        self.device = config.get("device", "cpu")

    def extract_features(self, image: Image) -> torch.Tensor:
        """ Extracts features from an image.
        Args:
            image: The input image.
        Returns:
            features: The extracted features.
        """
        raise NotImplementedError


class DINOFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Pin the slow processor explicitly so a future Transformers upgrade
        # does not silently change the experiment preprocessing.
        self.preprocess = AutoImageProcessor.from_pretrained(
            self.weights_path,
            use_fast=self.config.get("dino_use_fast", False),
        )
        self.model = AutoModel.from_pretrained(self.weights_path).to(self.device)
        self.model.eval()
        self.embed_size = self.config.get("embed_size", 384)
        self.dino_patch_preserve_aspect_ratio = self.config.get(
            "dino_patch_preserve_aspect_ratio", True)
        self.dino_patch_short_edge = int(
            self.config.get("dino_patch_short_edge", 280))

    def _geometry_preserving_patch_image(self, image: Image):
        """Resize the full image to a patch-aligned resolution without cropping.

        The default DINOv2 processor resizes then center-crops to 224x224. That
        is appropriate for global retrieval descriptors, but patch locations can
        no longer be mapped to the original RGB-D image by a simple grid scale.
        For geometric registration we therefore preserve the complete field of
        view and round both resized dimensions to multiples of the ViT patch size.
        """
        patch_size = self.model.config.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        else:
            patch_h = patch_w = int(patch_size)

        orig_w, orig_h = image.size
        short_edge = max(self.dino_patch_short_edge, patch_h, patch_w)
        scale = short_edge / min(orig_h, orig_w)
        resized_h = max(patch_h, int(round(orig_h * scale / patch_h)) * patch_h)
        resized_w = max(patch_w, int(round(orig_w * scale / patch_w)) * patch_w)

        resampling = getattr(Image, "Resampling", Image).BICUBIC
        resized = image.resize((resized_w, resized_h), resampling)
        return resized, (patch_h, patch_w)

    def extract_features(self, image: Image) -> torch.Tensor:
        """ Extracts DINOv2 features from an image. The features are normalized
            to be suitable for L2 distance search in the faiss database.
        Args:
            image: The input image.
        Returns:
            features: extracted DINOv2 features.
        """
        with torch.no_grad():
            inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
            features = features / features.norm(p=2, dim=1, keepdim=True)
            return features

    def extract_patch_tokens(self, image, return_metadata=False):
        """Return patch tokens plus an exact mapping to the original image.

        Returns (tokens, (grid_h, grid_w), (patch_h_px, patch_w_px)).
            tokens: L2-normalized [num_patches, embed_dim], CLS token dropped.
            patch_h_px/patch_w_px: patch size in ORIGINAL image pixels
            so a token center can be lifted with the original depth image.

        When ``return_metadata`` is True, a fourth dictionary records the
        preprocessing mode and processed resolution for paper diagnostics.
        """
        with torch.no_grad():
            orig_w, orig_h = image.size
            if self.dino_patch_preserve_aspect_ratio:
                model_image, (patch_h, patch_w) = self._geometry_preserving_patch_image(image)
                inputs = self.preprocess(
                    images=model_image,
                    return_tensors="pt",
                    do_resize=False,
                    do_center_crop=False,
                ).to(self.device)
                preprocess_mode = "geometry_preserving_full_fov"
            else:
                inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)
                patch_size = self.model.config.patch_size
                if isinstance(patch_size, (tuple, list)):
                    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
                else:
                    patch_h = patch_w = int(patch_size)
                preprocess_mode = "legacy_center_crop"

            outputs = self.model(**inputs)
            tokens = outputs.last_hidden_state[:, 1:, :]
            tokens = tokens / tokens.norm(p=2, dim=-1, keepdim=True)

            _, _, h_in, w_in = inputs["pixel_values"].shape
            grid_h, grid_w = h_in // patch_h, w_in // patch_w
            expected_tokens = grid_h * grid_w
            if tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    "DINOv2 patch-grid mismatch: "
                    f"received {tokens.shape[1]} tokens for a "
                    f"{grid_h}x{grid_w} grid")

            result = (
                tokens.squeeze(0),
                (grid_h, grid_w),
                (orig_h / grid_h, orig_w / grid_w),
            )
            if not return_metadata:
                return result

            metadata = {
                "dino_preprocess_mode": preprocess_mode,
                "patch_input_height": int(h_in),
                "patch_input_width": int(w_in),
                "patch_grid_height": int(grid_h),
                "patch_grid_width": int(grid_w),
            }
            return (*result, metadata)


class NetVLADFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model = NetVLAD({
            "checkpoint_path": self.weights_path,
            "whiten": True
        }).to(self.device)
        self.model.eval()
        self.embed_size = self.config.get("embed_size", 4096)

    def extract_features(self, image: Image) -> torch.Tensor:
        """ Extracts NetVLAD features from an image following CP-SLAM.
        Args:
            image: The input image.
        Returns:
            features: extracted NetVLAD features.
        """
        with torch.no_grad():
            return self.model(image)


def get_feature_extractor(config: dict) -> BaseFeatureExtractor:
    """ Returns the feature extractor based on the configuration.
    Args:
        config: The configuration dictionary.
    Returns:
        feature_extractor: The feature extractor object.
    """
    if config["feature_extractor_name"] == "dino":
        return DINOFeatureExtractor(config)
    elif config["feature_extractor_name"] == "netvlad":
        return NetVLADFeatureExtractor(config)
    else:
        raise NotImplementedError


def get_patch_feature_extractor(
        loop_detection_config: dict,
        registration_config: dict = None,
        existing_extractor=None) -> DINOFeatureExtractor:
    """Return the DINO patch extractor used by semantic registration.

    Loop retrieval and pairwise registration usually share one DINO model.  If
    loop retrieval uses another descriptor (for example NetVLAD), or the user
    requests another device/weight path for registration, create a dedicated
    DINO extractor instead.  This keeps Gaussian-landmark registration usable
    in the full SLAM pipeline without coupling it to the retrieval backend.
    """
    registration_config = registration_config or {}
    extractor_config = dict(loop_detection_config)
    extractor_config["feature_extractor_name"] = "dino"
    extractor_config["weights_path"] = registration_config.get(
        "registration_weights_path", extractor_config.get("weights_path"))
    extractor_config["device"] = registration_config.get(
        "registration_device", extractor_config.get("device", "cpu"))

    if not extractor_config.get("weights_path"):
        raise ValueError(
            "Semantic registration requires submap.registration_weights_path "
            "or loop_detection.weights_path.")

    can_reuse = (
        existing_extractor is not None
        and hasattr(existing_extractor, "extract_patch_tokens")
        and getattr(existing_extractor, "weights_path", None)
        == extractor_config["weights_path"]
        and getattr(existing_extractor, "device", None)
        == extractor_config["device"]
    )
    if can_reuse:
        return existing_extractor
    return DINOFeatureExtractor(extractor_config)
