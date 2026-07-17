import os
import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance


from src.inference.inference_segmentor import (
    load_frozen_segmentor,
    predict_binary_mask_segmentor,
)
from src.utils.utils import (
    image_to_tensor,
    mask_to_binary_tensor,
    normalized_tensor_to_rgb_uint8,
)
from src.data_processing.utils import (
    get_semantic_mask_translated_image_paths_pair,
    get_real_image_paths,
)
from src.model.segmentor import UNetSegmentor
from src.utils.constants import TARGET_IMAGE_SIZE, CLASS_TO_SEMANTIC_INDEX_MAPPING


def binary_iou(predicted_mask: torch.Tensor, ground_truth_mask: torch.Tensor) -> float:
    predicted_mask_bool: torch.Tensor = predicted_mask.bool()
    ground_truth_mask_bool: torch.Tensor = ground_truth_mask.bool()
    intersection: int = (predicted_mask_bool & ground_truth_mask_bool).sum().item()
    union: int = (predicted_mask_bool | ground_truth_mask_bool).sum().item()

    if union == 0:
        return 1.0  # both empty, meaning perfect agreement

    return intersection / union


def compute_overall_iou_fid(
    mask_translated_pairs: list[
        tuple[str, str]
    ],  # [(ground_truth_mask_path, translated_image_path), ...]
    real_image_paths: list[str],  # separate held-out real domain-B photos
    segmentor_model_path: str,
    target_image_size: int = TARGET_IMAGE_SIZE,
    device: torch.device | None = None,
) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes: int = len(CLASS_TO_SEMANTIC_INDEX_MAPPING)
    segmentor: UNetSegmentor = load_frozen_segmentor(
        segmentor_model_path, device, num_classes
    )

    fid: FrechetInceptionDistance = FrechetInceptionDistance(
        feature=2048, normalize=False
    ).to(device)

    ious: list[float] = []

    mask_path: str
    image_path: str
    for mask_path, image_path in mask_translated_pairs:
        image_tensor: torch.Tensor = (
            image_to_tensor(image_path, size=target_image_size).unsqueeze(0).to(device)
        )

        predicted_binary_mask: torch.Tensor = predict_binary_mask_segmentor(
            segmentor, image_tensor
        )
        true_mask: torch.Tensor = mask_to_binary_tensor(
            mask_path, size=target_image_size
        ).to(device)

        ious.append(binary_iou(predicted_binary_mask.squeeze(0), true_mask))

        fid.update(
            normalized_tensor_to_rgb_uint8(image_tensor.squeeze(0)).unsqueeze(0),
            real=False,
        )

    for real_path in real_image_paths:
        real_tensor: torch.Tensor = image_to_tensor(
            real_path, size=target_image_size
        ).to(device)
        fid.update(normalized_tensor_to_rgb_uint8(real_tensor).unsqueeze(0), real=True)

    print(
        f"FID: {float(fid.compute().item())}, Mean IoU: {float(np.mean(ious)) if ious else 0.0}, Num. Samples: {len(ious)}"
    )


if __name__ == "__main__":
    # compute metrics for CycleGAN
    compute_overall_iou_fid(
        get_semantic_mask_translated_image_paths_pair(
            "data/synthetic_split/test", "outputs/cyclegan/images"
        ),
        get_real_image_paths("data/real_images"),
        "models/segmentor/best/UNetSegmentor_bs16_lr0.0001_best.model",
    )

    # compute metrics for SPresGAN
    compute_overall_iou_fid(
        get_semantic_mask_translated_image_paths_pair(
            "data/synthetic_split/test", "outputs/spresgan"
        ),
        get_real_image_paths("data/real_images"),
        "models/segmentor/best/UNetSegmentor_bs16_lr0.0001_best.model",
    )
