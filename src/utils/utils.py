import cv2
import torch
import numpy as np

from typing import Any

from src.utils.constants import TARGET_IMAGE_SIZE


def image_to_tensor(image_path: str, size: int = TARGET_IMAGE_SIZE) -> torch.Tensor:
    image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image: Any = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image: Any = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    tensor: torch.Tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    tensor = tensor / 127.5 - 1.0  # normalize to [-1, 1]

    return tensor


def mask_to_binary_tensor(
    mask_path: str, size: int = TARGET_IMAGE_SIZE
) -> torch.Tensor:
    mask: np.ndarray = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask: Any = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    binary_mask: np.ndarray = (mask > 0).astype(np.float32)  # binary mask
    tensor: torch.Tensor = torch.from_numpy(binary_mask).unsqueeze(0)

    return tensor


def tensor_to_image_batched(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.squeeze(0).detach().cpu()
    image: np.ndarray = (
        ((tensor.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    )

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def normalized_tensor_to_rgb_uint8(image_tensor: torch.Tensor) -> torch.Tensor:
    # convert tensor with values normalized to [-1, 1] to an uint8 RGB tensor [0, 255]
    return ((image_tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
