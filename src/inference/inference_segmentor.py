import torch

from torch.utils.data import DataLoader

from src.model.segmentor import UNetSegmentor
from src.utils.constants import (
    CLASS_TO_SEMANTIC_INDEX_MAPPING,
)


def evaluate_segmentor(
    model: UNetSegmentor,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = len(CLASS_TO_SEMANTIC_INDEX_MAPPING),
) -> tuple[float, float]:
    model.eval()
    curr_error: float = 0.0
    curr_loss: float = 0.0
    total_pixels: int = 0

    with torch.no_grad():
        images: torch.Tensor
        masks: torch.Tensor
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits: torch.Tensor = model(images)
            loss: torch.Tensor = UNetSegmentor.criterion(logits, masks, num_classes)

            predictions: torch.Tensor = torch.argmax(logits, dim=1)
            num_pixels: int = masks.numel()
            total_pixels += num_pixels
            curr_error += torch.sum((predictions != masks).float()).item()
            curr_loss += loss.item() * num_pixels

    return curr_loss / total_pixels, curr_error / total_pixels
