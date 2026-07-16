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


def load_frozen_segmentor(
    model_path: str,
    device: torch.device,
    num_classes: int = len(CLASS_TO_SEMANTIC_INDEX_MAPPING),
) -> UNetSegmentor:
    model: UNetSegmentor = UNetSegmentor(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    param: torch.nn.Parameter
    for param in model.parameters():
        param.requires_grad = False  # freeze model

    return model


def predict_foreground_logit_segmentor(
    model: UNetSegmentor, image: torch.Tensor
) -> torch.Tensor:
    background_semantic_index: int = CLASS_TO_SEMANTIC_INDEX_MAPPING["background"]

    logits: torch.Tensor = model(image)
    background_logit: torch.Tensor = logits[
        :, background_semantic_index : background_semantic_index + 1, :, :
    ]
    foreground_logits: torch.Tensor = torch.cat(
        [
            logits[:, :background_semantic_index],
            logits[:, background_semantic_index + 1 :],
        ],
        dim=1,
    )
    foreground_logsumexp: torch.Tensor = torch.logsumexp(
        foreground_logits, dim=1, keepdim=True
    )
    return foreground_logsumexp - background_logit


def predict_binary_mask_segmentor(
    model: UNetSegmentor,
    image: torch.Tensor,
    is_foreground_probability_threshold: float = 0.5,
) -> torch.Tensor:
    with torch.no_grad():
        logit: torch.Tensor = predict_foreground_logit_segmentor(model, image)
        probability: torch.Tensor = torch.sigmoid(logit)

        return (probability > is_foreground_probability_threshold).float()
