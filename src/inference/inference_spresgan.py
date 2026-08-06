import os

import cv2
import torch
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from src.model.spresgan import ResNetGenerator, PatchGANDiscriminator
from src.model.segmentor import UNetSegmentor
from src.utils.utils import (
    image_to_tensor,
    mask_to_binary_tensor,
    tensor_to_image_batched,
    normalized_tensor_to_rgb_uint8,
)
from src.utils.constants import TARGET_IMAGE_SIZE
from src.data_processing.utils import get_synthetic_data_paths_with_semantic_mask
from src.inference.metrics import binary_iou
from src.inference.inference_segmentor import (
    predict_binary_mask_segmentor,
    predict_foreground_logit_segmentor,
)


def evaluate_spresgan(
    g_a_to_b: ResNetGenerator,
    g_b_to_a: ResNetGenerator,
    d_a: PatchGANDiscriminator,
    d_b: PatchGANDiscriminator,
    segmentor: UNetSegmentor,
    synthetic_validation_loader: DataLoader,
    real_validation_loader: DataLoader,
    adversarial_loss: nn.MSELoss,
    structure_loss_function: nn.BCEWithLogitsLoss,
    lambda_structure: float,
    device: torch.device,
    image_mask_as_generator_input: bool = True,
    compute_fid_iou: bool = False,
    fid_metric: FrechetInceptionDistance | None = None,
) -> dict[str, float]:
    if compute_fid_iou and fid_metric is None:
        raise ValueError(
            "fid_metric is None even though compute_fid_iou is set to True"
        )

    g_a_to_b.eval()
    g_b_to_a.eval()
    d_a.eval()
    d_b.eval()

    if compute_fid_iou:
        fid_metric.reset()

        real_b_validation: torch.Tensor
        for real_b_validation in real_validation_loader:
            real_b_validation = real_b_validation.to(device)

            sample_index: int
            for sample_index in range(real_b_validation.shape[0]):
                fid_metric.update(
                    normalized_tensor_to_rgb_uint8(
                        real_b_validation[sample_index]
                    ).unsqueeze(0),
                    real=True,
                )

    ious: list[float] = []
    validation_generator_a_losses: list[float] = []
    validation_generator_b_losses: list[float] = []
    validation_d_a_losses: list[float] = []
    validation_d_b_losses: list[float] = []
    validation_structure_losses: list[float] = []

    real_b_iterator = iter(real_validation_loader)

    real_a: torch.Tensor
    mask_a: torch.Tensor
    for real_a, mask_a in synthetic_validation_loader:
        real_a = real_a.to(device)
        mask_a = mask_a.to(device)

        try:
            real_b: torch.Tensor = next(real_b_iterator)
        except StopIteration:
            real_b_iterator = iter(real_validation_loader)
            real_b = next(real_b_iterator)
        real_b = real_b.to(device)

        with torch.no_grad():
            if image_mask_as_generator_input:
                mask_b_predicted: torch.Tensor = predict_binary_mask_segmentor(
                    segmentor, real_b
                )
                fake_b: torch.Tensor = g_a_to_b(torch.cat([real_a, mask_a], dim=1))
                fake_a: torch.Tensor = g_b_to_a(
                    torch.cat([real_b, mask_b_predicted], dim=1)
                )
            else:
                fake_b: torch.Tensor = g_a_to_b(real_a)
                fake_a: torch.Tensor = g_b_to_a(real_b)

            probe_shape: torch.Size = d_b(real_b).shape
            valid: torch.Tensor = torch.ones(probe_shape, device=device)
            fake_label: torch.Tensor = torch.zeros(probe_shape, device=device)

            gan_a_to_b_loss: torch.Tensor = adversarial_loss(d_b(fake_b), valid)
            gan_b_to_a_loss: torch.Tensor = adversarial_loss(d_a(fake_a), valid)

            predicted_logit: torch.Tensor = predict_foreground_logit_segmentor(
                segmentor, fake_b
            )
            structure_loss: torch.Tensor = structure_loss_function(
                predicted_logit, mask_a
            )

            generator_a_loss: torch.Tensor = (
                gan_a_to_b_loss + lambda_structure * structure_loss
            )
            generator_b_loss: torch.Tensor = gan_b_to_a_loss

            d_a_real_loss: torch.Tensor = adversarial_loss(d_a(real_a), valid)
            d_a_fake_loss: torch.Tensor = adversarial_loss(d_a(fake_a), fake_label)
            d_a_loss: torch.Tensor = 0.5 * (d_a_real_loss + d_a_fake_loss)

            d_b_real_loss: torch.Tensor = adversarial_loss(d_b(real_b), valid)
            d_b_fake_loss: torch.Tensor = adversarial_loss(d_b(fake_b), fake_label)
            d_b_loss: torch.Tensor = 0.5 * (d_b_real_loss + d_b_fake_loss)

        if compute_fid_iou:
            predicted_binary_mask: torch.Tensor = predict_binary_mask_segmentor(
                segmentor, fake_b
            )
            for sample_index in range(fake_b.shape[0]):
                ious.append(
                    binary_iou(
                        predicted_binary_mask[sample_index],
                        mask_a[sample_index],
                    )
                )
                fid_metric.update(
                    normalized_tensor_to_rgb_uint8(fake_b[sample_index]).unsqueeze(0),
                    real=False,
                )

        validation_generator_a_losses.append(generator_a_loss.item())
        validation_generator_b_losses.append(generator_b_loss.item())
        validation_d_a_losses.append(d_a_loss.item())
        validation_d_b_losses.append(d_b_loss.item())
        validation_structure_losses.append(structure_loss.item())

    g_a_to_b.train()
    g_b_to_a.train()
    d_a.train()
    d_b.train()

    return {
        "validation_iou": float(np.mean(ious)) if ious else 0.0,
        "validation_fid": (
            float(fid_metric.compute().item()) if compute_fid_iou else 0.0
        ),
        "validation_generator_a_loss": (
            float(np.mean(validation_generator_a_losses))
            if validation_generator_a_losses
            else 0.0
        ),
        "validation_generator_b_loss": (
            float(np.mean(validation_generator_b_losses))
            if validation_generator_b_losses
            else 0.0
        ),
        "validation_d_a_loss": (
            float(np.mean(validation_d_a_losses)) if validation_d_a_losses else 0.0
        ),
        "validation_d_b_loss": (
            float(np.mean(validation_d_b_losses)) if validation_d_b_losses else 0.0
        ),
        "validation_structure_loss": (
            float(np.mean(validation_structure_losses))
            if validation_structure_losses
            else 0.0
        ),
    }


def load_generator(
    model_path: str, device: torch.device, image_mask_as_generator_input: bool = True
) -> ResNetGenerator:
    generator: ResNetGenerator = ResNetGenerator(
        in_channels=4 if image_mask_as_generator_input else 3
    ).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))

    generator.eval()

    param: torch.nn.Parameter
    for param in generator.parameters():
        param.requires_grad = False

    return generator


def translate_one_image_spresgan(
    generator: ResNetGenerator,
    image_path: str,
    mask_path: str,
    output_path: str,
    device: torch.device,
    image_mask_as_generator_input: bool = True,
    target_image_size: int = TARGET_IMAGE_SIZE,
) -> None:
    image_tensor: torch.Tensor = (
        image_to_tensor(image_path, size=target_image_size).unsqueeze(0).to(device)
    )
    mask_tensor: torch.Tensor = (
        mask_to_binary_tensor(mask_path, size=target_image_size).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        if image_mask_as_generator_input:
            generator_input: torch.Tensor = torch.cat(
                [image_tensor, mask_tensor], dim=1
            )
        else:
            generator_input: torch.Tensor = image_tensor
        fake_b: torch.Tensor = generator(generator_input)

    output_image = tensor_to_image_batched(fake_b)
    cv2.imwrite(output_path, output_image)


def translate_all_images_spresgan(
    test_data_root_directory: str,
    output_directory: str,
    model_path: str,
    image_mask_as_generator_input: bool = True,
    target_image_size: int = TARGET_IMAGE_SIZE,
) -> None:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_directory, exist_ok=True)

    generator: ResNetGenerator = load_generator(
        model_path, device, image_mask_as_generator_input=image_mask_as_generator_input
    )

    data_paths: list[tuple[str, str]] = get_synthetic_data_paths_with_semantic_mask(
        test_data_root_directory
    )

    index: int
    image_path: str
    mask_path: str
    for index, (image_path, mask_path) in enumerate(data_paths, start=1):
        output_path: str = os.path.join(output_directory, f"{index}.png")
        translate_one_image_spresgan(
            generator,
            image_path,
            mask_path,
            output_path,
            device,
            image_mask_as_generator_input=image_mask_as_generator_input,
            target_image_size=target_image_size,
        )
        print(f"Translated image {index}")

    print(f"Translated {len(data_paths)} images to {output_directory}")


if __name__ == "__main__":
    translate_all_images_spresgan(
        "data/synthetic_split/test",
        "outputs/spresgan",
        "models/spresgan/best/SPresGAN_bs2_lr0.0002_g_a_to_b_final_l25.model",
        image_mask_as_generator_input=False,
        target_image_size=TARGET_IMAGE_SIZE,
    )
