import os

import cv2
import torch

from src.model.spresgan import ResNetGenerator
from src.utils.utils import (
    image_to_tensor,
    mask_to_binary_tensor,
    tensor_to_image_batched,
)
from src.utils.constants import TARGET_IMAGE_SIZE
from src.data_processing.utils import (
    get_synthetic_data_paths_with_semantic_mask,
)


def load_generator(model_path: str, device: torch.device) -> ResNetGenerator:
    generator: ResNetGenerator = ResNetGenerator(in_channels=4).to(device)
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
    target_image_size: int = TARGET_IMAGE_SIZE,
) -> None:
    image_tensor: torch.Tensor = (
        image_to_tensor(image_path, size=target_image_size).unsqueeze(0).to(device)
    )
    mask_tensor: torch.Tensor = (
        mask_to_binary_tensor(mask_path, size=target_image_size).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        generator_input: torch.Tensor = torch.cat([image_tensor, mask_tensor], dim=1)
        fake_b: torch.Tensor = generator(generator_input)

    output_image = tensor_to_image_batched(fake_b)
    cv2.imwrite(output_path, output_image)


def translate_all_images_spresgan(
    test_data_root_directory: str,
    output_directory: str,
    model_path: str,
    target_image_size: int = TARGET_IMAGE_SIZE,
) -> None:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_directory, exist_ok=True)

    generator: ResNetGenerator = load_generator(model_path, device)

    data_paths: list[tuple[str, str]] = get_synthetic_data_paths_with_semantic_mask(
        test_data_root_directory
    )

    index: int
    image_path: str
    mask_path: str
    for index, (image_path, mask_path) in enumerate(data_paths, start=1):
        output_path: str = os.path.join(output_directory, f"{index}.png")
        translate_one_image_spresgan(
            generator, image_path, mask_path, output_path, device, target_image_size
        )

    print(f"Translated {len(data_paths)} images to {output_directory}")


if __name__ == "__main__":
    translate_all_images_spresgan(
        "data/synthetic_split/test",
        "outputs/spresgan",
        "models/spresgan/best/SPresGAN_bs2_lr0.0002_g_a_to_b_final.model",
        target_image_size=TARGET_IMAGE_SIZE,
    )
