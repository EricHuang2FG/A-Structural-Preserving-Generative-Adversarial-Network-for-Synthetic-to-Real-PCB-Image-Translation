import os
import sys

sys.path.insert(0, "pytorch-CycleGAN-and-pix2pix/")

import torch

from typing import Any

from PIL import Image

# CycleGAN packages
from data import create_dataset
from models import create_model
from util.util import tensor2im
from options.test_options import TestOptions


def translate_one_image_cyclegan(model: Any, data: dict, output_directory: str) -> None:
    model.set_input(data)
    model.test()

    visuals: Any = model.get_current_visuals()
    generated_tensor: torch.Tensor = visuals["fake"]
    generated_image = tensor2im(generated_tensor)

    image_path: str = model.get_image_paths()[0]
    image_filename: str = os.path.basename(image_path)
    output_path: str = os.path.join(output_directory, image_filename)

    Image.fromarray(generated_image).save(output_path)


def translate_multiple_images_cyclegan() -> None:
    opt: Any = TestOptions().parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    dataset: Any = create_dataset(opt)
    model: Any = create_model(opt)
    model.setup(opt)

    if opt.eval:
        model.eval()

    output_directory: str = opt.results_dir
    os.makedirs(output_directory, exist_ok=True)

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        translate_one_image_cyclegan(model, data, output_directory)

    print(f"Saved {opt.num_test} images to {output_directory}")


if __name__ == "__main__":
    translate_multiple_images_cyclegan()
