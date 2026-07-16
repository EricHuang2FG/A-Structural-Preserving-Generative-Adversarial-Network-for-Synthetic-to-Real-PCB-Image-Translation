import os
import time
import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable

from torch.utils.data import DataLoader

from src.data_processing.datasets import (
    PCBSPresGANSyntheticDataset,
    PCBSPresGANRealDataset,
    PCBSPresGANUnpairedDomainPair,
)
from src.model.spresgan import PatchGANDiscriminator, ResNetGenerator
from src.inference.inference_segmentor import (
    load_frozen_segmentor,
    predict_binary_mask_segmentor,
    predict_foreground_logit_segmentor,
)
from src.utils.constants import (
    TARGET_IMAGE_SIZE,
    SEED,
    SPRESGAN_MODEL_CHECKPOINTS_DIRECTORY,
    SPRESGAN_MODEL_BEST_MODEL_DIRECTORY,
    SPRESGAN_MODEL_TRAINING_CURVE_DIRECTORY,
    CLASS_TO_SEMANTIC_INDEX_MAPPING,
    MODEL_NAME_TEMPLATE,
    TRAINING_CURVE_FILE_NAME_TEMPLATE,
)


class ImagePool:
    def __init__(self, max_pool_size: int = 50) -> None:
        self.max_pool_size: int = max_pool_size
        self.data: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        result: list[torch.Tensor] = []

        image: torch.Tensor
        for image in images:
            image = image.unsqueeze(0)

            if len(self.data) < self.max_pool_size:
                self.data.append(image)
                result.append(image)
            elif random.uniform(0, 1) > 0.5:
                index: int = random.randint(0, self.max_pool_size - 1)
                result.append(self.data[index].clone())
                self.data[index] = image
            else:
                result.append(image)

        return torch.cat(result, dim=0)


def initialize_weights(module: nn.Module) -> None:
    class_name: str = module.__class__.__name__

    if "Conv" in class_name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "InstanceNorm2d" in class_name and module.affine:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def linear_lr_schedule(
    num_epochs: int, linear_decay_start_epoch: int = 100
) -> Callable:
    def schedule_multiplier(epoch: int) -> float:
        if epoch < linear_decay_start_epoch:
            return 1.0

        return 1.0 - (epoch - linear_decay_start_epoch) / (
            num_epochs - linear_decay_start_epoch
        )

    return schedule_multiplier


def plot_spresgan_training_curves(
    generator_a_loss: np.ndarray,
    discriminator_a_loss: np.ndarray,
    discriminator_b_loss: np.ndarray,
    structural_loss: np.ndarray | None,
    output_path_template: str,  # must have {{ type }} in the string
) -> None:
    num_epochs: int = len(generator_a_loss) + 1

    # generator, discriminator losses
    plt.figure()
    plt.title("Generator and Discriminator Loss vs. Epochs")
    plt.plot(range(1, num_epochs), generator_a_loss, label="Generator Loss")
    plt.plot(range(1, num_epochs), discriminator_a_loss, label="Discriminator A Loss")
    plt.plot(range(1, num_epochs), discriminator_b_loss, label="Discriminator B Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(output_path_template.replace("{{ type }}", "loss"))
    plt.show()

    # structural loss
    plt.figure()
    plt.title("Structural Consistency Loss vs. Epochs")
    plt.plot(range(1, num_epochs), structural_loss, label="Structure Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(output_path_template.replace("{{ type }}", "structure_loss"))
    plt.show()

    # save raw metrics
    metrics_path: str = output_path_template.replace("{{ type }}", "metrics")
    metrics_path = metrics_path.rsplit(".", 1)[0] + ".npz"
    metrics_mapping: dict[str, np.ndarray] = {
        "generator_a_loss": generator_a_loss,
        "discriminator_a_loss": discriminator_a_loss,
        "discriminator_b_loss": discriminator_b_loss,
        "structure_loss": structural_loss,
    }
    np.savez(metrics_path, **metrics_mapping)


def train_spresgan(
    synthetic_data_root_directory: str,
    real_data_root_directory: str,
    segmentor_model_path: str = "models/segmentor/best/best.model",
    target_image_size: int = TARGET_IMAGE_SIZE,
    num_classes: int = len(CLASS_TO_SEMANTIC_INDEX_MAPPING),
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    num_epochs: int = 200,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 5.0,
    lambda_structure: float = 10.0,
    resume_checkpoint_path: str | None = None,
) -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(SPRESGAN_MODEL_TRAINING_CURVE_DIRECTORY, exist_ok=True)
    os.makedirs(SPRESGAN_MODEL_BEST_MODEL_DIRECTORY, exist_ok=True)
    os.makedirs(SPRESGAN_MODEL_CHECKPOINTS_DIRECTORY, exist_ok=True)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    synthetic_dataset: PCBSPresGANSyntheticDataset = PCBSPresGANSyntheticDataset(
        synthetic_data_root_directory, target_image_size
    )
    real_dataset: PCBSPresGANRealDataset = PCBSPresGANRealDataset(
        real_data_root_directory, target_image_size
    )
    paired: PCBSPresGANUnpairedDomainPair = PCBSPresGANUnpairedDomainPair(
        synthetic_dataset, real_dataset
    )
    loader: DataLoader = DataLoader(
        paired, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    print(
        f"{len(synthetic_dataset)} synthetic, {len(real_dataset)} real -> {len(paired)} pairs/epoch"
    )

    # define generators, discriminators and load the frozen segmentor
    generator_in_channels: int = 4
    g_a_to_b: ResNetGenerator = ResNetGenerator(
        in_channels=generator_in_channels, out_channels=3
    ).to(device)
    g_b_to_a: ResNetGenerator = ResNetGenerator(
        in_channels=generator_in_channels, out_channels=3
    ).to(device)
    d_a: PatchGANDiscriminator = PatchGANDiscriminator(in_channels=3).to(device)
    d_b: PatchGANDiscriminator = PatchGANDiscriminator(in_channels=3).to(device)

    g_a_to_b.apply(initialize_weights)
    g_b_to_a.apply(initialize_weights)
    d_a.apply(initialize_weights)
    d_b.apply(initialize_weights)

    segmentor = load_frozen_segmentor(segmentor_model_path, device, num_classes)

    # define losses
    adversarial_loss: nn.MSELoss = nn.MSELoss()
    cycle_loss: nn.L1Loss = nn.L1Loss()
    identity_loss: nn.L1Loss = nn.L1Loss()
    structure_loss_function: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    # define optimizers and learning rate scheduling
    opt_g: torch.optim.Adam = torch.optim.Adam(
        list(g_a_to_b.parameters()) + list(g_b_to_a.parameters()),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )
    opt_d: torch.optim.Adam = torch.optim.Adam(
        list(d_a.parameters()) + list(d_b.parameters()),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )
    decay_start_epoch: int = num_epochs // 2
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        opt_g, lr_lambda=linear_lr_schedule(num_epochs, decay_start_epoch)
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        opt_d, lr_lambda=linear_lr_schedule(num_epochs, decay_start_epoch)
    )

    buffer_fake_a: ImagePool = ImagePool()
    buffer_fake_b: ImagePool = ImagePool()

    # define arrays that track losses
    g_losses: np.ndarray = np.zeros(num_epochs)
    d_a_losses: np.ndarray = np.zeros(num_epochs)
    d_b_losses: np.ndarray = np.zeros(num_epochs)
    structure_losses: np.ndarray = np.zeros(num_epochs)

    start_epoch: int = 0
    total_epochs_ran: int = 0

    # load checkpoint information
    if resume_checkpoint_path is not None:
        checkpoint: dict = torch.load(resume_checkpoint_path, map_location=device)
        g_a_to_b.load_state_dict(checkpoint["g_a_to_b"])
        g_b_to_a.load_state_dict(checkpoint["g_b_to_a"])
        d_a.load_state_dict(checkpoint["d_a"])
        d_b.load_state_dict(checkpoint["d_b"])
        opt_g.load_state_dict(checkpoint["opt_g"])
        opt_d.load_state_dict(checkpoint["opt_d"])
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])
        scheduler_d.load_state_dict(checkpoint["scheduler_d"])

        total_epochs_ran = checkpoint["epoch"] + 1
        start_epoch = total_epochs_ran

        g_losses[: len(checkpoint["g_losses"])] = checkpoint["g_losses"]
        d_a_losses[: len(checkpoint["d_a_losses"])] = checkpoint["d_a_losses"]
        d_b_losses[: len(checkpoint["d_b_losses"])] = checkpoint["d_b_losses"]
        if checkpoint.get("structure_losses") is not None:
            structure_losses[: len(checkpoint["structure_losses"])] = checkpoint[
                "structure_losses"
            ]

        print(f"Resumed from epoch {total_epochs_ran} ({resume_checkpoint_path})")

    model_name_no_epoch: str = (
        MODEL_NAME_TEMPLATE.replace(
            "{{ model_name }}",
            "SPresGAN",
        )
        .replace("{{ batch_size }}", str(batch_size))
        .replace("{{ learning_rate }}", str(learning_rate))
    )

    start_time: float = time.perf_counter()

    for epoch in range(start_epoch, num_epochs):
        epoch_g_loss: float = 0.0
        epoch_d_a_loss: float = 0.0
        epoch_d_b_loss: float = 0.0
        epoch_loss_structure: float = 0.0
        num_batches: int = 0

        for batch in loader:
            real_a: torch.Tensor = batch["real_a"].to(device)
            real_b: torch.Tensor = batch["real_b"].to(device)
            mask_a: torch.Tensor = batch["mask_a"].to(device)

            mask_b_predicted: torch.Tensor | None = None
            mask_b_predicted = predict_binary_mask_segmentor(segmentor, real_b)

            with torch.no_grad():
                probe_shape: torch.Size = d_b(real_b).shape
            valid: torch.Tensor = torch.ones(probe_shape, device=device)
            fake_label: torch.Tensor = torch.zeros(probe_shape, device=device)

            # train generators
            opt_g.zero_grad()

            fake_b: torch.Tensor = g_a_to_b(torch.cat([real_a, mask_a], dim=1))
            fake_a: torch.Tensor = g_b_to_a(
                torch.cat([real_b, mask_b_predicted], dim=1)
            )

            gan_a_to_b_loss: torch.Tensor = adversarial_loss(d_b(fake_b), valid)
            gan_b_to_a_loss: torch.Tensor = adversarial_loss(d_a(fake_a), valid)

            recovered_a: torch.Tensor = g_b_to_a(torch.cat([fake_b, mask_a], dim=1))
            recovered_b: torch.Tensor = g_a_to_b(
                torch.cat([fake_a, mask_b_predicted], dim=1)
            )
            cycle_a_loss: torch.Tensor = cycle_loss(recovered_a, real_a)
            cycle_b_loss: torch.Tensor = cycle_loss(recovered_b, real_b)

            identity_a: torch.Tensor = g_b_to_a(torch.cat([real_a, mask_a], dim=1))
            identity_b: torch.Tensor = g_a_to_b(
                torch.cat([real_b, mask_b_predicted], dim=1)
            )
            identity_a_loss: torch.Tensor = identity_loss(identity_a, real_a)
            identity_b_loss: torch.Tensor = identity_loss(identity_b, real_b)

            g_loss: torch.Tensor = (
                gan_a_to_b_loss
                + gan_b_to_a_loss
                + lambda_cycle * (cycle_a_loss + cycle_b_loss)
                + lambda_identity * (identity_a_loss + identity_b_loss)
            )

            # add the segmentor structure loss
            loss_structure_value: float = 0.0
            predicted_logit: torch.Tensor = predict_foreground_logit_segmentor(
                segmentor, fake_b
            )
            loss_structure: torch.Tensor = structure_loss_function(
                predicted_logit, mask_a
            )
            g_loss = g_loss + lambda_structure * loss_structure
            loss_structure_value = loss_structure.item()

            g_loss.backward()
            opt_g.step()

            # train the discriminators
            opt_d.zero_grad()

            d_a_real_loss: torch.Tensor = adversarial_loss(d_a(real_a), valid)
            fake_a_pooled: torch.Tensor = buffer_fake_a.query(fake_a.detach())
            d_a_fake_loss: torch.Tensor = adversarial_loss(
                d_a(fake_a_pooled), fake_label
            )
            d_a_loss: torch.Tensor = 0.5 * (d_a_real_loss + d_a_fake_loss)

            d_b_real_loss: torch.Tensor = adversarial_loss(d_b(real_b), valid)
            fake_b_pooled: torch.Tensor = buffer_fake_b.query(fake_b.detach())
            d_b_fake_loss: torch.Tensor = adversarial_loss(
                d_b(fake_b_pooled), fake_label
            )
            d_b_loss: torch.Tensor = 0.5 * (d_b_real_loss + d_b_fake_loss)

            (d_a_loss + d_b_loss).backward()
            opt_d.step()

            epoch_g_loss += g_loss.item()
            epoch_d_a_loss += d_a_loss.item()
            epoch_d_b_loss += d_b_loss.item()
            epoch_loss_structure += loss_structure_value
            num_batches += 1

        scheduler_g.step()
        scheduler_d.step()

        g_losses[epoch] = epoch_g_loss / num_batches
        d_a_losses[epoch] = epoch_d_a_loss / num_batches
        d_b_losses[epoch] = epoch_d_b_loss / num_batches
        structure_losses[epoch] = epoch_loss_structure / num_batches

        total_epochs_ran += 1

        print(
            (
                f"Epoch {epoch + 1}/{num_epochs}: g_loss={g_losses[epoch]:.4f} "
                f"d_a_loss={d_a_losses[epoch]:.4f} d_b_loss={d_b_losses[epoch]:.4f}"
                f" loss_structure={structure_losses[epoch]:.4f}"
            )
        )

        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "g_a_to_b": g_a_to_b.state_dict(),
                    "g_b_to_a": g_b_to_a.state_dict(),
                    "d_a": d_a.state_dict(),
                    "d_b": d_b.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "scheduler_g": scheduler_g.state_dict(),
                    "scheduler_d": scheduler_d.state_dict(),
                    "g_losses": g_losses[: epoch + 1],
                    "d_a_losses": d_a_losses[: epoch + 1],
                    "d_b_losses": d_b_losses[: epoch + 1],
                    "structure_losses": (structure_losses[: epoch + 1]),
                },
                os.path.join(
                    SPRESGAN_MODEL_CHECKPOINTS_DIRECTORY,
                    model_name_no_epoch.replace("{{ epoch }}", str(epoch + 1)),
                ),
            )

    end_time: float = time.perf_counter()
    print(f"Total time elapsed: {(end_time - start_time):.4f}s")

    torch.save(
        g_a_to_b.state_dict(),
        os.path.join(
            SPRESGAN_MODEL_BEST_MODEL_DIRECTORY,
            model_name_no_epoch.replace("_epoch{{ epoch }}", "_g_a_to_b_final"),
        ),
    )
    torch.save(
        g_b_to_a.state_dict(),
        os.path.join(
            SPRESGAN_MODEL_BEST_MODEL_DIRECTORY,
            model_name_no_epoch.replace("_epoch{{ epoch }}", "_g_b_to_a_final"),
        ),
    )

    plot_spresgan_training_curves(
        g_losses[:total_epochs_ran],
        d_a_losses[:total_epochs_ran],
        d_b_losses[:total_epochs_ran],
        structure_losses[:total_epochs_ran],
        os.path.join(
            SPRESGAN_MODEL_TRAINING_CURVE_DIRECTORY,
            TRAINING_CURVE_FILE_NAME_TEMPLATE.replace(
                "{{ model_name }}",
                "SPresGAN",
            )
            .replace("{{ batch_size }}", str(batch_size))
            .replace("{{ learning_rate }}", str(learning_rate)),
        ),
    )


if __name__ == "__main__":
    train_spresgan(
        "data/synthetic_split/train",
        "data/real_images",
        segmentor_model_path="models/segmentor/best/UNetSegmentor_bs16_lr0.0001_best.model",
        num_epochs=30,
    )
