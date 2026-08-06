import os
import time
import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable

from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

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
from src.inference.inference_spresgan import evaluate_spresgan
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


def linear_lambda_stucture_schedule(
    lambda_start: float = 0.0,
    lambda_end: float = 5.0,
    warmup_start_epoch: int = 25,
    warmup_end_epoch: int = 75,
) -> Callable:
    def schedule(epoch: int) -> float:
        if epoch < warmup_start_epoch:
            return lambda_start

        if epoch >= warmup_end_epoch:
            return lambda_end

        slope: float = (epoch - warmup_start_epoch) / (
            warmup_end_epoch - warmup_start_epoch
        )
        return lambda_end * slope

    return schedule


def plot_spresgan_training_curves(
    generator_a_loss: np.ndarray,
    generator_b_loss: np.ndarray,
    discriminator_a_loss: np.ndarray,
    discriminator_b_loss: np.ndarray,
    structural_loss: np.ndarray | None,
    validation_epochs: list[int],
    validation_iou: list[float],
    validation_fid: list[float],
    validation_generator_a_loss: list[float],
    validation_generator_b_loss: list[float],
    validation_d_a_loss: list[float],
    validation_d_b_loss: list[float],
    validation_structure_loss: list[float],
    output_path_template: str,  # must have {{ type }} in the string
    compute_fid_iou: bool = False,
    plot: bool = False,
) -> None:
    num_epochs: int = len(generator_a_loss) + 1
    title_fontsize: int = 14
    title_fontweight = "bold"

    # generator loss: A and B, training vs validation
    plt.figure()
    plt.title(
        "Training and Validation Generator Loss vs. Epochs",
        fontsize=title_fontsize,
        fontweight=title_fontweight,
    )
    plt.plot(range(1, num_epochs), generator_a_loss, label="Generator A (Train)")
    plt.plot(range(1, num_epochs), generator_b_loss, label="Generator B (Train)")
    plt.plot(
        validation_epochs,
        validation_generator_a_loss,
        label="Generator A (Validation)",
        marker="o",
    )
    plt.plot(
        validation_epochs,
        validation_generator_b_loss,
        label="Generator B (Validation)",
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(output_path_template.replace("{{ type }}", "generator_loss"))
    if plot:
        plt.show()

    # discriminator loss: A and B, training vs. validation
    plt.figure()
    plt.title(
        "Training and Validation Discriminator Loss vs. Epochs",
        fontsize=title_fontsize,
        fontweight=title_fontweight,
    )
    plt.plot(
        range(1, num_epochs), discriminator_a_loss, label="Discriminator A (Train)"
    )
    plt.plot(
        range(1, num_epochs), discriminator_b_loss, label="Discriminator B (Train)"
    )
    plt.plot(
        validation_epochs,
        validation_d_a_loss,
        label="Discriminator A (Validation)",
        marker="o",
    )
    plt.plot(
        validation_epochs,
        validation_d_b_loss,
        label="Discriminator B (Validation)",
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(output_path_template.replace("{{ type }}", "discriminator_loss"))
    if plot:
        plt.show()

    # structural loss: training vs validation
    plt.figure()
    plt.title(
        "Training and Validation Structural Loss vs. Epochs",
        fontsize=title_fontsize,
        fontweight=title_fontweight,
    )
    plt.plot(range(1, num_epochs), structural_loss, label="Structure Loss (Train)")
    plt.plot(
        validation_epochs,
        validation_structure_loss,
        label="Structure Loss (Validation)",
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(output_path_template.replace("{{ type }}", "structure_loss"))
    if plot:
        plt.show()

    if compute_fid_iou:
        plt.figure()
        plt.title(
            "Validation Segmentor IoU vs. Epochs",
            fontsize=title_fontsize,
            fontweight=title_fontweight,
        )
        plt.plot(validation_epochs, validation_iou, label="Validation IoU", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend(loc="best")
        plt.savefig(output_path_template.replace("{{ type }}", "validation_iou"))
        if plot:
            plt.show()

        plt.figure()
        plt.title(
            "Validation FID vs. Epochs",
            fontsize=title_fontsize,
            fontweight=title_fontweight,
        )
        plt.plot(validation_epochs, validation_fid, label="Validation FID", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("FID")
        plt.legend(loc="best")
        plt.savefig(output_path_template.replace("{{ type }}", "validation_fid"))
        if plot:
            plt.show()

    metrics_path: str = output_path_template.replace("{{ type }}", "metrics")
    metrics_path = metrics_path.rsplit(".", 1)[0] + ".npz"
    metrics_mapping: dict[str, np.ndarray] = {
        "generator_a_loss": generator_a_loss,
        "generator_b_loss": generator_b_loss,
        "discriminator_a_loss": discriminator_a_loss,
        "discriminator_b_loss": discriminator_b_loss,
        "structure_loss": structural_loss,
        "validation_epochs": np.array(validation_epochs),
        "validation_iou": np.array(validation_iou),
        "validation_fid": np.array(validation_fid),
        "validation_generator_a_loss": np.array(validation_generator_a_loss),
        "validation_generator_b_loss": np.array(validation_generator_b_loss),
        "validation_d_a_loss": np.array(validation_d_a_loss),
        "validation_d_b_loss": np.array(validation_d_b_loss),
        "validation_structure_loss": np.array(validation_structure_loss),
    }
    np.savez(metrics_path, **metrics_mapping)


def plot_spresgan_training_curves_from_npz(
    npz_path: str,
    output_path_template: str,  # must have {{ type }} in the string
    compute_fid_iou: bool = False,
    plot: bool = False,
) -> None:
    data = np.load(npz_path)

    plot_spresgan_training_curves(
        generator_a_loss=data["generator_a_loss"],
        generator_b_loss=data["generator_b_loss"],
        discriminator_a_loss=data["discriminator_a_loss"],
        discriminator_b_loss=data["discriminator_b_loss"],
        structural_loss=data["structure_loss"],
        validation_epochs=data["validation_epochs"].tolist(),
        validation_iou=data["validation_iou"].tolist(),
        validation_fid=data["validation_fid"].tolist(),
        validation_generator_a_loss=data["validation_generator_a_loss"].tolist(),
        validation_generator_b_loss=data["validation_generator_b_loss"].tolist(),
        validation_d_a_loss=data["validation_d_a_loss"].tolist(),
        validation_d_b_loss=data["validation_d_b_loss"].tolist(),
        validation_structure_loss=data["validation_structure_loss"].tolist(),
        output_path_template=output_path_template,
        compute_fid_iou=compute_fid_iou,
        plot=plot,
    )


def train_spresgan(
    synthetic_data_root_directory: str,
    real_data_root_directory: str,
    synthetic_validation_data_root_directory: str,
    real_validation_data_root_directory: str,
    segmentor_model_path: str = "models/segmentor/best/best.model",
    target_image_size: int = TARGET_IMAGE_SIZE,
    num_classes: int = len(CLASS_TO_SEMANTIC_INDEX_MAPPING),
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    num_epochs: int = 200,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 5.0,
    lambda_structure_start: float = 0.0,
    lambda_structure_end: float = 1.0,
    lambda_structure_warmup_start_epoch: int = 25,
    lambda_structure_warmup_end_epoch: int = 75,
    validation_frequency: int = 5,
    compute_validation_fid_iou: bool = False,
    image_mask_as_generator_input: bool = True,
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

    # load synthetic and real train datasets and make DataLoaders
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
        paired,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
    )
    print(
        f"{len(synthetic_dataset)} synthetic, {len(real_dataset)} real -> {len(paired)} pairs/epoch",
        flush=True,
    )

    # load synthetic and real validation datasets and make DataLoaders
    synthetic_validation_dataset: PCBSPresGANSyntheticDataset = (
        PCBSPresGANSyntheticDataset(
            synthetic_validation_data_root_directory, target_image_size
        )
    )
    synthetic_validation_loader: DataLoader = DataLoader(
        synthetic_validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    real_validation_dataset: PCBSPresGANRealDataset = PCBSPresGANRealDataset(
        real_validation_data_root_directory, target_image_size
    )
    real_validation_loader: DataLoader = DataLoader(
        real_validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    print(
        f"{len(synthetic_validation_dataset)} synthetic validation, "
        f"{len(real_validation_dataset)} real validation",
        flush=True,
    )

    fid_metric: FrechetInceptionDistance | None = None
    if compute_validation_fid_iou:
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    # model initialization
    generator_in_channels: int = 4 if image_mask_as_generator_input else 3
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

    adversarial_loss: nn.MSELoss = nn.MSELoss()
    cycle_loss: nn.L1Loss = nn.L1Loss()
    identity_loss: nn.L1Loss = nn.L1Loss()
    structure_loss_function: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

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

    lambda_structure_schedule: Callable = linear_lambda_stucture_schedule(
        lambda_start=lambda_structure_start,
        lambda_end=lambda_structure_end,
        warmup_start_epoch=lambda_structure_warmup_start_epoch,
        warmup_end_epoch=lambda_structure_warmup_end_epoch,
    )

    generator_a_losses: np.ndarray = np.zeros(num_epochs)
    generator_b_losses: np.ndarray = np.zeros(num_epochs)
    d_a_losses: np.ndarray = np.zeros(num_epochs)
    d_b_losses: np.ndarray = np.zeros(num_epochs)
    structure_losses: np.ndarray = np.zeros(num_epochs)

    validation_epochs: list[int] = []
    validation_iou_values: list[float] = []
    validation_fid_values: list[float] = []
    validation_generator_a_loss_values: list[float] = []
    validation_generator_b_loss_values: list[float] = []
    validation_d_a_loss_values: list[float] = []
    validation_d_b_loss_values: list[float] = []
    validation_structure_loss_values: list[float] = []

    start_epoch: int = 0
    total_epochs_ran: int = 0
    best_validation_iou: float = -float("inf")
    best_validation_fid: float = float("inf")

    # loading from checkpoint
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

        generator_a_losses[: len(checkpoint["generator_a_losses"])] = checkpoint[
            "generator_a_losses"
        ]
        generator_b_losses[: len(checkpoint["generator_b_losses"])] = checkpoint[
            "generator_b_losses"
        ]
        d_a_losses[: len(checkpoint["d_a_losses"])] = checkpoint["d_a_losses"]
        d_b_losses[: len(checkpoint["d_b_losses"])] = checkpoint["d_b_losses"]
        if checkpoint.get("structure_losses") is not None:
            structure_losses[: len(checkpoint["structure_losses"])] = checkpoint[
                "structure_losses"
            ]
        if checkpoint.get("validation_epochs") is not None:
            validation_epochs = list(checkpoint["validation_epochs"])
            validation_iou_values = list(checkpoint["validation_iou_values"])
            validation_fid_values = list(checkpoint["validation_fid_values"])
            validation_generator_a_loss_values = list(
                checkpoint.get("validation_generator_a_loss_values", [])
            )
            validation_generator_b_loss_values = list(
                checkpoint.get("validation_generator_b_loss_values", [])
            )
            validation_d_a_loss_values = list(
                checkpoint.get("validation_d_a_loss_values", [])
            )
            validation_d_b_loss_values = list(
                checkpoint.get("validation_d_b_loss_values", [])
            )
            validation_structure_loss_values = list(
                checkpoint.get("validation_structure_loss_values", [])
            )
        if checkpoint.get("best_validation_iou") is not None:
            best_validation_iou = checkpoint["best_validation_iou"]
        if checkpoint.get("best_validation_fid") is not None:
            best_validation_fid = checkpoint["best_validation_fid"]

        print(
            f"Resumed from epoch {total_epochs_ran} ({resume_checkpoint_path})",
            flush=True,
        )

    model_name_no_epoch: str = (
        MODEL_NAME_TEMPLATE.replace("{{ model_name }}", "SPresGAN")
        .replace("{{ batch_size }}", str(batch_size))
        .replace("{{ learning_rate }}", str(learning_rate))
    )

    start_time: float = time.perf_counter()

    # training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_generator_a_loss: float = 0.0
        epoch_generator_b_loss: float = 0.0
        epoch_d_a_loss: float = 0.0
        epoch_d_b_loss: float = 0.0
        epoch_loss_structure: float = 0.0
        num_batches: int = 0

        curr_lambda_structure: float = lambda_structure_schedule(epoch)

        epoch_start_time: float = time.perf_counter()
        for batch in loader:
            real_a: torch.Tensor = batch["real_a"].to(device)
            real_b: torch.Tensor = batch["real_b"].to(device)
            mask_a: torch.Tensor = batch["mask_a"].to(device)

            if image_mask_as_generator_input:
                mask_b_predicted: torch.Tensor = predict_binary_mask_segmentor(
                    segmentor, real_b
                )

            with torch.no_grad():
                probe_shape: torch.Size = d_b(real_b).shape
            valid: torch.Tensor = torch.ones(probe_shape, device=device)
            fake_label: torch.Tensor = torch.zeros(probe_shape, device=device)

            opt_g.zero_grad()

            if image_mask_as_generator_input:
                fake_b: torch.Tensor = g_a_to_b(torch.cat([real_a, mask_a], dim=1))
                fake_a: torch.Tensor = g_b_to_a(
                    torch.cat([real_b, mask_b_predicted], dim=1)
                )
            else:
                fake_b: torch.Tensor = g_a_to_b(real_a)
                fake_a: torch.Tensor = g_b_to_a(real_b)

            gan_a_to_b_loss: torch.Tensor = adversarial_loss(d_b(fake_b), valid)
            gan_b_to_a_loss: torch.Tensor = adversarial_loss(d_a(fake_a), valid)

            if image_mask_as_generator_input:
                recovered_a: torch.Tensor = g_b_to_a(torch.cat([fake_b, mask_a], dim=1))
                recovered_b: torch.Tensor = g_a_to_b(
                    torch.cat([fake_a, mask_b_predicted], dim=1)
                )
                identity_a: torch.Tensor = g_b_to_a(torch.cat([real_a, mask_a], dim=1))
                identity_b: torch.Tensor = g_a_to_b(
                    torch.cat([real_b, mask_b_predicted], dim=1)
                )
            else:
                recovered_a: torch.Tensor = g_b_to_a(fake_b)
                recovered_b: torch.Tensor = g_a_to_b(fake_a)
                identity_a: torch.Tensor = g_b_to_a(real_a)
                identity_b: torch.Tensor = g_a_to_b(real_b)

            cycle_a_loss: torch.Tensor = cycle_loss(recovered_a, real_a)
            cycle_b_loss: torch.Tensor = cycle_loss(recovered_b, real_b)
            identity_a_loss: torch.Tensor = identity_loss(identity_a, real_a)
            identity_b_loss: torch.Tensor = identity_loss(identity_b, real_b)

            predicted_logit: torch.Tensor = predict_foreground_logit_segmentor(
                segmentor, fake_b
            )
            loss_structure: torch.Tensor = structure_loss_function(
                predicted_logit, mask_a
            )

            generator_a_loss: torch.Tensor = (
                gan_a_to_b_loss + curr_lambda_structure * loss_structure
            )
            generator_b_loss: torch.Tensor = gan_b_to_a_loss

            g_loss: torch.Tensor = (
                generator_a_loss
                + generator_b_loss
                + lambda_cycle * (cycle_a_loss + cycle_b_loss)
                + lambda_identity * (identity_a_loss + identity_b_loss)
            )
            g_loss.backward()
            opt_g.step()

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

            epoch_generator_a_loss += generator_a_loss.item()
            epoch_generator_b_loss += generator_b_loss.item()
            epoch_d_a_loss += d_a_loss.item()
            epoch_d_b_loss += d_b_loss.item()
            epoch_loss_structure += loss_structure.item()
            num_batches += 1

        scheduler_g.step()
        scheduler_d.step()

        generator_a_losses[epoch] = epoch_generator_a_loss / num_batches
        generator_b_losses[epoch] = epoch_generator_b_loss / num_batches
        d_a_losses[epoch] = epoch_d_a_loss / num_batches
        d_b_losses[epoch] = epoch_d_b_loss / num_batches
        structure_losses[epoch] = epoch_loss_structure / num_batches

        total_epochs_ran += 1

        epoch_end_time: float = time.perf_counter()

        print(
            (
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"generator_a_loss={generator_a_losses[epoch]:.4f} "
                f"generator_b_loss={generator_b_losses[epoch]:.4f} "
                f"d_a_loss={d_a_losses[epoch]:.4f} d_b_loss={d_b_losses[epoch]:.4f} "
                f"loss_structure={structure_losses[epoch]:.4f} "
                f"lambda_structure={curr_lambda_structure:.4f} "
                f"Epoch training time={(epoch_end_time - epoch_start_time):.4f} s"
            ),
            flush=True,
        )

        if (epoch + 1) % validation_frequency == 0:
            validation_metrics: dict[str, float] = evaluate_spresgan(
                g_a_to_b,
                g_b_to_a,
                d_a,
                d_b,
                segmentor,
                synthetic_validation_loader,
                real_validation_loader,
                adversarial_loss,
                structure_loss_function,
                curr_lambda_structure,
                device,
                image_mask_as_generator_input=image_mask_as_generator_input,
                compute_fid_iou=compute_validation_fid_iou,
                fid_metric=fid_metric,
            )

            validation_epochs.append(epoch + 1)
            validation_generator_a_loss_values.append(
                validation_metrics["validation_generator_a_loss"]
            )
            validation_generator_b_loss_values.append(
                validation_metrics["validation_generator_b_loss"]
            )
            validation_d_a_loss_values.append(validation_metrics["validation_d_a_loss"])
            validation_d_b_loss_values.append(validation_metrics["validation_d_b_loss"])
            validation_structure_loss_values.append(
                validation_metrics["validation_structure_loss"]
            )

            print(
                f"Epoch {epoch + 1} validation | "
                f"Gen A: {validation_metrics['validation_generator_a_loss']:.4f} "
                f"Gen B: {validation_metrics['validation_generator_b_loss']:.4f} "
                f"D_A: {validation_metrics['validation_d_a_loss']:.4f} "
                f"D_B: {validation_metrics['validation_d_b_loss']:.4f} "
                f"Structure: {validation_metrics['validation_structure_loss']:.4f}",
                flush=True,
            )

            if compute_validation_fid_iou:
                validation_iou_values.append(validation_metrics["validation_iou"])
                validation_fid_values.append(validation_metrics["validation_fid"])

                print(
                    f"Epoch {epoch + 1} validation | "
                    f"IoU: {validation_metrics['validation_iou']:.4f} "
                    f"FID: {validation_metrics['validation_fid']:.4f}",
                    flush=True,
                )

                if validation_metrics["validation_iou"] > best_validation_iou:
                    best_validation_iou = validation_metrics["validation_iou"]
                    print(
                        f"New best validation_iou {best_validation_iou:.4f} at epoch {epoch + 1}",
                        flush=True,
                    )

                if validation_metrics["validation_fid"] < best_validation_fid:
                    best_validation_fid = validation_metrics["validation_fid"]
                    print(
                        f"New best validation_fid {best_validation_fid:.4f} at epoch {epoch + 1}",
                        flush=True,
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
                    "generator_a_losses": generator_a_losses[: epoch + 1],
                    "generator_b_losses": generator_b_losses[: epoch + 1],
                    "d_a_losses": d_a_losses[: epoch + 1],
                    "d_b_losses": d_b_losses[: epoch + 1],
                    "structure_losses": structure_losses[: epoch + 1],
                    "validation_epochs": validation_epochs,
                    "validation_iou_values": validation_iou_values,
                    "validation_fid_values": validation_fid_values,
                    "validation_generator_a_loss_values": validation_generator_a_loss_values,
                    "validation_generator_b_loss_values": validation_generator_b_loss_values,
                    "validation_d_a_loss_values": validation_d_a_loss_values,
                    "validation_d_b_loss_values": validation_d_b_loss_values,
                    "validation_structure_loss_values": validation_structure_loss_values,  # NEW
                    "best_validation_iou": best_validation_iou,
                    "best_validation_fid": best_validation_fid,
                },
                os.path.join(
                    SPRESGAN_MODEL_CHECKPOINTS_DIRECTORY,
                    model_name_no_epoch.replace("{{ epoch }}", str(epoch + 1)),
                ),
            )

    end_time: float = time.perf_counter()
    print(f"Total time elapsed: {(end_time - start_time):.4f}s", flush=True)
    if compute_validation_fid_iou:
        print(f"Best validation IoU achieved: {best_validation_iou:.4f}", flush=True)
        print(f"Best validation FID achieved: {best_validation_fid:.4f}", flush=True)

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
        generator_a_loss=generator_a_losses[:total_epochs_ran],
        generator_b_loss=generator_b_losses[:total_epochs_ran],
        discriminator_a_loss=d_a_losses[:total_epochs_ran],
        discriminator_b_loss=d_b_losses[:total_epochs_ran],
        structural_loss=structure_losses[:total_epochs_ran],
        validation_epochs=validation_epochs,
        validation_iou=validation_iou_values,
        validation_fid=validation_fid_values,
        validation_generator_a_loss=validation_generator_a_loss_values,
        validation_generator_b_loss=validation_generator_b_loss_values,
        validation_d_a_loss=validation_d_a_loss_values,
        validation_d_b_loss=validation_d_b_loss_values,
        validation_structure_loss=validation_structure_loss_values,
        output_path_template=os.path.join(
            SPRESGAN_MODEL_TRAINING_CURVE_DIRECTORY,
            TRAINING_CURVE_FILE_NAME_TEMPLATE.replace("{{ model_name }}", "SPresGAN")
            .replace("{{ batch_size }}", str(batch_size))
            .replace("{{ learning_rate }}", str(learning_rate)),
        ),
        compute_fid_iou=compute_validation_fid_iou,
    )


if __name__ == "__main__":
    train_spresgan(
        "data/synthetic_split/train",
        "data/real_images_split/train",
        "data/synthetic_split/validation",
        "data/real_images_split/validation",
        segmentor_model_path="models/segmentor/best/UNetSegmentor_BaseChannels128_bs8_lr0.001_best.model",
        num_epochs=200,
        lambda_structure_start=0.0,
        lambda_structure_end=2.5,
        lambda_structure_warmup_start_epoch=25,
        lambda_structure_warmup_end_epoch=75,
        compute_validation_fid_iou=False,
        image_mask_as_generator_input=False,
    )
