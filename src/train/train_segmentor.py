import os
import time
import torch
import numpy as np

from torch.utils.data import random_split, DataLoader

from src.data_processing.datasets import PCBSegmentorDataset
from src.model.segmentor import UNetSegmentor
from src.train.utils import plot_training_validation_curves
from src.inference.inference_segmentor import evaluate_segmentor
from src.utils.constants import (
    CLASS_TO_SEMANTIC_INDEX_MAPPING,
    SEED,
    TARGET_IMAGE_SIZE,
    EARLY_STOPPING_PATIENCE,
    MODEL_NAME_TEMPLATE,
    TRAINING_CURVE_FILE_NAME_TEMPLATE,
    SEGMENTOR_MODEL_CHECKPOINTS_DIRECTORY,
    SEGMENTOR_MODEL_BEST_MODEL_DIRECTORY,
    SEGMENTOR_MODEL_TRAINING_CURVE_DIRECTORY,
)


def train_segmentor(
    model: UNetSegmentor,
    data_root_directory: str,
    num_classes: int = len(CLASS_TO_SEMANTIC_INDEX_MAPPING),
    validation_split_fraction: float = 0.1,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    num_epochs: int = 30,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    target_image_size: int = TARGET_IMAGE_SIZE,
) -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    os.makedirs(SEGMENTOR_MODEL_BEST_MODEL_DIRECTORY, exist_ok=True)
    os.makedirs(SEGMENTOR_MODEL_CHECKPOINTS_DIRECTORY, exist_ok=True)
    os.makedirs(SEGMENTOR_MODEL_TRAINING_CURVE_DIRECTORY, exist_ok=True)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # dataset preparation
    dataset: PCBSegmentorDataset = PCBSegmentorDataset(
        data_root_directory, target_image_size=target_image_size
    )
    dataset_size: int = len(dataset)

    validation_size: int = int(dataset_size * validation_split_fraction)
    train_size: int = dataset_size - validation_size

    train_dataset: PCBSegmentorDataset
    validation_dataset: PCBSegmentorDataset
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"{train_size} training data, {validation_size} validation data")

    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    validation_loader: DataLoader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    # training loop
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_error: np.ndarray = np.zeros(num_epochs)
    validation_error: np.ndarray = np.zeros(num_epochs)
    train_loss: np.ndarray = np.zeros(num_epochs)
    validation_loss: np.ndarray = np.zeros(num_epochs)

    best_validation_loss: float = float("inf")
    epochs_without_improvement: int = 0

    start_time: float = time.perf_counter()
    total_epochs_ran: int = 0

    epoch: int
    for epoch in range(num_epochs):
        model.train()

        curr_train_loss: float = 0.0
        curr_train_error: float = 0.0
        total_num_pixels: int = 0

        images: torch.Tensor
        masks: torch.Tensor
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits: torch.Tensor = model(images)

            loss: torch.Tensor = UNetSegmentor.criterion(
                logits, masks, num_classes=num_classes
            )
            loss.backward()
            optimizer.step()

            predictions: torch.Tensor = torch.argmax(logits, dim=1)
            num_pixels: int = masks.numel()
            total_num_pixels += num_pixels
            curr_train_error += torch.sum((predictions != masks).float()).item()
            curr_train_loss += loss.item() * num_pixels

        train_loss[epoch] = curr_train_loss / total_num_pixels
        train_error[epoch] = curr_train_error / total_num_pixels
        validation_loss[epoch], validation_error[epoch] = evaluate_segmentor(
            model, validation_loader, device, num_classes=num_classes
        )

        total_epochs_ran += 1

        print(
            f"Epoch {epoch + 1}: Train err: {train_error[epoch]:.4f}, Train loss: {train_loss[epoch]:.4f} | "
            f"Validation err: {validation_error[epoch]:.4f}, Validation loss: {validation_loss[epoch]:.4f}"
        )

        # save checkpoint every 10 epochs
        model_name_no_epoch: str = (
            MODEL_NAME_TEMPLATE.replace(
                "{{ model_name }}",
                f"{model.name}",
            )
            .replace("{{ batch_size }}", str(batch_size))
            .replace("{{ learning_rate }}", str(learning_rate))
        )
        if epoch % 10 == 0 and epoch != 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    SEGMENTOR_MODEL_CHECKPOINTS_DIRECTORY,
                    model_name_no_epoch.replace("{{ epoch }}", str(epoch + 1)),
                ),
            )

        # check if the current model is the best model and perform
        # early stopping check
        if validation_loss[epoch] < best_validation_loss:
            best_validation_loss = validation_loss[epoch]
            epochs_without_improvement = 0
            torch.save(
                model.state_dict(),
                os.path.join(
                    SEGMENTOR_MODEL_BEST_MODEL_DIRECTORY,
                    model_name_no_epoch.replace("_epoch{{ epoch }}", "_best"),
                ),
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping on epoch {epoch + 1}")
                break

    end_time: float = time.perf_counter()
    print(f"Total time elapsed: {(end_time - start_time):.4f}")

    # plot the training and validation curves
    plot_training_validation_curves(
        train_error[:total_epochs_ran],
        train_loss[:total_epochs_ran],
        validation_error[:total_epochs_ran],
        validation_loss[:total_epochs_ran],
        TRAINING_CURVE_FILE_NAME_TEMPLATE.replace(
            "{{ model_name }}",
            f"{model.name}",
        )
        .replace("{{ batch_size }}", str(batch_size))
        .replace("{{ learning_rate }}", str(learning_rate)),
    )


if __name__ == "__main__":
    train_segmentor(UNetSegmentor(), "data/synthetic_split/train")
