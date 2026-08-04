from src.model.segmentor import UNetSegmentor
from src.train.train_segmentor import train_segmentor
from src.utils.constants import SEGMENTOR_MODEL_SWEEP_CURVE_DIRECTORY


def sweep_learning_rate(
    candidate_lrs: list[float], batch_size: int = 16, base_channel_size: int = 64
) -> None:
    best_validation_loss: float = float("inf")
    best_lr_by_validation_loss: float = candidate_lrs[0]

    print(f"Sweeping UNetSegmentor learning rates: {candidate_lrs}")

    lr: float
    for lr in candidate_lrs:
        curr_validation_loss: float = train_segmentor(
            UNetSegmentor(base_channels=base_channel_size),
            "data/synthetic_split/train",
            learning_rate=lr,
            batch_size=batch_size,
            num_epochs=10,
            early_stopping_patience=0,
            training_curve_output_directory=SEGMENTOR_MODEL_SWEEP_CURVE_DIRECTORY,
        )
        if curr_validation_loss < best_validation_loss:
            best_validation_loss = curr_validation_loss
            best_lr_by_validation_loss = lr

        print(f"Learning rate {lr} final validation loss: {curr_validation_loss}")

    print(
        f"Lowest validation loss {best_validation_loss} achieved by a learing rate of {best_lr_by_validation_loss}"
    )


def sweep_batch_size(
    candidate_batch_sizes: list[int],
    learning_rate: float = 1e-4,
    base_channel_size: int = 64,
) -> None:
    best_validation_loss: float = float("inf")
    best_batch_size_by_validation_loss: int = candidate_batch_sizes[0]

    print(f"Sweeping UNetSegmentor batch sizes: {candidate_batch_sizes}")

    batch_size: float
    for batch_size in candidate_batch_sizes:
        curr_validation_loss: float = train_segmentor(
            UNetSegmentor(base_channels=base_channel_size),
            "data/synthetic_split/train",
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=10,
            early_stopping_patience=0,
            training_curve_output_directory=SEGMENTOR_MODEL_SWEEP_CURVE_DIRECTORY,
        )
        if curr_validation_loss < best_validation_loss:
            best_validation_loss = curr_validation_loss
            best_batch_size_by_validation_loss = batch_size

        print(f"Batch size {batch_size} final validation loss: {curr_validation_loss}")

    print(
        f"Lowest validation loss {best_validation_loss} achieved by a batch size of {best_batch_size_by_validation_loss}"
    )


def sweep_base_channel_size(
    candidate_base_channel_sizes: list[int],
    learning_rate: float = 1e-4,
    batch_size: int = 16,
) -> None:
    best_validation_loss: float = float("inf")
    best_base_channel_size_by_validation_loss: int = candidate_base_channel_sizes[0]

    print(f"Sweeping UNetSegmentor base channel size: {candidate_base_channel_sizes}")

    base_channel_size: float
    for base_channel_size in candidate_base_channel_sizes:
        curr_validation_loss: float = train_segmentor(
            UNetSegmentor(base_channels=base_channel_size),
            "data/synthetic_split/train",
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=10,
            early_stopping_patience=0,
            training_curve_output_directory=SEGMENTOR_MODEL_SWEEP_CURVE_DIRECTORY,
        )
        if curr_validation_loss < best_validation_loss:
            best_validation_loss = curr_validation_loss
            best_base_channel_size_by_validation_loss = base_channel_size

        print(
            f"Base channel size {base_channel_size} final validation loss: {curr_validation_loss}"
        )

    print(
        f"Lowest validation loss {best_validation_loss} achieved by a base channel size of {best_base_channel_size_by_validation_loss}"
    )


if __name__ == "__main__":
    sweep_learning_rate([1e-4, 1e-3, 1e-2], batch_size=16, base_channel_size=64)
