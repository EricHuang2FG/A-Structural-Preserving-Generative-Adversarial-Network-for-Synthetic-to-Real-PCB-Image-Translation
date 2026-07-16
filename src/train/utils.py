import numpy as np
import matplotlib.pyplot as plt


def plot_training_validation_curves(
    train_error: np.ndarray,
    train_loss: np.ndarray,
    validation_error: np.ndarray,
    validation_loss: np.ndarray,
    output_path_template: str,  # must have {{ type }} in the string
) -> None:
    # plot the error curves
    plt.figure()
    plt.title("Train and Validation Error vs. Epochs")
    num_epochs: int = len(train_error) + 1
    plt.plot(range(1, num_epochs), train_error, label="Train Error")
    plt.plot(range(1, num_epochs), validation_error, label="Validation Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc="best")
    plt.savefig(output_path_template.replace("{{ type }}", "error"))
    plt.show()

    # plot the loss curves
    plt.figure()
    plt.title("Train and Validation Loss vs. Epochs")
    plt.plot(range(1, num_epochs), train_loss, label="Train Loss")
    plt.plot(range(1, num_epochs), validation_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(output_path_template.replace("{{ type }}", "loss"))
    plt.show()

    # save the raw metrics
    metrics_path: str = output_path_template.replace("{{ type }}", "metrics")
    metrics_path = metrics_path.rsplit(".", 1)[0] + ".npz"

    np.savez(
        metrics_path,
        train_error=train_error,
        validation_error=validation_error,
        train_loss=train_loss,
        validation_loss=validation_loss,
    )
