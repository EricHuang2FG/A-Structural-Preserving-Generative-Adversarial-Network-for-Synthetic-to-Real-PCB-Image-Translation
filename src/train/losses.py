import torch


def dice_loss(
    logits: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    num_classes: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    probabilities: torch.Tensor = torch.softmax(logits, dim=1)
    targets_onehot: torch.Tensor = (
        torch.nn.functional.one_hot(ground_truth_mask, num_classes)
        .permute(0, 3, 1, 2)
        .float()
    )
    intersection: torch.Tensor = (probabilities * targets_onehot).sum(dim=(2, 3))
    union: torch.Tensor = probabilities.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))

    return 1.0 - ((2 * intersection + epsilon) / (union + epsilon)).mean()
