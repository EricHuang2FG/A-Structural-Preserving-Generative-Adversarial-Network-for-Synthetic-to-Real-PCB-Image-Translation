import os
import cv2

import torch
import numpy as np

from torch.utils.data import Dataset

from src.utils.utils import image_to_tensor, mask_to_binary_tensor
from src.utils.constants import TARGET_IMAGE_SIZE
from src.data_processing.utils import (
    get_synthetic_data_paths_with_semantic_mask,
    get_real_image_paths,
)


class PCBSegmentorDataset(Dataset):

    def __init__(
        self, root_directory: str, target_image_size: int = TARGET_IMAGE_SIZE
    ) -> None:
        self.target_image_size: int = target_image_size
        self.data_paths: list[tuple[str, str]] = (
            get_synthetic_data_paths_with_semantic_mask(root_directory)
        )

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path: str
        mask_path: str
        image_path, mask_path = self.data_paths[index]

        image_tensor: torch.Tensor = image_to_tensor(
            image_path, size=self.target_image_size
        )
        mask: np.ndarray = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(
            mask,
            (self.target_image_size, self.target_image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_tensor: torch.Tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


class PCBSPresGANSyntheticDataset(Dataset):

    def __init__(
        self, root_directory: str, target_image_size: int = TARGET_IMAGE_SIZE
    ) -> None:
        self.target_image_size: int = target_image_size
        self.data_paths: list[tuple[str, str]] = (
            get_synthetic_data_paths_with_semantic_mask(root_directory)
        )

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path: str
        mask_path: str
        image_path, mask_path = self.data_paths[index]

        image_tensor: torch.Tensor = image_to_tensor(
            image_path, size=self.target_image_size
        )
        mask_tensor: torch.Tensor = mask_to_binary_tensor(
            mask_path, size=self.target_image_size
        )

        return image_tensor, mask_tensor


class PCBSPresGANRealDataset(Dataset):

    def __init__(
        self, root_directory: str, target_image_size: int = TARGET_IMAGE_SIZE
    ) -> None:
        self.target_image_size: int = target_image_size
        self.data_paths: list[str] = get_real_image_paths(root_directory)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return image_to_tensor(self.data_paths[index], size=self.target_image_size)


class PCBSPresGANUnpairedDomainPair(Dataset):

    def __init__(
        self,
        synthetic_dataset: PCBSPresGANSyntheticDataset,
        real_dataset: PCBSPresGANRealDataset,
    ) -> None:
        self.synthetic_dataset: PCBSPresGANSyntheticDataset = synthetic_dataset
        self.real_dataset: PCBSPresGANRealDataset = real_dataset

    def __len__(self) -> int:
        return max(len(self.synthetic_dataset), len(self.real_dataset))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        real_a: torch.Tensor
        mask_a: torch.Tensor
        real_a, mask_a = self.synthetic_dataset[index % len(self.synthetic_dataset)]
        real_b: torch.Tensor = self.real_dataset[index % len(self.real_dataset)]
        return {"real_a": real_a, "mask_a": mask_a, "real_b": real_b}
