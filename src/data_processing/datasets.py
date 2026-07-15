import os
import cv2

import torch
import numpy as np

from torch.utils.data import Dataset

from src.utils.constants import TARGET_IMAGE_SIZE


class PCBSegmentorDataset(Dataset):
    def __init__(
        self, root_directory: str, target_image_size: int = TARGET_IMAGE_SIZE
    ) -> None:
        self.target_image_size: int = target_image_size
        self.data_paths: list[tuple[str, str]] = PCBSegmentorDataset._get_data(
            root_directory
        )

    @staticmethod
    def _get_data(root_directory: str) -> None:
        data_paths: list[tuple[str, str]] = []
        skipped_data: list[str] = []

        pcb_folder: str
        for pcb_folder in sorted(os.listdir(root_directory)):
            pcb_directory: str = os.path.join(root_directory, pcb_folder)

            if not os.path.isdir(pcb_directory):
                continue

            # sample top side only
            image_path: str = os.path.join(pcb_directory, "top_image.png")
            mask_path: str = os.path.join(pcb_directory, "top_semantic_mask.png")

            if os.path.exists(image_path) and os.path.exists(mask_path):
                data_paths.append((image_path, mask_path))
            else:
                skipped_data.append(pcb_directory)

        if skipped_data:
            print(f"{len(skipped_data)} images skipped: {skipped_data}")

        return data_paths

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path: str
        mask_path: str
        image_path, mask_path = self.data_paths[index]

        image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 4:
            image = image[:, :, :3]  # if RGBA, remove alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask: np.ndarray = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        image = cv2.resize(
            image,
            (self.target_image_size, self.target_image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (self.target_image_size, self.target_image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        # normalize to between [-1, 1]
        # convert from BGR to RGB
        image_tensor: torch.Tensor = (
            torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1.0
        )
        mask_tensor: torch.Tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor
