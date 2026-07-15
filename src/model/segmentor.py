import torch
import torch.nn as nn


from src.utils.constants import CLASS_TO_SEMANTIC_INDEX_MAPPING
from src.train.metrics import dice_loss


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv_block: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool: nn.MaxPool2d = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip: torch.Tensor = self.conv_block(x)
        return self.pool(skip), skip


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.upsample: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv_block: nn.Sequential = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)

        return self.conv_block(x)


class UNetSegmentor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = len(CLASS_TO_SEMANTIC_INDEX_MAPPING),
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.name: str = "UNetSegmentor"
        self.downsample_layer_1: DownsampleBlock = DownsampleBlock(
            in_channels, base_channels
        )
        self.downsample_layer_2: DownsampleBlock = DownsampleBlock(
            base_channels, base_channels * 2
        )
        self.downsample_layer_3: DownsampleBlock = DownsampleBlock(
            base_channels * 2, base_channels * 4
        )
        self.downsample_layer_4: DownsampleBlock = DownsampleBlock(
            base_channels * 4, base_channels * 8
        )

        self.bottleneck: nn.Sequential = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
        )

        self.upsample_layer_1: UpsampleBlock = UpsampleBlock(
            base_channels * 16, base_channels * 8
        )
        self.upsample_layer_2: UpsampleBlock = UpsampleBlock(
            base_channels * 8, base_channels * 4
        )
        self.upsample_layer_3: UpsampleBlock = UpsampleBlock(
            base_channels * 4, base_channels * 2
        )
        self.upsample_layer_4: UpsampleBlock = UpsampleBlock(
            base_channels * 2, base_channels
        )
        self.output_conv: nn.Conv2d = nn.Conv2d(
            base_channels, num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1: torch.Tensor
        skip_1: torch.Tensor
        x_1, skip_1 = self.downsample_layer_1(x)

        x_2: torch.Tensor
        skip_2: torch.Tensor
        x_2, skip_2 = self.downsample_layer_2(x_1)

        x_3: torch.Tensor
        skip_3: torch.Tensor
        x_3, skip_3 = self.downsample_layer_3(x_2)

        x_4: torch.Tensor
        skip_4: torch.Tensor
        x_4, skip_4 = self.downsample_layer_4(x_3)

        b: torch.Tensor = self.bottleneck(x_4)

        x = self.upsample_layer_1(b, skip_4)
        x = self.upsample_layer_2(x, skip_3)
        x = self.upsample_layer_3(x, skip_2)
        x = self.upsample_layer_4(x, skip_1)

        return self.output_conv(x)

    @staticmethod
    def criterion(
        logits: torch.Tensor,
        ground_truth_mask: torch.Tensor,
        num_classes: int = len(CLASS_TO_SEMANTIC_INDEX_MAPPING),
    ) -> torch.Tensor:
        cross_entropy_loss: torch.Tensor = nn.functional.cross_entropy(
            logits, ground_truth_mask
        )
        return cross_entropy_loss + dice_loss(
            logits, ground_truth_mask, num_classes=num_classes
        )
