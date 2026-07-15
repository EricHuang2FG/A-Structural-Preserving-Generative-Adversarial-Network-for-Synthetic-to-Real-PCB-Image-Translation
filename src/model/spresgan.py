import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.res_block: nn.Sequential = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_block(x)


class ResNetGenerator(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_residual_blocks: int = 9,
        downsampling_in_channels: int = 64,
        num_downsampling_layers: int = 2,
        num_upsampling_layers: int = 2,
        out_channels: int = 3,
    ) -> None:
        super().__init__()

        generator_layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, downsampling_in_channels, kernel_size=7),
            nn.InstanceNorm2d(downsampling_in_channels),
            nn.ReLU(inplace=True),
        ]

        # downsampling layers
        for _ in range(num_downsampling_layers):
            downsampling_out_channels: int = downsampling_in_channels * 2
            generator_layers.extend(
                [
                    nn.Conv2d(
                        downsampling_in_channels,
                        downsampling_out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.InstanceNorm2d(downsampling_out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            downsampling_in_channels = downsampling_out_channels

        # ResNet blocks
        for _ in range(num_residual_blocks):
            generator_layers.append(ResNetBlock(downsampling_out_channels))

        # upsampling layers
        upsampling_in_channels: int = downsampling_out_channels
        for _ in range(num_upsampling_layers):
            upsampling_out_channels: int = upsampling_in_channels // 2
            generator_layers.extend(
                [
                    nn.ConvTranspose2d(
                        upsampling_in_channels,
                        upsampling_out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.InstanceNorm2d(upsampling_out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            upsampling_in_channels = upsampling_out_channels

        # output layers
        generator_layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(upsampling_out_channels, out_channels, kernel_size=7),
                nn.Tanh(),
            ]
        )

        self.generator: nn.Sequential = nn.Sequential(*generator_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


class PatchGANDiscriminator(nn.Module):

    def __init__(self, in_channels: int = 3, use_bias: bool = True) -> None:
        super().__init__()

        self.discriminator: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels, 64, kernel_size=4, stride=2, padding=1, bias=use_bias
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 512, kernel_size=4, padding=1, bias=use_bias),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=use_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)
