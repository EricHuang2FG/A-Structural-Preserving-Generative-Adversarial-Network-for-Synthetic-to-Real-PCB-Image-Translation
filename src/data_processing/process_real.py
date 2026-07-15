import os
import re
import cv2

import numpy as np

from src.utils.constants import BOARD_RENDER_WIDTH, BOARD_RENDER_HEIGHT


def process_real_images(source_directory: str, destination_directory: str) -> None:
    os.makedirs(destination_directory, exist_ok=True)

    pattern: re.Pattern[str] = re.compile(r"^rec\d+\.jpg$", re.IGNORECASE)

    counter: int = 1

    root: str
    files: list[str]
    for root, _, files in os.walk(source_directory):
        filename: str
        for filename in sorted(files):
            if pattern.match(filename):
                src: str = os.path.join(root, filename)

                image: np.ndarray = cv2.imread(src)

                if image is None:
                    continue

                height: int
                width: int
                height, width = image.shape[:2]

                scale: float = (BOARD_RENDER_WIDTH - 16) / max(height, width)

                new_width: int = int(width * scale)
                new_height: int = int(height * scale)

                image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

                canvas: np.ndarray = (
                    np.ones(
                        (BOARD_RENDER_WIDTH - 16, BOARD_RENDER_HEIGHT - 16, 3),
                        dtype=np.uint8,
                    )
                    * 255
                )

                x_offset: int = (BOARD_RENDER_WIDTH - 16 - new_width) // 2
                y_offset: int = (BOARD_RENDER_HEIGHT - 16 - new_height) // 2

                canvas[
                    y_offset : y_offset + new_height,
                    x_offset : x_offset + new_width,
                ] = image

                destination: str = os.path.join(
                    destination_directory,
                    f"{counter}_{filename}",
                )

                cv2.imwrite(destination, canvas)

                counter += 1


if __name__ == "__main__":
    process_real_images("data/PCB-DSLR", "data/real_images")
