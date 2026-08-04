import os
import re
import cv2

import numpy as np

from src.utils.constants import BOARD_RENDER_WIDTH, BOARD_RENDER_HEIGHT


def remove_background_rotate_axis_aligned(
    image: np.ndarray,
    mask: np.ndarray,
    crop_padding_px: int = 10,
) -> np.ndarray:
    height: int
    width: int
    height, width = image.shape[:2]

    # ground-truth mask to binary
    foreground_mask: np.ndarray
    _, foreground_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if foreground_mask.shape[:2] != (height, width):
        foreground_mask = cv2.resize(
            foreground_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )

    contours: list[np.ndarray]
    contours, _ = cv2.findContours(
        foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return image

    # compute rotation from the ground-truth mask's contour
    largest_contour: np.ndarray = max(contours, key=cv2.contourArea)
    hull: np.ndarray = cv2.convexHull(largest_contour)

    rect: tuple[tuple[float, float], tuple[float, float], float] = cv2.minAreaRect(hull)
    angle: float = rect[2]

    angle = angle % 90.0
    if angle > 45:
        angle -= 90

    rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(
        (width / 2, height / 2), angle, 1.0
    )

    rotated_image: np.ndarray = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    rotated_mask: np.ndarray = cv2.warpAffine(
        foreground_mask,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    rotated_contours: list[np.ndarray]
    rotated_contours, _ = cv2.findContours(
        rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not rotated_contours:
        return image

    x: int
    y: int
    w: int
    h: int
    x, y, w, h = cv2.boundingRect(max(rotated_contours, key=cv2.contourArea))

    x = max(0, x - crop_padding_px)
    y = max(0, y - crop_padding_px)
    w = min(width - x, w + 2 * crop_padding_px)
    h = min(height - y, h + 2 * crop_padding_px)

    cropped_image: np.ndarray = rotated_image[y : y + h, x : x + w]
    cropped_mask: np.ndarray = rotated_mask[y : y + h, x : x + w]

    result: np.ndarray = np.full_like(cropped_image, 255)
    result[cropped_mask > 0] = cropped_image[cropped_mask > 0]

    return result


def process_real_images(
    source_directory: str,
    destination_directory: str,
    crop_padding_px: int = 10,
) -> None:
    os.makedirs(destination_directory, exist_ok=True)

    pattern: re.Pattern[str] = re.compile(r"^rec(\d+)\.jpg$", re.IGNORECASE)

    counter: int = 1

    root: str
    files: list[str]
    for root, _, files in os.walk(source_directory):
        filename: str
        for filename in sorted(files):
            match: re.Match[str] | None = pattern.match(filename)
            if not match:
                continue

            image_number: str = match.group(1)
            image_path: str = os.path.join(root, filename)
            mask_path: str = os.path.join(root, f"rec{image_number}-mask.png")

            if not os.path.exists(mask_path):
                print(
                    f"Skip processing {filename}: no matching mask found at {mask_path}"
                )
                continue

            image: np.ndarray | None = cv2.imread(image_path)
            mask: np.ndarray | None = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                continue

            image = remove_background_rotate_axis_aligned(
                image, mask, crop_padding_px=crop_padding_px
            )

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

            print(f"Real PCB image {counter} finished processing")

            counter += 1


if __name__ == "__main__":
    process_real_images("data/PCB-DSLR", "data/real_images")
