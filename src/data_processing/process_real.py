import os
import re
import cv2

import numpy as np

from typing import Any

from src.utils.constants import BOARD_RENDER_WIDTH, BOARD_RENDER_HEIGHT


def remove_background_rotate_axis_aligned(
    image: np.ndarray,
    border_fraction: float = 0.04,
    center_fraction: float = 0.5,
    working_max_dimension: int = 900,
    grabcut_iterations: int = 8,
    crop_padding_px: int = 6,
) -> np.ndarray:
    height: int
    width: int
    height, width = image.shape[:2]

    scale: float = min(1.0, working_max_dimension / max(height, width))
    small_image: np.ndarray = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )
    small_height: int
    small_width: int
    small_height, small_width = small_image.shape[:2]

    mask: np.ndarray = np.full(
        (small_height, small_width), cv2.GC_PR_FGD, dtype=np.uint8
    )

    border_h: int = max(1, int(small_height * border_fraction))
    border_w: int = max(1, int(small_width * border_fraction))
    mask[0:border_h, :] = cv2.GC_PR_BGD
    mask[small_height - border_h :, :] = cv2.GC_PR_BGD
    mask[:, 0:border_w] = cv2.GC_PR_BGD
    mask[:, small_width - border_w :] = cv2.GC_PR_BGD

    center_h: int = int(small_height * center_fraction)
    center_w: int = int(small_width * center_fraction)
    y0: int = (small_height - center_h) // 2
    x0: int = (small_width - center_w) // 2
    mask[y0 : y0 + center_h, x0 : x0 + center_w] = cv2.GC_FGD

    bgd_model: np.ndarray = np.zeros((1, 65), dtype=np.float64)
    fgd_model: np.ndarray = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(
        small_image,
        mask,
        None,
        bgd_model,
        fgd_model,
        grabcut_iterations,
        cv2.GC_INIT_WITH_MASK,
    )

    foreground_mask_small: np.ndarray = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    close_kernel: np.ndarray = np.ones((15, 15), np.uint8)
    open_kernel: np.ndarray = np.ones((5, 5), np.uint8)
    foreground_mask_small = cv2.morphologyEx(
        foreground_mask_small, cv2.MORPH_CLOSE, close_kernel
    )
    foreground_mask_small = cv2.morphologyEx(
        foreground_mask_small, cv2.MORPH_OPEN, open_kernel
    )

    foreground_mask: np.ndarray = cv2.resize(
        foreground_mask_small, (width, height), interpolation=cv2.INTER_LINEAR
    )
    foreground_mask = cv2.GaussianBlur(foreground_mask, (9, 9), 0)
    _, foreground_mask = cv2.threshold(foreground_mask, 127, 255, cv2.THRESH_BINARY)

    contours: list[np.ndarray]
    contours, _ = cv2.findContours(
        foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return image

    area_threshold: float = 0.02 * (width * height)
    significant_points: list[np.ndarray] = [
        c for c in contours if cv2.contourArea(c) > area_threshold
    ]
    if not significant_points:
        significant_points = [max(contours, key=cv2.contourArea)]
    all_points: np.ndarray = np.vstack(significant_points)
    hull: np.ndarray = cv2.convexHull(all_points)

    min_area_rect: tuple = cv2.minAreaRect(hull)

    center_x: Any
    center_y: Any
    (center_x, center_y), (_, _), angle = min_area_rect

    # normalize to the minimal correction angle
    angle = angle % 90.0
    if angle > 45.0:
        angle -= 90.0

    rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(
        (center_x, center_y), angle, 1.0
    )
    rotated_image: np.ndarray = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    rotated_mask: np.ndarray = cv2.warpAffine(
        foreground_mask,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
    )

    rotated_contours: list[np.ndarray]
    rotated_contours, _ = cv2.findContours(
        rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not rotated_contours:
        return image

    rotated_significant: list[np.ndarray] = [
        c for c in rotated_contours if cv2.contourArea(c) > area_threshold
    ]
    if not rotated_significant:
        rotated_significant = [max(rotated_contours, key=cv2.contourArea)]
    rotated_hull_points: np.ndarray = np.vstack(rotated_significant)

    x: int
    y: int
    w: int
    h: int
    x, y, w, h = cv2.boundingRect(rotated_hull_points)

    # small padding margin, since the mask blur/threshold can shave a
    # few pixels off the true board edge
    x = max(0, x - crop_padding_px)
    y = max(0, y - crop_padding_px)
    w = min(width - x, w + 2 * crop_padding_px)
    h = min(height - y, h + 2 * crop_padding_px)

    cropped_image: np.ndarray = rotated_image[y : y + h, x : x + w]
    cropped_mask: np.ndarray = rotated_mask[y : y + h, x : x + w]

    result: np.ndarray = cropped_image.copy()
    result[cropped_mask == 0] = (255, 255, 255)

    return result


def process_real_images(
    source_directory: str,
    destination_directory: str,
    border_fraction: float = 0.04,
    center_fraction: float = 0.5,
    working_max_dimension: int = 900,
    grabcut_iterations: int = 8,
    crop_padding_px: int = 6,
) -> None:
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

                image = remove_background_rotate_axis_aligned(
                    image,
                    border_fraction=border_fraction,
                    center_fraction=center_fraction,
                    working_max_dimension=working_max_dimension,
                    grabcut_iterations=grabcut_iterations,
                    crop_padding_px=crop_padding_px,
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

                counter += 1


if __name__ == "__main__":
    process_real_images("data/PCB-DSLR", "data/real_images")
