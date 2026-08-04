import os
import re
import cv2

import numpy as np

from src.utils.constants import BOARD_RENDER_WIDTH, BOARD_RENDER_HEIGHT


def remove_background_rotate_axis_aligned(
    image: np.ndarray,
    corner_fraction: float = 0.05,
    center_fraction: float = 0.5,
    working_max_dimension: int = 800,
    grabcut_iterations: int = 8,
) -> np.ndarray:
    height: int
    width: int
    height, width = image.shape[:2]

    # downscale the image first so the processing runs faster
    scale: float = min(1.0, working_max_dimension / max(height, width))
    small_image: np.ndarray = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )
    small_height: int
    small_width: int
    small_height, small_width = small_image.shape[:2]

    # default everything to probable foreground so nothing is hard-locked to be removed
    mask: np.ndarray = np.full(
        (small_height, small_width), cv2.GC_PR_FGD, dtype=np.uint8
    )

    # sure-background seeds which are small patches in the four corners
    corner_h: int = max(1, int(small_height * corner_fraction))
    corner_w: int = max(1, int(small_width * corner_fraction))
    mask[0:corner_h, 0:corner_w] = cv2.GC_BGD
    mask[0:corner_h, small_width - corner_w :] = cv2.GC_BGD
    mask[small_height - corner_h :, 0:corner_w] = cv2.GC_BGD
    mask[small_height - corner_h :, small_width - corner_w :] = cv2.GC_BGD

    # sure-foreground seed, which is a solid block in the center (the PCB)
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

    kernel: np.ndarray = np.ones((7, 7), np.uint8)
    foreground_mask_small = cv2.morphologyEx(
        foreground_mask_small, cv2.MORPH_OPEN, kernel
    )
    foreground_mask_small = cv2.morphologyEx(
        foreground_mask_small, cv2.MORPH_CLOSE, kernel
    )

    # upscale the mask back to full resolution
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

    largest_contour: np.ndarray = max(contours, key=cv2.contourArea)

    min_area_rect: tuple = cv2.minAreaRect(largest_contour)
    (center_x, center_y), (rect_w, rect_h), angle = min_area_rect

    if rect_w < rect_h:
        angle += 90.0

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

    x: int
    y: int
    w: int
    h: int
    x, y, w, h = cv2.boundingRect(max(rotated_contours, key=cv2.contourArea))

    cropped_image: np.ndarray = rotated_image[y : y + h, x : x + w]
    cropped_mask: np.ndarray = rotated_mask[y : y + h, x : x + w]

    result: np.ndarray = cropped_image.copy()
    result[cropped_mask == 0] = (255, 255, 255)

    return result


def process_real_images(
    source_directory: str,
    destination_directory: str,
    corner_fraction: float = 0.05,
    center_fraction: float = 0.5,
    working_max_dimension: int = 510,
    grabcut_iterations: int = 8,
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
                    corner_fraction=corner_fraction,
                    center_fraction=center_fraction,
                    working_max_dimension=working_max_dimension,
                    grabcut_iterations=grabcut_iterations,
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
