import os
import re
import cv2

import numpy as np

from typing import Any

from src.utils.constants import BOARD_RENDER_WIDTH, BOARD_RENDER_HEIGHT


def remove_background_rotate_axis_aligned(
    image: np.ndarray,
    threshold: int = 25,
    crop_padding_px: int = 10,
) -> np.ndarray:
    height: int
    width: int
    height, width = image.shape[:2]

    # segment foreground from black background
    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    foreground_mask: np.ndarray
    _, foreground_mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    # cleanup
    kernel: np.ndarray = np.ones((5, 5), np.uint8)

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # keep largest connected component (the PCB)
    num_labels: int
    labels: np.ndarray
    stats: np.ndarray

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground_mask, connectivity=8
    )

    if num_labels <= 1:
        return image

    largest: int = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    foreground_mask = np.zeros_like(
        foreground_mask,
    )

    foreground_mask[labels == largest] = 255

    contours: list[np.ndarray]

    contours, _ = cv2.findContours(
        foreground_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return image

    cv2.drawContours(
        foreground_mask,
        contours,
        -1,
        255,
        thickness=cv2.FILLED,
    )

    # compute rotation
    largest_contour: np.ndarray = max(contours, key=cv2.contourArea)

    hull: np.ndarray = cv2.convexHull(largest_contour)

    rect: tuple[
        tuple[float, float],
        tuple[float, float],
        float,
    ] = cv2.minAreaRect(hull)

    angle: float = rect[2]

    angle = angle % 90.0

    if angle > 45:
        angle -= 90

    # rotate the image and the mask separately
    rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        angle,
        1.0,
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

    # crop
    contours, _ = cv2.findContours(
        rotated_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return image

    largest: np.ndarray = max(
        contours,
        key=cv2.contourArea,
    )

    x: int
    y: int
    w: int
    h: int
    x, y, w, h = cv2.boundingRect(
        largest,
    )
    x = max(0, x - crop_padding_px)
    y = max(0, y - crop_padding_px)
    w = min(width - x, w + 2 * crop_padding_px)
    h = min(height - y, h + 2 * crop_padding_px)

    cropped_image: np.ndarray = rotated_image[y : y + h, x : x + w]

    cropped_mask: np.ndarray = rotated_mask[y : y + h, x : x + w]

    # attempt to remove dark edges
    gray_crop: np.ndarray = cv2.cvtColor(
        cropped_image,
        cv2.COLOR_BGR2GRAY,
    )

    distance: np.ndarray = cv2.distanceTransform(
        cropped_mask,
        cv2.DIST_L2,
        3,
    )

    # pixels close to the PCB boundary
    edge_region: np.ndarray = distance < 15

    # dark pixels near boundary are likely black background bleed
    dark_edge: np.ndarray = (gray_crop < 45) & edge_region

    final_mask: np.ndarray = cropped_mask.copy()

    final_mask[dark_edge] = 0

    # cast the image onto white background
    result: np.ndarray = np.full_like(cropped_image, 255)

    result[final_mask > 0] = cropped_image[final_mask > 0]

    return result


def process_real_images(
    source_directory: str,
    destination_directory: str,
    threshold: int = 25,
    crop_padding_px: int = 10,
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
                    threshold=threshold,
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

                print(f"Real PCB image {counter} finished processing")

                counter += 1


if __name__ == "__main__":
    process_real_images("data/PCB-DSLR", "data/real_images")
