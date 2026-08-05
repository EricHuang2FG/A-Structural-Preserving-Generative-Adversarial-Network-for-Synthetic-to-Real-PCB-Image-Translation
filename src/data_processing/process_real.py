import os
import re
import random
import shutil

import cv2

import numpy as np

from src.utils.constants import BOARD_RENDER_WIDTH, BOARD_RENDER_HEIGHT, SEED


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


def process_real_images_pcb_dslr(
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

            print(f"Real PCB image from PCB-DSLR: {counter} finished processing")

            counter += 1


def process_real_images_pcb_data(
    source_directory: str,
    destination_directory: str,
) -> None:
    os.makedirs(destination_directory, exist_ok=True)

    counter: int = (
        len(
            [
                file
                for file in os.listdir(destination_directory)
                if file.lower().endswith(".jpg")
            ]
        )
        + 1
    )

    skipped: list[str] = []

    filename: str
    for filename in sorted(os.listdir(source_directory)):
        if not filename.lower().endswith(".jpg"):
            continue

        image_path: str = os.path.join(source_directory, filename)
        image: np.ndarray | None = cv2.imread(image_path)

        if image is None:
            skipped.append(filename)
            continue

        height: int
        width: int
        height, width = image.shape[:2]

        scale: float = min(
            (BOARD_RENDER_WIDTH - 16) / max(height, width),
            (BOARD_RENDER_HEIGHT - 16) / max(height, width),
        )
        new_width: int = int(round(width * scale))
        new_height: int = int(round(height * scale))

        resized: np.ndarray = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        canvas: np.ndarray = (
            np.ones(
                ((BOARD_RENDER_HEIGHT - 16), (BOARD_RENDER_WIDTH - 16), 3),
                dtype=np.uint8,
            )
            * 255
        )

        x_offset: int = (BOARD_RENDER_WIDTH - 16 - new_width) // 2
        y_offset: int = (BOARD_RENDER_HEIGHT - 16 - new_height) // 2

        canvas[
            y_offset : y_offset + new_height,
            x_offset : x_offset + new_width,
        ] = resized

        destination: str = os.path.join(destination_directory, f"{counter}_{filename}")
        cv2.imwrite(destination, canvas)

        print(f"Real PCB image from pcb_data: {counter} finished processing")
        counter += 1

    if skipped:
        print(f"{len(skipped)} images skipped: {skipped}")


def split_real_dataset(
    source_directory: str,
    train_directory: str,
    validation_directory: str,
    test_directory: str,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = SEED,
) -> None:
    if os.path.isdir(train_directory):
        shutil.rmtree(train_directory)
    if os.path.isdir(validation_directory):
        shutil.rmtree(validation_directory)
    if os.path.isdir(test_directory):
        shutil.rmtree(test_directory)

    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(validation_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    image_files: list[str] = [
        filename
        for filename in os.listdir(source_directory)
        if os.path.isfile(os.path.join(source_directory, filename))
    ]

    # seed random
    random.seed(seed)
    random.shuffle(image_files)

    num_files: int = len(image_files)
    test_split_index: int = int(num_files * (1 - test_ratio - validation_ratio))
    validation_split_index: int = int(num_files * (1 - test_ratio))

    train_files: list[str] = image_files[:test_split_index]
    validation_files: list[str] = image_files[test_split_index:validation_split_index]
    test_files: list[str] = image_files[validation_split_index:]

    print(f"Total number of real images: {num_files}")
    print(f"Train images: {len(train_files)}")
    print(f"Validation images: {len(validation_files)}")
    print(f"Test images: {len(test_files)}")

    for filename in train_files:
        shutil.copy2(
            os.path.join(source_directory, filename),
            os.path.join(train_directory, filename),
        )

    for filename in validation_files:
        shutil.copy2(
            os.path.join(source_directory, filename),
            os.path.join(validation_directory, filename),
        )

    for filename in test_files:
        shutil.copy2(
            os.path.join(source_directory, filename),
            os.path.join(test_directory, filename),
        )


if __name__ == "__main__":
    process_real_images_pcb_dslr("data/PCB-DSLR", "data/real_images")
    process_real_images_pcb_data("data/pcb_data/train", "data/real_images")
    split_real_dataset(
        "data/real_images",
        "data/real_images_split/train",
        "data/real_images_split/validation",
        "data/real_images_split/test",
    )
