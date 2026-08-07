import os
import shutil
import argparse

from src.utils.args import parse_args_external_dataset_flag


def process_cyclegan_dataset(
    synthetic_train_dir: str | None,
    synthetic_test_dir: str,
    real_dir: str | None,
    output_dir: str,
    image_name: str = "top_image.png",
    process_test_data_only: bool = False,
) -> None:
    if not process_test_data_only and (synthetic_train_dir is None or real_dir is None):
        raise ValueError(
            "synthetic_train_dir and real_dir cannot be none since process_test_data_only=False"
        )

    valid_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    train_a_dir: str | None = None
    train_b_dir: str | None = None
    if not process_test_data_only:
        train_a_dir = os.path.join(output_dir, "trainA")
        train_b_dir = os.path.join(output_dir, "trainB")
    test_a_dir: str = os.path.join(output_dir, "testA")

    directory: str
    for directory in (
        (train_a_dir, train_b_dir, test_a_dir)
        if not process_test_data_only
        else (test_a_dir,)
    ):
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    # take synthetic_split/train images into trainA
    counter: int = 1
    pcb_folder: str
    if not process_test_data_only:
        for pcb_folder in os.listdir(synthetic_train_dir):
            folder_path: str = os.path.join(synthetic_train_dir, pcb_folder)

            if not os.path.isdir(folder_path):
                continue

            image_path: str = os.path.join(folder_path, image_name)

            if os.path.exists(image_path):
                shutil.copy2(
                    image_path,
                    os.path.join(train_a_dir, f"{counter}.png"),
                )
                counter += 1
        print(f"{counter - 1} trainA images")

    # take synthetic_split/test images into testA
    counter = 1
    for pcb_folder in sorted(
        [
            folder
            for folder in os.listdir(synthetic_test_dir)
            if os.path.isdir(os.path.join(synthetic_test_dir, folder))
        ],
        key=lambda s: int(s.split("_")[0]),
    ):
        folder_path = os.path.join(synthetic_test_dir, pcb_folder)

        if not os.path.isdir(folder_path):
            continue

        image_path = os.path.join(folder_path, image_name)

        if os.path.exists(image_path):
            shutil.copy2(
                image_path,
                os.path.join(test_a_dir, f"{counter}.png"),
            )
            counter += 1
    print(f"{counter - 1} testA images")

    # take real images into trainB
    if not process_test_data_only:
        counter = 1
        for filename in os.listdir(real_dir):
            src: str = os.path.join(real_dir, filename)

            if os.path.isfile(src):
                _, ext = os.path.splitext(filename)

                if ext.lower() in valid_extensions:
                    shutil.copy2(
                        src,
                        os.path.join(train_b_dir, f"{counter}{ext}"),
                    )
                    counter += 1
        print(f"{counter - 1} trainB images")


if __name__ == "__main__":
    args: argparse.Namespace = parse_args_external_dataset_flag(
        "Process CycleGAN dataset structure for either the full open-schematics training data or the external test dataset"
    )
    if not args.external:
        process_cyclegan_dataset(
            synthetic_train_dir="data/synthetic_split/train",
            synthetic_test_dir="data/synthetic_split/test",
            real_dir="data/real_images_split/train",
            output_dir="data/cyclegan_data",
        )
    else:
        process_cyclegan_dataset(
            synthetic_train_dir=None,
            synthetic_test_dir="data/external_test_datasets/synthetic",
            real_dir=None,
            output_dir="data/external_test_datasets/cyclegan",
            process_test_data_only=args.external,
        )
