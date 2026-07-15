import os
import shutil


def process_cyclegan_dataset(
    synthetic_train_dir: str,
    synthetic_test_dir: str,
    real_dir: str,
    output_dir: str,
    image_name: str = "top_image.png",
) -> None:
    valid_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    train_a_dir: str = os.path.join(output_dir, "trainA")
    train_b_dir: str = os.path.join(output_dir, "trainB")
    test_a_dir: str = os.path.join(output_dir, "testA")

    os.makedirs(train_a_dir, exist_ok=True)
    os.makedirs(train_b_dir, exist_ok=True)
    os.makedirs(test_a_dir, exist_ok=True)

    # take synthetic_split/train images into trainA
    counter: int = 1
    pcb_folder: str
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

    # take synthetic_split/test images into testA
    counter = 1
    for pcb_folder in os.listdir(synthetic_test_dir):
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

    # take real images into trainB
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


if __name__ == "__main__":
    process_cyclegan_dataset(
        synthetic_train_dir="data/synthetic_split/train",
        synthetic_test_dir="data/synthetic_split/test",
        real_dir="data/real_images",
        output_dir="data/cyclegan_data",
    )
