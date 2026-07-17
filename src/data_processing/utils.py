import os


def get_real_image_paths(root_directory: str) -> list[str]:
    return [
        os.path.join(root_directory, file)
        for file in sorted(os.listdir(root_directory))
        if file.lower().endswith(".jpg")
    ]  # only valid extension is .jpg (not case sensitive)


def get_semantic_mask_translated_image_paths_pair(  # for SPresGAN
    mask_root_directory: str, translated_image_directory: str
) -> list[tuple[str, str]]:
    path_pairs: list[tuple[str, str]] = []
    skipped_data: list[str] = []

    # assume that if sorted, the files in the two directories are paired correctly
    # the masks live in subfolders while the translated images do not
    # translated images have names in the format "1.png", "2.png", ...
    # mask folders have names in the format "1_...", "2_..."
    translated_image_filenames: list = sorted(
        [
            filename
            for filename in os.listdir(translated_image_directory)
            if filename.lower().endswith((".jpg", ".png"))
        ],
        key=lambda s: int(s.split(".")[0]),
    )

    index: int
    pcb_folder: str
    for index, pcb_folder in enumerate(
        sorted(
            [
                folder
                for folder in os.listdir(mask_root_directory)
                if os.path.isdir(os.path.join(mask_root_directory, folder))
            ],
            key=lambda s: int(s.split("_")[0]),
        )
    ):
        pcb_directory: str = os.path.join(mask_root_directory, pcb_folder)

        if not os.path.isdir(pcb_directory):
            continue

        # sample top mask
        mask_path: str = os.path.join(pcb_directory, "top_semantic_mask.png")
        translated_image_path: str = os.path.join(
            translated_image_directory, translated_image_filenames[index]
        )

        if os.path.exists(mask_path) and os.path.exists(translated_image_path):
            path_pairs.append((mask_path, translated_image_path))
        else:
            skipped_data.append(pcb_directory)

    if skipped_data:
        print(f"{len(skipped_data)} images skipped: {skipped_data}")

    return path_pairs


def get_synthetic_data_paths_with_semantic_mask(
    root_directory: str,
) -> list[tuple[str, str]]:
    data_paths: list[tuple[str, str]] = []
    skipped_data: list[str] = []

    pcb_folder: str
    for pcb_folder in sorted(
        [
            folder
            for folder in os.listdir(root_directory)
            if os.path.isdir(os.path.join(root_directory, folder))
        ],
        key=lambda s: int(s.split("_")[0]),
    ):
        pcb_directory: str = os.path.join(root_directory, pcb_folder)

        if not os.path.isdir(pcb_directory):
            continue

        # sample top side only
        image_path: str = os.path.join(pcb_directory, "top_image.png")
        mask_path: str = os.path.join(pcb_directory, "top_semantic_mask.png")

        if os.path.exists(image_path) and os.path.exists(mask_path):
            data_paths.append((image_path, mask_path))
        else:
            skipped_data.append(pcb_directory)

    if skipped_data:
        print(f"{len(skipped_data)} images skipped: {skipped_data}")

    return data_paths


if __name__ == "__main__":
    get_semantic_mask_translated_image_paths_pair(
        "data/synthetic_split/test", "outputs/cyclegan/images"
    )
