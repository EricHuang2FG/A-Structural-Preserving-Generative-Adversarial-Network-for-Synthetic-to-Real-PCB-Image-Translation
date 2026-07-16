import os


def get_synthetic_data_paths_with_semantic_mask(
    root_directory: str,
) -> list[tuple[str, str]]:
    data_paths: list[tuple[str, str]] = []
    skipped_data: list[str] = []

    pcb_folder: str
    for pcb_folder in sorted(os.listdir(root_directory)):
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
