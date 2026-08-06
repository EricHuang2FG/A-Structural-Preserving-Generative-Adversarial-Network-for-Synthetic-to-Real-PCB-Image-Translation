import os
import shutil
import subprocess


def clone_repository(repo_url: str, target_directory: str) -> None:
    print(f"Cloning {repo_url} into {target_directory}")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, target_directory],
        check=True,
    )


def get_kicad_pcb_files_from_sub_folders(search_directory: str) -> list[str]:
    pcb_paths: list[str] = []

    root: str
    files: list[str]
    for root, _, files in os.walk(search_directory):
        filename: str
        for filename in files:
            if filename.lower().endswith(".kicad_pcb"):
                pcb_paths.append(os.path.join(root, filename))

    return pcb_paths


def get_kicad_pcb_files_in_named_folder(
    repo_directory: str, folder_name: str = "kicad_pcb"
) -> list[str]:
    target_directory: str = os.path.join(repo_directory, folder_name)

    if not os.path.isdir(target_directory):
        print(f"Expected folder '{folder_name}' not found in {repo_directory}")
        return []

    pcb_paths: list[str] = []
    root: str
    files: list[str]
    for root, _, files in os.walk(target_directory):
        filename: str
        for filename in files:
            if filename.lower().endswith(".kicad_pcb"):
                pcb_paths.append(os.path.join(root, filename))

    return pcb_paths


def copy_pcb_files(
    pcb_paths: list[str], output_directory: str, start_counter: int
) -> int:
    counter: int = start_counter

    pcb_path: str
    for pcb_path in pcb_paths:
        if counter in [16, 18, 20, 24]:  # skip known invalid data
            counter += 1
            continue

        parent_folder_name: str = os.path.basename(os.path.dirname(pcb_path))
        original_filename: str = os.path.basename(pcb_path)

        destination_filename: str = (
            f"{counter}_{parent_folder_name}_{original_filename}"
        )
        destination_path: str = os.path.join(output_directory, destination_filename)

        shutil.copy2(pcb_path, destination_path)
        print(f"Copied: {pcb_path} -> {destination_path}")

        counter += 1

    return counter


def fetch_external_test_datasets(
    data_directory: str,
    output_directory: str,
    named_folder: str = "kicad_pcb",
) -> None:
    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)

    kicad_templates_directory: str = os.path.join(data_directory, "kicad_templates")
    dataset_srj18_directory: str = os.path.join(data_directory, "dataset-srj18")

    if os.path.isdir(kicad_templates_directory):
        shutil.rmtree(kicad_templates_directory)
    if os.path.isdir(dataset_srj18_directory):
        shutil.rmtree(dataset_srj18_directory)

    print("Cloning repositories")
    clone_repository(
        "https://github.com/sethhillbrand/kicad_templates.git",
        kicad_templates_directory,
    )
    clone_repository(
        "https://github.com/tscircuit/dataset-srj18.git", dataset_srj18_directory
    )

    print("Parsing from sethhillbrand/kicad_templates")
    kicad_templates_pcb_paths: list[str] = get_kicad_pcb_files_from_sub_folders(
        kicad_templates_directory
    )
    print(f"{len(kicad_templates_pcb_paths)} .kicad_pcb files found")

    print("Parsing from tscircuit/dataset-srj18")
    dataset_srj18_pcb_paths: list[str] = get_kicad_pcb_files_in_named_folder(
        dataset_srj18_directory, folder_name=named_folder
    )
    print(f"{len(dataset_srj18_pcb_paths)} .kicad_pcb files found")

    counter: int = 1
    counter = copy_pcb_files(
        kicad_templates_pcb_paths, output_directory, start_counter=counter
    )
    counter = copy_pcb_files(
        dataset_srj18_pcb_paths, output_directory, start_counter=counter
    )

    print(f"{counter - 1} .kicad_pcb files copied to {output_directory}")


if __name__ == "__main__":
    fetch_external_test_datasets(
        "data/", "data/external_test_datasets/kicad_pcb", named_folder="kicad_pcb"
    )
