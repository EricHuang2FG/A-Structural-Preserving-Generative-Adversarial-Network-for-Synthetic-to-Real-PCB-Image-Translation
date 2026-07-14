import re

from typing import TextIO
from datasets import load_dataset, IterableDataset

from src.utils.constants import RAW_DATA_DIR


def convert_to_safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def main() -> None:
    dataset: IterableDataset = load_dataset(
        "bshada/open-schematics",
        split="train",
        streaming=True,
    )
    dataset = dataset.select_columns(["name", "extensions_used", "pcb_files"])

    data_count: int = 1
    i: int
    row: dict
    for i, row in enumerate(dataset):
        if data_count <= 19999: # skip the first 85544 PCBs which are fetched already
            continue

        if ".kicad_pcb" not in row["extensions_used"]:
            continue

        name: str = convert_to_safe_name(row["name"])

        j: int
        pcb_text: str | None
        for j, pcb_text in enumerate(row["pcb_files"]):
            if pcb_text is None:
                continue
            if not pcb_text.lstrip().startswith("(kicad_pcb"):  # not .kicad_pcb
                continue

            file_path: str = f"{RAW_DATA_DIR}/{data_count}_{name}_{j}.kicad_pcb"

            f: TextIO
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(pcb_text)

            data_count += 1
            print(f"Rows fetched: {data_count}")


if __name__ == "__main__":
    main()
