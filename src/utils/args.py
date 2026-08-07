import argparse


def parse_args_external_dataset_flag(description: str) -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--external",
        action="store_true",
        help="Use the external test dataset instead of the standard open-schematic dataset.",
    )
    return parser.parse_args()
