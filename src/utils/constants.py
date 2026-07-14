RAW_DATA_DIR: str = "data/open-schematics"

BOARD_RENDER_HEIGHT: int = 1020
BOARD_RENDER_WIDTH: int = 1020
RENDER_QUALITY: str = "high"

BOARD_SIDE_TOP: str = "top"
BOARD_SIDE_BOTTOM: str = "bottom"

REFERENCE_TO_CLASS_MAPPING: dict[str, str] = {
    "R": "resistor",  # R1, R2, ...
    "C": "capacitor",
    "L": "inductor",
    "U": "ic",
    "Q": "transistor",
    "D": "diode",
    "J": "connector",
    "P": "connector",
    "SW": "switch",
    "Y": "crystal",
    "F": "fuse",
    "K": "relay",
}
DEFAULT_CLASS: str = "other"
