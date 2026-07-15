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

CLASS_TO_SEMANTIC_INDEX_MAPPING: dict[str, int] = {
    "background": 0,
    "resistor": 1,
    "capacitor": 2,
    "inductor": 3,
    "ic": 4,
    "transistor": 5,
    "diode": 6,
    "connector": 7,
    "switch": 8,
    "crystal": 9,
    "fuse": 10,
    "relay": 11,
    "other": 12,
}

GAN_IMAGE_SIZE: int = 512

SEED: int = 42
