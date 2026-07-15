RAW_DATA_DIR: str = "data/open-schematics"

BOARD_RENDER_HEIGHT: int = 1020
BOARD_RENDER_WIDTH: int = 1020
TARGET_IMAGE_SIZE: int = 256  # square
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
EARLY_STOPPING_PATIENCE: int = 6

MODEL_NAME_TEMPLATE: str = (
    "{{ model_name }}_bs{{ batch_size }}_lr{{ learning_rate }}_{{ epoch }}.model"
)
TRAINING_CURVE_FILE_NAME_TEMPLATE: str = (
    "{{ model_name }}_bs{{ batch_size }}_lr{{ learning_rate }}_{{ type }}.pdf"
)
SEGMENTOR_MODEL_CHECKPOINTS_DIRECTORY: str = "models/segmentor/checkpoints"
SEGMENTOR_MODEL_BEST_MODEL_DIRECTORY: str = "models/segmentor/best"
SEGMENTOR_MODEL_TRAINING_CURVE_DIRECTORY: str = "models/segmentor/training_curves"
