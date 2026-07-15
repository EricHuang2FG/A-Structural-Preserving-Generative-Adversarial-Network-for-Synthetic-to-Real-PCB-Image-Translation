import os
import time
import json
import shutil
import random
import subprocess

import wx
import cv2
import pcbnew
import numpy as np

from typing import Any, Optional, Callable, TextIO

from src.utils.constants import (
    BOARD_RENDER_HEIGHT,
    BOARD_RENDER_WIDTH,
    RENDER_QUALITY,
    BOARD_SIDE_BOTTOM,
    BOARD_SIDE_TOP,
    REFERENCE_TO_CLASS_MAPPING,
    CLASS_TO_SEMANTIC_INDEX_MAPPING,
    DEFAULT_CLASS,
    SEED,
)


def iu_to_mm(val: int) -> float:
    return pcbnew.ToMM(val)


def polygon_to_coordinates(polygon_set: Any) -> Optional[list[tuple[float, float]]]:
    if polygon_set.OutlineCount() == 0:
        return None

    outline: Any = polygon_set.Outline(0)
    points: list[tuple[float, float]] = []

    i: int
    for i in range(outline.PointCount()):
        point: Any = outline.CPoint(i)
        points.append((iu_to_mm(point.x), iu_to_mm(point.y)))

    return points if len(points) >= 3 else None


def get_class_name_from_reference(reference: str) -> str:
    # first get the letter prefix of the reference
    # e.g. R13 becomes R
    reference_prefix: str = ""

    character: str
    for character in reference:
        if character.isalpha():
            reference_prefix += character
        else:
            break

    # return the class name from the letter prefix
    return REFERENCE_TO_CLASS_MAPPING.get(reference_prefix.upper(), DEFAULT_CLASS)


def mirror_x(x: float, edge_bbox: tuple[float, float, float, float]) -> float:
    x_min: float
    x_max: float
    x_min, _, x_max, _ = edge_bbox
    return x_max - (x - x_min)


def get_adaptive_alpha_threshold(alpha: np.ndarray) -> int:
    non_zero: np.ndarray = alpha[alpha > 0]
    if len(non_zero) == 0:
        raise ValueError("Alpha channel is fully transparent. Nothing to threshold.")

    values_as_image: np.ndarray = non_zero.reshape(1, -1).astype(np.uint8)

    otsu_threshold: Any
    otsu_threshold, _ = cv2.threshold(
        values_as_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return int(otsu_threshold)


def get_pcb_bbox_from_image(image_path: str) -> tuple[int, int, int, int]:
    # bounding box of the pcb, which are the non-transparent pixels
    image: Optional[np.ndarray] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None or image.shape[2] != 4:
        raise ValueError("The provided image is not an RGBA image.")

    alpha: np.ndarray = image[:, :, 3]
    threshold: int = get_adaptive_alpha_threshold(alpha)

    ys: np.ndarray
    xs: np.ndarray
    ys, xs = np.where(alpha > threshold)

    if len(xs) == 0:
        raise ValueError("The provided image is fully transparent.")

    x_min: int = int(xs.min())
    y_min: int = int(ys.min())
    x_max: int = int(xs.max())
    y_max: int = int(ys.max())

    return x_min, y_min, x_max, y_max


def get_layer_bbox(
    footprint: pcbnew.FOOTPRINT, target_layer: int
) -> Optional[list[tuple[float, float]]]:
    x_coords: list[float] = []
    y_coords: list[float] = []

    item: Any
    for item in footprint.GraphicalItems():
        if item.GetLayer() != target_layer:
            continue

        item_bbox: pcbnew.BOX2I = item.GetBoundingBox()
        x_coords.append(iu_to_mm(item_bbox.GetLeft()))
        x_coords.append(iu_to_mm(item_bbox.GetRight()))
        y_coords.append(iu_to_mm(item_bbox.GetTop()))
        y_coords.append(iu_to_mm(item_bbox.GetBottom()))

    if not x_coords:
        return None  # no items at all on this layer for this footprint

    x_min: float = min(x_coords)
    x_max: float = max(x_coords)
    y_min: float = min(y_coords)
    y_max: float = max(y_coords)

    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]


def get_pad_hull_bbox(
    footprint: pcbnew.FOOTPRINT,
) -> Optional[list[tuple[float, float]]]:
    points: list[tuple[float, float]] = []

    pad: pcbnew.PAD
    for pad in footprint.Pads():
        poly: Any = pad.GetEffectivePolygon()
        if poly is None or poly.OutlineCount() == 0:
            continue
        outline = poly.Outline(0)
        for i in range(outline.PointCount()):
            point: Any = outline.CPoint(i)
            points.append((iu_to_mm(point.x), iu_to_mm(point.y)))

    if len(points) < 3:
        return None

    points: np.ndarray = np.array(points, dtype=np.float32)
    rectangle: Any = cv2.minAreaRect(points)
    box: Any = cv2.boxPoints(rectangle)
    return [(float(p[0][0]), float(p[0][1])) for p in box]


def get_image_dimensions(image_path: str) -> tuple[int, int]:
    image: Optional[np.ndarray] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"{image_path} does not exist or could not be read.")

    height: int
    width: int
    height, width = image.shape[:2]
    return width, height


def render_pcb(pcb_file_path: str, output_file_path: str, side: str) -> None:
    if not os.path.exists(pcb_file_path):
        raise FileNotFoundError(f"{pcb_file_path} does not exist.")

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    command: list[str] = [
        "kicad-cli",
        "pcb",
        "render",
        "--output",
        output_file_path,
        "--side",
        side,
        "--width",
        str(BOARD_RENDER_WIDTH),
        "--height",
        str(BOARD_RENDER_HEIGHT),
        "--background",
        "transparent",
        "--quality",
        RENDER_QUALITY,
        pcb_file_path,
    ]
    subprocess.run(command, check=True, capture_output=True)


def get_pcb_geometry(pcb_file_path: str) -> dict:
    board: pcbnew.BOARD = pcbnew.LoadBoard(pcb_file_path)

    edge_bbox: pcbnew.BOX2I = board.GetBoardEdgesBoundingBox()
    edge_bbox: tuple[float, float, float, float] = (
        iu_to_mm(edge_bbox.GetLeft()),
        iu_to_mm(edge_bbox.GetTop()),
        iu_to_mm(edge_bbox.GetRight()),
        iu_to_mm(edge_bbox.GetBottom()),
    )

    footprints: list[dict] = []

    footprint: pcbnew.FOOTPRINT
    for footprint in board.GetFootprints():
        reference: str = footprint.GetReference()
        is_bottom: bool = footprint.IsFlipped()

        side: str = BOARD_SIDE_BOTTOM if is_bottom else BOARD_SIDE_TOP
        target_layer: int = pcbnew.B_CrtYd if is_bottom else pcbnew.F_CrtYd
        # top and bottom layers. Excluding the copper connections for now

        polygon: Optional[list[tuple[float, float]]] = None
        item: Any
        for item in footprint.GraphicalItems():
            if item.GetLayer() != target_layer:
                continue

            polygon_set: Optional[Any] = (
                item.GetPolyShape() if hasattr(item, "GetPolyShape") else None
            )
            if polygon_set is not None:
                raw_polygon: Optional[list[tuple[float, float]]] = (
                    polygon_to_coordinates(polygon_set)
                )
                if raw_polygon:
                    xs: list
                    ys: list
                    xs = [p[0] for p in raw_polygon]
                    ys = [p[1] for p in raw_polygon]

                    x_min: float
                    x_max: float
                    y_min: float
                    y_max: float

                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    polygon = [
                        (x_min, y_min),
                        (x_max, y_min),
                        (x_max, y_max),
                        (x_min, y_max),
                    ]
                    print("Using courtyard")
                    break

        if not polygon:
            # if polygon does not exist, then create an approximation mask
            approx_bbox: pcbnew.BOX2I = footprint.GetBoundingBox(False, False)

            x_min: float
            y_min: float
            x_max: float
            y_max: float
            x_min, y_min = iu_to_mm(approx_bbox.GetLeft()), iu_to_mm(
                approx_bbox.GetTop()
            )
            x_max, y_max = iu_to_mm(approx_bbox.GetRight()), iu_to_mm(
                approx_bbox.GetBottom()
            )
            polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

        footprints.append(
            {
                "reference_designator": reference,
                "class_name": get_class_name_from_reference(reference),
                "side": side,
                "points": polygon,
            }
        )

    return {"edge_bbox": edge_bbox, "footprints": footprints}


def get_pcb_to_image_coordinate_transformation(
    image_path: str, edge_bbox: tuple[float, float, float, float]
) -> Callable:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    x_min, y_min, x_max, y_max = edge_bbox

    board_width: float = x_max - x_min
    board_height: float = y_max - y_min

    x_min_pixels: int
    y_min_pixels: int
    x_max_pixels: int
    y_max_pixels: int

    x_min_pixels, y_min_pixels, x_max_pixels, y_max_pixels = get_pcb_bbox_from_image(
        image_path
    )

    board_width_pixels: int = x_max_pixels - x_min_pixels
    board_height_pixels: int = y_max_pixels - y_min_pixels

    scale_x: float = board_width_pixels / board_width
    scale_y: float = board_height_pixels / board_height

    def transform(x: float, y: float) -> tuple[float, float]:
        x_pixels: float = (x - x_min) * scale_x + x_min_pixels
        y_pixels: float = (y - y_min) * scale_y + y_min_pixels

        return x_pixels, y_pixels

    return transform


def get_annotation_instance_mask(
    footprints: list[dict],
    side: str,
    pcb_to_image_coordinate_transformation: Callable,
    edge_bbox: tuple[float, float, float, float],
    image_size: tuple[float, float] = (BOARD_RENDER_WIDTH, BOARD_RENDER_HEIGHT),
) -> tuple[np.ndarray, list[dict]]:
    # builds the instance mask and the annotations
    image_width: float
    image_height: float
    image_width, image_height = image_size

    instance_mask: np.ndarray = np.zeros((image_height, image_width), dtype=np.uint16)
    annotations: list[dict] = []

    instance_id: int = 0
    footprint: dict
    for footprint in footprints:
        if footprint["side"] != side:
            continue

        instance_id += 1

        points_on_image: list[tuple[float, float]] = []
        x: float
        y: float
        for x, y in footprint["points"]:
            if side == BOARD_SIDE_BOTTOM:
                x = mirror_x(x, edge_bbox)

            u: float
            v: float
            u, v = pcb_to_image_coordinate_transformation(x, y)
            points_on_image.append((u, v))

        points_on_image: np.ndarray = np.array(points_on_image, dtype=np.int32)
        cv2.fillPoly(instance_mask, [points_on_image], color=instance_id)

        w: float
        h: float
        x, y, w, h = cv2.boundingRect(points_on_image)
        annotations.append(
            {
                "id": instance_id,
                "reference_designator": footprint["reference_designator"],
                "class_name": footprint["class_name"],
                "bbox": [int(x), int(y), int(w), int(h)],
                "segmentation": [
                    [float(num) for point in list(points_on_image) for num in point]
                ],
            }
        )

    return instance_mask, annotations


def create_segmentation_mask_visualization(mask_path: str, image_path: str) -> None:
    mask: Optional[np.ndarray] = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"{mask_path} does not exist or could not be read.")

    render: Optional[np.ndarray] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if render is None:
        raise FileNotFoundError(f"{image_path} does not exist or could not be read.")

    if render.shape[2] == 4:
        render_bgr: np.ndarray = render[:, :, :3]
    else:
        render_bgr = render

    if mask.shape[:2] != render_bgr.shape[:2]:
        raise ValueError(
            f"Size mismatch between mask and render: mask is {mask.shape[:2]}, "
            f"render is {render_bgr.shape[:2]}."
        )

    max_instance_id: int = int(mask.max())

    rng: np.random.Generator = np.random.default_rng(seed=0)
    colours: np.ndarray = rng.integers(
        0, 255, size=(max_instance_id + 1, 3), dtype=np.uint8
    )
    colours[0] = 0  # background (instance id 0) stays black

    colorized_mask: np.ndarray = colours[mask]  # (H, W, 3)

    overlay: np.ndarray = cv2.addWeighted(render_bgr, 0.7, colorized_mask, 0.6, 0)

    output_directory: str = os.path.dirname(image_path)
    mask_filename: str = os.path.splitext(os.path.basename(mask_path))[0]
    overlay_path: str = os.path.join(output_directory, f"{mask_filename}_overlay.png")

    cv2.imwrite(overlay_path, overlay)
    print(f"Visualization overlay saved to {overlay_path}")


def create_semantic_mask(
    instance_mask_path: str, annotations_path: str, output_path: str
) -> None:
    instance_mask: np.ndarray = cv2.imread(instance_mask_path, cv2.IMREAD_UNCHANGED)
    f: TextIO
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations: list[dict] = json.load(f)

    semantic_mask: np.ndarray = np.zeros_like(instance_mask, dtype=np.uint8)
    annotation: dict
    for annotation in annotations:
        class_index: int = CLASS_TO_SEMANTIC_INDEX_MAPPING.get(
            annotation["class_name"], 0
        )
        semantic_mask[instance_mask == annotation["id"]] = class_index

    cv2.imwrite(output_path, semantic_mask)


def process_pcb(pcb_file_path: str, output_directory: str) -> None:
    if not os.path.exists(pcb_file_path):
        raise FileNotFoundError(f"{pcb_file_path} does not exist.")
    os.makedirs(os.path.dirname(output_directory), exist_ok=True)

    geometry: dict = get_pcb_geometry(pcb_file_path)

    side: str
    for side in (BOARD_SIDE_TOP, BOARD_SIDE_BOTTOM):
        image_path: str = f"{output_directory}/{side}_image.png"
        render_pcb(pcb_file_path, image_path, side)
        print(f"Image for {side} rendered")

        edge_bbox: tuple[float, float, float, float] = geometry["edge_bbox"]
        pcb_to_image_coordinate_transformation: Callable = (
            get_pcb_to_image_coordinate_transformation(image_path, edge_bbox)
        )
        segmentation_mask: np.ndarray
        annotations: list[dict]
        segmentation_mask, annotations = get_annotation_instance_mask(
            geometry["footprints"],
            side,
            pcb_to_image_coordinate_transformation,
            edge_bbox,
            image_size=get_image_dimensions(image_path),
        )
        segmentation_mask_path: str = f"{output_directory}/{side}_mask.png"
        cv2.imwrite(segmentation_mask_path, segmentation_mask)

        annotations_path: str = f"{output_directory}/{side}_annotations.json"
        f: TextIO
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)

        create_segmentation_mask_visualization(segmentation_mask_path, image_path)

        semantic_mask_path: str = f"{output_directory}/{side}_semantic_mask.png"
        create_semantic_mask(
            segmentation_mask_path, annotations_path, semantic_mask_path
        )

        print(f"Segmentation mask and annotations **{side}** saved for {pcb_file_path}")


def process_multiple_pcbs(
    pcb_file_directory: str, output_directory: str, start_num: int, end_num: int
) -> None:
    # start_num and end_num are inclusive, and are NOT zero-indexed
    if not os.path.isdir(pcb_file_directory):
        raise NotADirectoryError(f"{pcb_file_directory} is not a directory.")

    os.makedirs(output_directory, exist_ok=True)
    pcb_files_sorted: list = sorted(
        (
            filename
            for filename in os.listdir(pcb_file_directory)
            if filename.endswith(".kicad_pcb")
        ),  # filter for .kicad_pcb extension
        key=lambda f: int(f.split("_")[0]),
    )

    falied_processes: list[tuple[str, str]] = []
    for curr_pcb_filename in pcb_files_sorted:
        curr_pcb_file_count: int = int(
            curr_pcb_filename.split("_")[0]
        )  # always starts the filename
        if curr_pcb_file_count < start_num:
            continue

        if curr_pcb_file_count > end_num:
            break

        start_time: float = time.perf_counter()

        pcb_file_path: str = os.path.join(pcb_file_directory, curr_pcb_filename)

        curr_pcb_output_directory: str
        curr_pcb_output_directory = os.path.splitext(curr_pcb_filename)[0]
        curr_pcb_output_directory = os.path.join(
            output_directory, curr_pcb_output_directory
        )

        try:
            process_pcb(pcb_file_path, curr_pcb_output_directory)
        except Exception as e:
            print(f"PCB {curr_pcb_file_count} failed to process due to error {e}")
            falied_processes.append((pcb_file_path, str(e)))

            if os.path.isdir(curr_pcb_output_directory):
                shutil.rmtree(curr_pcb_output_directory)

        end_time: float = time.perf_counter()
        print(
            f"PCB {curr_pcb_file_count}/{end_num} processed in {(end_time - start_time):.4f} seconds"
        )

    if falied_processes:
        file_path: str
        error_message: str
        for file_path, error_message in falied_processes:
            print(f"{file_path}: {error_message}")


def split_dataset(
    source_directory: str,
    train_directory: str,
    test_directory: str,
    test_ratio: float = 0.1,
    seed: int = SEED,
) -> None:
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    pcb_folders: list[str] = [
        folder
        for folder in os.listdir(source_directory)
        if os.path.isdir(os.path.join(source_directory, folder))
    ]

    # seed random
    random.seed(seed)
    random.shuffle(pcb_folders)

    split_index: int = int(len(pcb_folders) * (1 - test_ratio))

    train_folders: list[str] = pcb_folders[:split_index]
    test_folders: list[str] = pcb_folders[split_index:]

    print(f"Total number of PCBs: {len(pcb_folders)}")
    print(f"Train PCBs: {len(train_folders)}")
    print(f"Test PCBs: {len(test_folders)}")

    for folder in train_folders:
        shutil.copytree(
            os.path.join(source_directory, folder),
            os.path.join(train_directory, folder),
        )

    for folder in test_folders:
        shutil.copytree(
            os.path.join(source_directory, folder),
            os.path.join(test_directory, folder),
        )


if __name__ == "__main__":
    wx.Log.SetLogLevel(wx.LOG_Error)
    app: wx.App = wx.App(False)
    # process_multiple_pcbs("data/open-schematics", "data/synthetic", 1701, 1800)
    split_dataset(
        "data/synthetic", "data/synthetic_split/train", "data/synthetic_split/test"
    )
