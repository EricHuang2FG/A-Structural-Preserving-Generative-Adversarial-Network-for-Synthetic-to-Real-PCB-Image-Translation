import os
import time
import json
import shutil
import random
import tempfile
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
    DIFFERENTIAL_RENDERING_QUALITY,
    DIFFERENTIAL_THRESHOLD,
    DIFFERENTIAL_RENDERING_COLOUR_GROUP_MARGIN_PX,
    MIN_BLOB_AREA,
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


def render_pcb(
    pcb_file_path: str, output_file_path: str, side: str, quality: str = RENDER_QUALITY
) -> None:
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
        quality,
        pcb_file_path,
    ]
    subprocess.run(command, check=True, capture_output=True)


def render_pcb_differential_baseline(
    pcb_file_path: str, out_path: str, side: str
) -> None:
    board: pcbnew.BOARD = pcbnew.LoadBoard(pcb_file_path)
    hide_footprint_text(board)

    temp: Any = tempfile.NamedTemporaryFile(suffix=".kicad_pcb", delete=False)
    temp.close()

    temp_path: str = temp.name
    try:
        pcbnew.SaveBoard(temp_path, board)
        render_pcb(temp_path, out_path, side, quality=DIFFERENTIAL_RENDERING_QUALITY)
    finally:
        os.remove(temp_path)  # remove the temporary file that we no longer need


def hide_footprint_text(board: pcbnew.BOARD) -> None:
    footprint: pcbnew.FOOTPRINT
    for footprint in board.GetFootprints():
        footprint.Reference().SetVisible(False)
        footprint.Value().SetVisible(False)

        item: Any
        for item in footprint.GraphicalItems():
            if isinstance(item, pcbnew.PCB_TEXT):
                item.SetVisible(False)


def differential_rendering(
    pcb_file_path: str,
    refs_to_remove: list[str],
    side: str,
    out_path: str,
) -> None:
    board: pcbnew.BOARD = pcbnew.LoadBoard(pcb_file_path)  # reload pristine each pass
    hide_footprint_text(board)

    ref: str
    for ref in refs_to_remove:
        footprint: Optional[pcbnew.FOOTPRINT] = board.FindFootprintByReference(ref)
        if footprint is not None:
            board.RemoveNative(footprint)

    temp: Any = tempfile.NamedTemporaryFile(suffix=".kicad_pcb", delete=False)
    temp.close()

    temp_path: str = temp.name
    try:
        pcbnew.SaveBoard(temp_path, board)
        render_pcb(temp_path, out_path, side, quality=DIFFERENTIAL_RENDERING_QUALITY)
    finally:
        os.remove(temp_path)


def poly_set_to_polygons(poly_set: Any) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []

    outline_index: int
    for outline_index in range(poly_set.OutlineCount()):
        outline: Any = poly_set.Outline(outline_index)

        points: list[tuple[float, float]] = []
        i: int
        for i in range(outline.PointCount()):
            point: Any = outline.CPoint(i)
            points.append((iu_to_mm(point.x), iu_to_mm(point.y)))

        if len(points) >= 3:
            polygons.append(points)

    return polygons


def load_rgb_on_black(path: str) -> np.ndarray:
    image: Optional[np.ndarray] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"{path} does not exist or could not be read.")

    if image.ndim == 3 and image.shape[2] == 4:
        alpha: np.ndarray = image[:, :, 3].astype(np.float32) / 255.0
        rgb: np.ndarray = image[:, :, :3].astype(np.float32) * alpha[..., None]
        return rgb

    return image.astype(np.float32)


def get_differential_mask(full_rgb: np.ndarray, variant_rgb: np.ndarray) -> np.ndarray:
    diff: np.ndarray = np.abs(full_rgb - variant_rgb).max(axis=2)  # max over channels
    binary: np.ndarray = (diff > DIFFERENTIAL_THRESHOLD).astype(np.uint8)

    kernel: np.ndarray = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # kill speckle
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # close AA gaps

    return binary


def get_footprint_image_hull(
    fp: pcbnew.FOOTPRINT,
    side: str,
    transform: Callable[[float, float], tuple[float, float]],
    edge_bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> Optional[np.ndarray]:
    # approximate region used for grouping & blob to footprint assignment,
    polys: list[list[tuple[float, float]]] = poly_set_to_polygons(fp.GetBoundingHull())
    if polys:
        poly: list[tuple[float, float]] = max(polys, key=len)
    else:
        bbox: pcbnew.BOX2I = fp.GetBoundingBox(False, False)
        x_min: float = iu_to_mm(bbox.GetLeft())
        y_min: float = iu_to_mm(bbox.GetTop())
        x_max: float = iu_to_mm(bbox.GetRight())
        y_max: float = iu_to_mm(bbox.GetBottom())
        poly = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    width: int
    height: int
    width, height = image_size

    pts: list[tuple[float, float]] = []
    x: float
    y: float
    for x, y in poly:
        if side == BOARD_SIDE_BOTTOM:
            x = mirror_x(x, edge_bbox)
        u: float
        v: float
        u, v = transform(x, y)
        pts.append((u, v))

    arr: np.ndarray = np.array(pts, dtype=np.int32)
    arr[:, 0] = np.clip(arr[:, 0], 0, width - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, height - 1)

    return arr


def is_bbox_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int,
) -> bool:
    a_x: int
    a_y: int
    a_w: int
    a_h: int
    a_x, a_y, a_w, a_h = a

    b_x: int
    b_y: int
    b_w: int
    b_h: int
    b_x, b_y, b_w, b_h = b

    return not (
        a_x - margin > b_x + b_w
        or b_x - margin > a_x + a_w
        or a_y - margin > b_y + b_h
        or b_y - margin > a_y + a_h
    )


def get_colour_groups(
    regions_bbox: dict[str, tuple[int, int, int, int]],
    margin: int,
) -> list[list[str]]:
    refs: list[str] = list(regions_bbox)
    adj: dict[str, set[str]] = {r: set() for r in refs}

    i: int
    a: str
    b: str
    for i, a in enumerate(refs):
        for b in refs[i + 1 :]:
            if is_bbox_overlap(regions_bbox[a], regions_bbox[b], margin):
                adj[a].add(b)
                adj[b].add(a)

    color: dict[str, int] = {}
    r: str
    for r in sorted(
        refs, key=lambda r: len(adj[r]), reverse=True
    ):  # highest degree first
        used: set[int] = {color[n] for n in adj[r] if n in color}
        c: int = 0
        while c in used:
            c += 1
        color[r] = c

    groups: dict[int, list[str]] = {}
    for r, c in color.items():
        groups.setdefault(c, []).append(r)

    return list(groups.values())


def get_pcb_rectangular_geometry(pcb_file_path: str) -> dict:
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
    pcb_file_path: str,
    side: str,
    work_dir: str,
) -> tuple[np.ndarray, list[dict]]:
    board: pcbnew.BOARD = pcbnew.LoadBoard(pcb_file_path)

    e: pcbnew.BOX2I = board.GetBoardEdgesBoundingBox()
    edge_bbox: tuple[float, float, float, float] = (
        iu_to_mm(e.GetLeft()),
        iu_to_mm(e.GetTop()),
        iu_to_mm(e.GetRight()),
        iu_to_mm(e.GetBottom()),
    )

    footprint: pcbnew.FOOTPRINT
    side_footprints: list[pcbnew.FOOTPRINT] = [
        footprint
        for footprint in board.GetFootprints()
        if (BOARD_SIDE_BOTTOM if footprint.IsFlipped() else BOARD_SIDE_TOP) == side
    ]

    # get the transform, size, and baseline pixels of a flat render
    full_path: str = os.path.join(work_dir, f"{side}_full.png")
    render_pcb_differential_baseline(pcb_file_path, full_path, side)

    transform: Callable[[float, float], tuple[float, float]]
    transform = get_pcb_to_image_coordinate_transformation(full_path, edge_bbox)

    width: int
    height: int
    width, height = get_image_dimensions(full_path)

    full_rgb: np.ndarray = load_rgb_on_black(full_path)

    # per-footprint image-space hull raster and ids
    region_raster: dict[str, np.ndarray] = {}
    regions_bbox: dict[str, tuple[int, int, int, int]] = {}
    id_of: dict[str, int] = {}
    meta: dict[int, tuple[str, str]] = {}
    next_id: int = 0

    for footprint in side_footprints:
        ref: str = footprint.GetReference()
        poly: Optional[np.ndarray] = get_footprint_image_hull(
            footprint, side, transform, edge_bbox, (width, height)
        )
        if poly is None:
            continue

        next_id += 1
        id_of[ref] = next_id
        meta[next_id] = (ref, get_class_name_from_reference(ref))

        raster: np.ndarray = np.zeros((height, width), np.uint8)
        cv2.fillPoly(raster, [poly], 1)
        region_raster[ref] = raster.astype(bool)
        regions_bbox[ref] = tuple(cv2.boundingRect(poly))

    instance_mask: np.ndarray = np.zeros((height, width), dtype=np.uint16)

    # one render & difference per color group. Blobs are assigned by max hull overlap
    group: list[str]
    for group in get_colour_groups(
        regions_bbox, DIFFERENTIAL_RENDERING_COLOUR_GROUP_MARGIN_PX
    ):
        variant_path: str = os.path.join(work_dir, f"{side}_variant.png")
        differential_rendering(pcb_file_path, group, side, variant_path)

        binary: np.ndarray = get_differential_mask(
            full_rgb, load_rgb_on_black(variant_path)
        )

        n: int
        labels: np.ndarray
        stats: np.ndarray
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

        lbl: int
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] < MIN_BLOB_AREA:
                continue

            blob: np.ndarray = labels == lbl
            best_ref: Optional[str] = None
            best_overlap: int = 0

            for ref in group:
                ov: int = int(np.count_nonzero(blob & region_raster[ref]))
                if ov > best_overlap:
                    best_overlap, best_ref = ov, ref

            if (
                best_ref is not None
            ):  # blobs matching no hull = stray silkscreen -> dropped
                instance_mask[blob] = id_of[best_ref]

    # fallback: if footprint is too low-contrast to differentiate, fill its hull
    instance_id: int
    for ref, instance_id in id_of.items():
        if not np.any(instance_mask == instance_id):
            instance_mask[region_raster[ref]] = instance_id

    # polygon annotations from the mask
    annotations: list[dict] = []
    cls: str
    for instance_id, (ref, cls) in meta.items():
        blob_u8: np.ndarray = (instance_mask == instance_id).astype(np.uint8)
        if not blob_u8.any():
            continue

        contours: list[np.ndarray]
        contours, _ = cv2.findContours(
            blob_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [c for c in contours if len(c) >= 3]
        if not contours:
            continue

        x: int
        y: int
        w: int
        h: int
        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        annotations.append(
            {
                "id": instance_id,
                "reference_designator": ref,
                "class_name": cls,
                "bbox": [int(x), int(y), int(w), int(h)],
                "segmentation": [
                    c.reshape(-1).astype(float).tolist() for c in contours
                ],
            }
        )

    return instance_mask, annotations


def get_rectangular_annotation_instance_mask(
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


def process_pcb(
    pcb_file_path: str, output_directory: str, process_both_sides: bool = True
) -> None:
    if not os.path.exists(pcb_file_path):
        raise FileNotFoundError(f"{pcb_file_path} does not exist.")
    os.makedirs(os.path.dirname(output_directory), exist_ok=True)

    all_sides: list[str] = (
        [BOARD_SIDE_TOP, BOARD_SIDE_BOTTOM] if process_both_sides else [BOARD_SIDE_TOP]
    )
    side: str
    for side in all_sides:
        image_path: str = f"{output_directory}/{side}_image.png"
        render_pcb(pcb_file_path, image_path, side)
        print(f"Image for {side} rendered")

        segmentation_mask: np.ndarray
        annotations: list[dict]
        work_dir_ctx: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        work_dir: str
        with work_dir_ctx as work_dir:
            segmentation_mask, annotations = get_annotation_instance_mask(
                pcb_file_path, side, work_dir
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
    pcb_file_directory: str,
    output_directory: str,
    start_num: int,
    end_num: int,
    process_both_sides: bool = True,
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
            process_pcb(
                pcb_file_path,
                curr_pcb_output_directory,
                process_both_sides=process_both_sides,
            )
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
    validation_directory: str,
    test_directory: str,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = SEED,
) -> None:
    if os.path.isdir(train_directory):
        shutil.rmtree(train_directory)
    if os.path.isdir(validation_directory):
        shutil.rmtree(validation_directory)
    if os.path.isdir(test_directory):
        shutil.rmtree(test_directory)

    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(validation_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    pcb_folders: list[str] = [
        folder
        for folder in os.listdir(source_directory)
        if os.path.isdir(os.path.join(source_directory, folder))
    ]

    # seed random
    random.seed(seed)
    random.shuffle(pcb_folders)

    num_folders: int = len(pcb_folders)
    test_split_index: int = int(num_folders * (1 - test_ratio - validation_ratio))
    validation_split_index: int = int(num_folders * (1 - test_ratio))

    train_folders: list[str] = pcb_folders[:test_split_index]
    validation_folders: list[str] = pcb_folders[test_split_index:validation_split_index]
    test_folders: list[str] = pcb_folders[validation_split_index:]

    print(f"Total number of PCBs: {num_folders}")
    print(f"Train PCBs: {len(train_folders)}")
    print(f"Validation PCBs: {len(validation_folders)}")
    print(f"Test PCBs: {len(test_folders)}")

    for folder in train_folders:
        shutil.copytree(
            os.path.join(source_directory, folder),
            os.path.join(train_directory, folder),
        )

    for folder in validation_folders:
        shutil.copytree(
            os.path.join(source_directory, folder),
            os.path.join(validation_directory, folder),
        )

    for folder in test_folders:
        shutil.copytree(
            os.path.join(source_directory, folder),
            os.path.join(test_directory, folder),
        )


if __name__ == "__main__":
    # wx.Log.SetLogLevel(wx.LOG_Error)
    # app: wx.App = wx.App(False)
    # process_multiple_pcbs(
    #     "data/open-schematics", "data/synthetic", 1, 2500, process_both_sides=False
    # )
    split_dataset(
        "data/synthetic",
        "data/synthetic_split/train",
        "data/synthetic_split/validation",
        "data/synthetic_split/test",
    )
