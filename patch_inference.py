from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
from skimage import measure


PATCH_SIZE = 256
OVERLAP = 64
MAX_DIM = 2000
MAX_SIZE = 10

Array = npt.NDArray[Any]
BoolMask = npt.NDArray[np.bool_]
IntMask = npt.NDArray[np.int32]


class Candidate(TypedDict):
    local_mask: BoolMask
    y_start: int
    x_start: int
    priority: float
    area: float


def _patch_starts(dim_size: int, patch_size: int, overlap: int) -> list[int]:
    stride = patch_size - overlap
    if dim_size <= patch_size:
        return [0]
    starts = list(range(0, dim_size - patch_size + 1, stride))
    if starts[-1] + patch_size < dim_size:
        starts.append(dim_size - patch_size)
    return starts


def _touches_non_image_border(
    region_slice: tuple[slice, slice],
    tile_shape: tuple[int, ...],
    tile_x: int,
    tile_y: int,
    image_w: int,
    image_h: int,
) -> bool:
    row_slice, col_slice = region_slice
    tile_h, tile_w = tile_shape
    return (
        (tile_y != 0 and row_slice.start == 0)
        or (tile_y + row_slice.stop != image_h and row_slice.stop == tile_h)
        or (tile_x != 0 and col_slice.start == 0)
        or (tile_x + col_slice.stop != image_w and col_slice.stop == tile_w)
    )


def _remove_border_instances(
    instances: Array, tile_x: int, tile_y: int, image_w: int, image_h: int
) -> None:
    for prop in measure.regionprops(instances):
        if _touches_non_image_border(
            prop.slice, instances.shape, tile_x, tile_y, image_w, image_h
        ):
            instances[instances == prop.label] = 0


def _instance_priority(region: Any, patch_size: int) -> float:
    cy, cx = region.centroid_local
    tile_center = patch_size / 2.0
    max_dist = tile_center * 2**0.5
    dist = ((cx - tile_center) ** 2 + (cy - tile_center) ** 2) ** 0.5
    return 1.0 - dist / max_dist


def _collect_regions(
    cleaned_instances: Array,
    tile_x: int,
    tile_y: int,
    patch_size: int,
    candidates: list[Candidate],
) -> None:
    for region in measure.regionprops(cleaned_instances):
        if region.area == 0:
            continue
        r = region.slice
        label = region.label
        local_mask = (cleaned_instances[r[0], r[1]] == label).astype(bool)
        candidates.append(
            {
                "local_mask": local_mask,
                "y_start": tile_y + r[0].start,
                "x_start": tile_x + r[1].start,
                "priority": _instance_priority(region, patch_size),
                "area": region.area,
            }
        )


def predict_large_image(
    predict_patch_fn: Callable[[Array], Array],
    img_rgb: Array,
    patch_size: int = PATCH_SIZE,
    overlap: int = OVERLAP,
    iou_thresh: float = 0.5,
    max_dim: int = MAX_DIM,
    max_size: int = MAX_SIZE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> IntMask:
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must be non-negative and smaller than patch_size")

    height, width = img_rgb.shape[:2]

    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Image must be at least {patch_size}x{patch_size} pixels, "
            f"got {width}x{height}"
        )
    if height > max_dim or width > max_dim:
        raise ValueError(
            f"Image too large (max {max_dim}px per side), got {width}x{height}"
        )

    x_starts = _patch_starts(width, patch_size, overlap)
    y_starts = _patch_starts(height, patch_size, overlap)

    total_tiles = len(x_starts) * len(y_starts)
    tiles_done = 0

    candidates: list[Candidate] = []

    for y in y_starts:
        for x in x_starts:
            tile = img_rgb[y : y + patch_size, x : x + patch_size]
            instances = predict_patch_fn(tile)

            _remove_border_instances(instances, x, y, width, height)

            _collect_regions(instances, x, y, patch_size, candidates)

            tiles_done += 1
            if progress_callback is not None:
                progress_callback(tiles_done, total_tiles)

    candidates.sort(key=lambda c: c["priority"], reverse=True)

    canvas = np.zeros((height, width), dtype=np.int32)
    next_id = 1

    for candidate in candidates:
        y0, x0 = candidate["y_start"], candidate["x_start"]
        local_mask = candidate["local_mask"]
        h, w = local_mask.shape
        slice_ = (slice(y0, y0 + h), slice(x0, x0 + w))
        canvas_view = canvas[slice_]
        overlap_pixels = (canvas_view > 0) & local_mask
        ratio = (
            overlap_pixels.sum() / candidate["area"] if candidate["area"] > 0 else 1.0
        )

        if ratio <= iou_thresh:
            unoccupied = canvas_view == 0
            write_mask = local_mask & unoccupied
            if write_mask.any():
                canvas_view[write_mask] = next_id
                next_id += 1

    if max_size > 0:
        for prop in measure.regionprops(canvas):
            if prop.area <= max_size:
                canvas[canvas == prop.label] = 0

    return canvas
