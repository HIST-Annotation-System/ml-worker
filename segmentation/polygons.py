"""Cellpose label masks to the polygons the annotation editor can use."""

import cv2
import numpy as np
import numpy.typing as npt
from scipy import ndimage

MIN_RING_VERTICES = 3
# OpenCV traces the centers of the boundary pixels; a pixel's center is half a pixel from its corner.
PIXEL_CENTER_OFFSET = 0.5
COORDINATE_DECIMALS = 1

type Ring = list[tuple[float, float]]


def masks_to_polygons(masks: npt.NDArray[np.integer], simplify_px: float) -> list[Ring]:
    """One simplified open ring per labelled cell.

    The backend's annotations have a single exterior ring and no holes, so a cell keeps the outer boundary of its
    largest piece. Cells too small to outline with a real area are dropped.
    """
    rings: list[Ring] = []
    # find_objects gives each label's bounding box, so no cell needs a pass over the whole image.
    for index, box in enumerate(ndimage.find_objects(masks)):
        if box is None:
            continue
        cell = (masks[box] == index + 1).astype(np.uint8)
        contours, _ = cv2.findContours(
            cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(box[1].start, box[0].start)
        )
        if not contours:
            continue
        outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), simplify_px, closed=True)
        if len(outline) < MIN_RING_VERTICES or cv2.contourArea(outline) <= 0:
            continue
        rings.append(
            [
                (
                    round(float(x) + PIXEL_CENTER_OFFSET, COORDINATE_DECIMALS),
                    round(float(y) + PIXEL_CENTER_OFFSET, COORDINATE_DECIMALS),
                )
                for x, y in outline.reshape(-1, 2)
            ]
        )
    return rings
