import numpy as np

from segmentation.polygons import masks_to_polygons


def _area(ring: list[tuple[float, float]]) -> float:
    xs, ys = np.array(ring).T
    return float(abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))) / 2)


def test_each_label_becomes_one_ring_in_image_coordinates() -> None:
    masks = np.zeros((100, 200), dtype=np.uint16)
    masks[10:30, 20:50] = 1
    masks[60:90, 150:190] = 2

    (first_label, first), (second_label, second) = masks_to_polygons(masks, simplify_px=1.0)

    assert (first_label, second_label) == (1, 2)

    assert (min(x for x, _ in first), min(y for _, y in first)) == (20.5, 10.5)
    assert (max(x for x, _ in second), max(y for _, y in second)) == (189.5, 89.5)
    assert _area(first) > 0.8 * 20 * 30


def test_skipped_label_numbers_and_empty_masks() -> None:
    masks = np.zeros((50, 50), dtype=np.uint16)
    assert masks_to_polygons(masks, simplify_px=1.0) == []
    masks[5:20, 5:20] = 7
    ((label, _),) = masks_to_polygons(masks, simplify_px=1.0)
    assert label == 7


def test_split_cell_keeps_its_largest_piece() -> None:
    masks = np.zeros((60, 60), dtype=np.uint16)
    masks[5:10, 5:10] = 1
    masks[20:50, 20:50] = 1

    ((_, ring),) = masks_to_polygons(masks, simplify_px=1.0)

    assert min(x for x, _ in ring) == 20.5


def test_cell_with_no_area_is_dropped() -> None:
    masks = np.zeros((20, 20), dtype=np.uint16)
    masks[5, 2:18] = 1
    assert masks_to_polygons(masks, simplify_px=1.0) == []


def test_simplifying_reduces_vertices_of_a_round_cell() -> None:
    yy, xx = np.ogrid[:200, :200]
    masks = (((xx - 100) ** 2 + (yy - 100) ** 2) < 80**2).astype(np.uint16)

    ((_, exact),) = masks_to_polygons(masks, simplify_px=0.1)
    ((_, simple),) = masks_to_polygons(masks, simplify_px=1.0)

    assert len(simple) < len(exact)
    assert _area(simple) > 0.97 * np.pi * 80**2
