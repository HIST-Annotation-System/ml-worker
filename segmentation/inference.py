import logging
import re
from functools import lru_cache
from importlib.metadata import version
from typing import Any, NamedTuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from classpose.model_configs import DEFAULT_MODEL_CONFIGS, ModelConfig
from classpose.models import ClassposeModel
from scipy.special import expit

from segmentation.config import Settings

logger = logging.getLogger(__name__)

NO_GPU = "No CUDA GPU is visible to the worker."
UNKNOWN_MODEL = "CLASSPOSE_MODEL is '{name}'; it must be one of: {known}."
CLASS_COUNT_MISMATCH = "The weights of '{name}' predict {found} classes, but its configuration names {named}."
RESCALE_LIMITED = "Rescaling by %.2f to the model's resolution would exceed SEGMENTATION_MAX_PIXELS; using %.2f."

# How the weights name the layers of a U-Net classification head; without them the head is a single convolution.
HEAD_LAYER = re.compile(r"out_class\.encoder_blocks\.[0-9]+\.block.conv1.weight")
# The weight whose second dimension is the number of classes, background included.
CLASS_WEIGHT = "W3"
# Class 0 in the model's output: a cell with it has no predicted type.
BACKGROUND_CLASS = 0
# Where `ClassposeModel.eval` puts the raw cell-probability logits (Ly, Lx) and the class logits (classes, Ly, Lx) in
# its `flows`, at the size of the image it was given (Classpose commit pinned in scripts/fetch_classpose.py).
CELLPROB_FLOW = 2
CLASS_LOGITS_FLOW = 3
CONFIDENCE_DECIMALS = 4
NO_CONFIDENCE = "The model's output has no usable %s; cells are saved without that confidence."


class Segmentation(NamedTuple):
    """The label mask of an image (0 is background, each cell has its own number) and, indexed by label, each cell's
    type and the model's confidence in its outline and in its type; None where there is none."""

    masks: npt.NDArray[np.integer]
    cell_types: list[str | None]
    outline_confidence: list[float | None]
    class_confidence: list[float | None]


def model_version(settings: Settings) -> str:
    return f"classpose-{version('classpose')}/{settings.CLASSPOSE_MODEL}"


def _head_structure(path: str) -> tuple[list[int] | None, int]:
    """The classification head's layer sizes and the number of classes, which the weights file does not state.

    Classpose's own `infer_structure` does this, but importing its module resets the root logger's handlers.
    """
    weights = torch.load(path, map_location="cpu", weights_only=True)
    structure = [weights[key].shape[0] for key in weights if HEAD_LAYER.search(key)]
    return structure or None, weights[CLASS_WEIGHT].shape[1]


@lru_cache(maxsize=1)
def _load_model(name: str, precision: str) -> tuple[ClassposeModel, ModelConfig]:
    if not torch.cuda.is_available():
        raise RuntimeError(NO_GPU)
    if name not in DEFAULT_MODEL_CONFIGS:
        raise RuntimeError(UNKNOWN_MODEL.format(name=name, known=", ".join(sorted(DEFAULT_MODEL_CONFIGS))))
    config = ModelConfig(**DEFAULT_MODEL_CONFIGS[name])
    config.download_if_necessary()
    structure, class_count = _head_structure(config.path)
    if class_count != len(config.cell_types) + 1:
        raise RuntimeError(
            CLASS_COUNT_MISMATCH.format(name=name, found=class_count - 1, named=len(config.cell_types))
        )
    model = ClassposeModel(
        gpu=True,
        pretrained_model=config.path,
        nclasses=class_count,
        feature_transformation_structure=structure,
        precision=precision,
    )
    model.net.eval()
    return model, config


def load_model(settings: Settings) -> None:
    """Loads the model once, at worker startup, so a missing GPU shows up before any job is taken."""
    _load_model(settings.CLASSPOSE_MODEL, settings.CLASSPOSE_PRECISION)


def _scale_to_model(mpp: float | None, model_mpp: float, pixels: int, max_pixels: int) -> float:
    """How much to resize an image so its cells are as large as in the model's training data."""
    if mpp is None or mpp <= 0:
        return 1.0
    scale = mpp / model_mpp
    # Enlarging multiplies the inference time; an image is never enlarged past the size the worker accepts.
    limit = max(1.0, (max_pixels / pixels) ** 0.5)
    if scale > limit:
        logger.warning(RESCALE_LIMITED, scale, limit)
        return limit
    return scale


def _label_means(
    masks: npt.NDArray[np.integer], label_count: int, values: npt.NDArray[np.floating]
) -> list[float | None]:
    """The mean of `values` over each label's pixels, indexed by label; None for the background and empty labels."""
    flat = masks.ravel()
    counts = np.bincount(flat, minlength=label_count)
    sums = np.bincount(flat, weights=values.ravel().astype(np.float64), minlength=label_count)
    return [
        round(float(sums[label] / counts[label]), CONFIDENCE_DECIMALS) if label and counts[label] else None
        for label in range(label_count)
    ]


def _confidences(
    masks: npt.NDArray[np.integer], label_classes: npt.NDArray[np.intp], flows: Any
) -> tuple[list[float | None], list[float | None]]:
    """Per label: the mean cell probability of its pixels, and the mean probability of its type over its pixels."""
    label_count = len(label_classes)
    unknown: list[float | None] = [None] * label_count
    # A model without a class head has no class logits (None, which becomes a 0-dimensional array here).
    cellprob = np.squeeze(np.asarray(flows[CELLPROB_FLOW], dtype=np.float32))
    logits = np.squeeze(np.asarray(flows[CLASS_LOGITS_FLOW], dtype=np.float32))
    if cellprob.shape == masks.shape:
        outline = _label_means(masks, label_count, expit(cellprob))
    else:
        logger.warning(NO_CONFIDENCE, "cell probability")
        outline = unknown
    if logits.ndim != 3 or logits.shape[1:] != masks.shape:
        logger.warning(NO_CONFIDENCE, "class probabilities")
        return outline, unknown
    # Softmax probability of each pixel's cell type, one class at a time to keep memory at one image's worth.
    peak = logits.max(axis=0)
    total = np.zeros_like(peak)
    for class_logits_map in logits:
        total += np.exp(class_logits_map - peak)
    chosen = np.take_along_axis(logits, label_classes[masks][np.newaxis], axis=0)[0]
    probabilities = np.exp(chosen - peak) / total
    typed = _label_means(masks, label_count, probabilities)
    return outline, [None if label_classes[label] == BACKGROUND_CLASS else typed[label] for label in range(label_count)]


def segment(image: npt.NDArray[np.uint8], mpp: float | None, settings: Settings) -> Segmentation:
    """Segments an RGB image into cells, each with its type and the model's confidence (see `Segmentation`).

    `mpp` is the image's microns per pixel. When known, the model sees the image at its training resolution.
    """
    model, config = _load_model(settings.CLASSPOSE_MODEL, settings.CLASSPOSE_PRECISION)
    height, width = image.shape[:2]
    scale = _scale_to_model(mpp, config.mpp, height * width, settings.SEGMENTATION_MAX_PIXELS)
    if scale != 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    masks, flows, class_masks, _styles = model.eval(
        image,
        batch_size=settings.CLASSPOSE_BATCH_SIZE,
        flow_threshold=settings.CLASSPOSE_FLOW_THRESHOLD,
        cellprob_threshold=settings.CLASSPOSE_CELLPROB_THRESHOLD,
    )
    # Every pixel of a cell carries the cell's class, so any of them gives the class of its label.
    label_classes = np.zeros(int(masks.max()) + 1, dtype=np.intp)
    label_classes[masks.ravel()] = class_masks.ravel()
    # Before resizing back: the model's outputs have the size of the image it saw.
    outline_confidence, class_confidence = _confidences(masks, label_classes, flows)
    if scale != 1.0:
        # OpenCV cannot resize 32-bit integers; labels beyond 16 bits fit a float32 exactly.
        masks = cv2.resize(masks.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    names: list[str | None] = [None, *config.cell_types]
    return Segmentation(masks, [names[index] for index in label_classes], outline_confidence, class_confidence)
