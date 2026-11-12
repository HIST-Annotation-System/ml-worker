import logging
import re
from functools import lru_cache
from importlib.metadata import version

import cv2
import numpy as np
import numpy.typing as npt
import torch
from classpose.model_configs import DEFAULT_MODEL_CONFIGS, ModelConfig
from classpose.models import ClassposeModel

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


def segment(
    image: npt.NDArray[np.uint8], mpp: float | None, settings: Settings
) -> tuple[npt.NDArray[np.integer], list[str | None]]:
    """The label mask of an RGB image (0 is background, each cell has its own number), and the cell type of each
    label, indexed by label; None where the model gave no type.

    `mpp` is the image's microns per pixel. When known, the model sees the image at its training resolution.
    """
    model, config = _load_model(settings.CLASSPOSE_MODEL, settings.CLASSPOSE_PRECISION)
    height, width = image.shape[:2]
    scale = _scale_to_model(mpp, config.mpp, height * width, settings.SEGMENTATION_MAX_PIXELS)
    if scale != 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    masks, _flows, class_masks, _styles = model.eval(
        image,
        batch_size=settings.CLASSPOSE_BATCH_SIZE,
        flow_threshold=settings.CLASSPOSE_FLOW_THRESHOLD,
        cellprob_threshold=settings.CLASSPOSE_CELLPROB_THRESHOLD,
    )
    # Every pixel of a cell carries the cell's class, so any of them gives the class of its label.
    label_classes = np.zeros(int(masks.max()) + 1, dtype=np.intp)
    label_classes[masks.ravel()] = class_masks.ravel()
    if scale != 1.0:
        # OpenCV cannot resize 32-bit integers; labels beyond 16 bits fit a float32 exactly.
        masks = cv2.resize(masks.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    names: list[str | None] = [None, *config.cell_types]
    return masks, [names[index] for index in label_classes]
