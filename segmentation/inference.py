from functools import lru_cache

import cellpose
import numpy as np
import numpy.typing as npt
import torch
from cellpose import models

from segmentation.config import Settings

NO_GPU = "No CUDA GPU is visible to the worker."


def model_version(settings: Settings) -> str:
    return f"cellpose-{cellpose.version}/{settings.CELLPOSE_MODEL}"


@lru_cache(maxsize=1)
def _load_model(name: str) -> models.CellposeModel:
    if not torch.cuda.is_available():
        raise RuntimeError(NO_GPU)
    return models.CellposeModel(gpu=True, pretrained_model=name)


def load_model(settings: Settings) -> None:
    """Loads the model once, at worker startup, so a missing GPU shows up before any job is taken."""
    _load_model(settings.CELLPOSE_MODEL)


def segment(image: npt.NDArray[np.uint8], settings: Settings) -> npt.NDArray[np.integer]:
    """The label mask of an RGB image: 0 is background, each cell has its own number."""
    masks, _flows, _styles = _load_model(settings.CELLPOSE_MODEL).eval(
        image,
        batch_size=settings.CELLPOSE_BATCH_SIZE,
        flow_threshold=settings.CELLPOSE_FLOW_THRESHOLD,
        cellprob_threshold=settings.CELLPOSE_CELLPROB_THRESHOLD,
    )
    return masks
