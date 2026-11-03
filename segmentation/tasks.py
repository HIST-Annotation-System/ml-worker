import logging
import sys
from functools import lru_cache
from typing import Any

import torch
from botocore.exceptions import BotoCoreError, ClientError
from celery.signals import worker_init

from segmentation import inference
from segmentation.celery_app import celery_app
from segmentation.config import get_settings
from segmentation.contract import (
    RESULT_KEY,
    RESULTS_QUEUE,
    SCHEMA_VERSION,
    STATUS_DONE,
    STATUS_FAILED,
    TASK_RUN,
    TASK_SAVE_RESULT,
    SegmentationResult,
)
from segmentation.errors import SegmentationError
from segmentation.polygons import masks_to_polygons
from segmentation.storage import Storage

logger = logging.getLogger(__name__)

UNSUPPORTED_SCHEMA = "The worker does not support job format {schema_version}; update ml-worker or the backend."
OUT_OF_GPU_MEMORY = "The GPU ran out of memory on this image."
UNEXPECTED_ERROR = "Unexpected error during segmentation."


@worker_init.connect
def _load_model_at_startup(**_: Any) -> None:
    try:
        inference.load_model(get_settings())
    except Exception:
        # Celery's signal dispatch swallows ordinary exceptions; a worker that cannot infer must not take jobs.
        logger.exception("Could not load the segmentation model")
        sys.exit(1)


@lru_cache(maxsize=1)
def _storage() -> Storage:
    return Storage(get_settings())


def _report(hotspot_id: str, status: str, **fields: Any) -> None:
    celery_app.send_task(
        TASK_SAVE_RESULT,
        kwargs={"schema_version": SCHEMA_VERSION, "hotspot_id": hotspot_id, "status": status, **fields},
        queue=RESULTS_QUEUE,
    )


def _segment_and_store(hotspot_id: str, image_key: str) -> str:
    """Runs the model on the image and stores the polygons; returns the key of the stored file."""
    settings = get_settings()
    image = _storage().load_rgb(image_key, settings.SEGMENTATION_MAX_PIXELS)
    masks = inference.segment(image, settings)
    result = SegmentationResult(
        schema_version=SCHEMA_VERSION,
        hotspot_id=hotspot_id,
        model_version=inference.model_version(settings),
        image_width=image.shape[1],
        image_height=image.shape[0],
        cells=masks_to_polygons(masks, settings.POLYGON_SIMPLIFY_PX),
    )
    key = RESULT_KEY.format(hotspot_id=hotspot_id)
    _storage().put_json(key, result.model_dump_json())
    logger.info("Hotspot %s: %d cells", hotspot_id, len(result.cells))
    return key


@celery_app.task(name=TASK_RUN)
def run_segmentation(schema_version: int, hotspot_id: str, image_key: str) -> None:
    try:
        if schema_version != SCHEMA_VERSION:
            raise SegmentationError(UNSUPPORTED_SCHEMA.format(schema_version=schema_version))
        key = _segment_and_store(hotspot_id, image_key)
    except SegmentationError as exc:
        logger.warning("Hotspot %s failed: %s", hotspot_id, exc)
        _report(hotspot_id, STATUS_FAILED, error=str(exc))
    except torch.cuda.OutOfMemoryError:
        logger.warning("Hotspot %s: out of GPU memory", hotspot_id)
        torch.cuda.empty_cache()
        _report(hotspot_id, STATUS_FAILED, error=OUT_OF_GPU_MEMORY)
    except (BotoCoreError, ClientError):
        # Possibly temporary: the hotspot stays pending, and the backend sends the job again later.
        logger.exception("Storage error on hotspot %s; leaving it pending", hotspot_id)
    except Exception:
        logger.exception("Segmentation of hotspot %s failed", hotspot_id)
        _report(hotspot_id, STATUS_FAILED, error=UNEXPECTED_ERROR)
    else:
        _report(hotspot_id, STATUS_DONE, model_version=inference.model_version(get_settings()), result_key=key)
