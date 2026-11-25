"""What the worker and the backend agree on. Nothing checks this across the two repositories: a change here must
be made in `backend/app/services/segmentation.py` too, and `SCHEMA_VERSION` raised when the format changes."""

from pydantic import BaseModel

SCHEMA_VERSION = 2

# Hyphens only: SQS queue names may not contain dots.
JOBS_QUEUE = "segmentation-jobs"
RESULTS_QUEUE = "segmentation-results"

TASK_RUN = "segmentation.run"
TASK_SAVE_RESULT = "segmentation.save_result"

STATUS_DONE = "done"
STATUS_FAILED = "failed"

# Under the hotspot's own prefix, so deleting the hotspot deletes this too.
RESULT_KEY = "hotspots/{hotspot_id}/segmentation.json"


class Cell(BaseModel):
    # An open ring of [x, y] in image pixels.
    ring: list[tuple[float, float]]
    # The model's name for the cell's type; null when it gave none.
    cell_type: str | None
    # Added without raising SCHEMA_VERSION: optional both ways, so either side may be updated first.
    # How sure the model is of the outline (mean cell probability of its pixels), 0..1.
    outline_confidence: float | None = None
    # How sure it is of the type (that type's mean probability over its pixels), 0..1; null without a type.
    class_confidence: float | None = None


class SegmentationResult(BaseModel):
    """The file stored at RESULT_KEY. Messages only carry its key: polygons do not reliably fit in an SQS message."""

    schema_version: int
    hotspot_id: str
    model_version: str
    image_width: int
    image_height: int
    cells: list[Cell]
