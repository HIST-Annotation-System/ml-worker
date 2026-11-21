"""How a job ends: what the worker reports back to the backend for each outcome."""

import json
from typing import Any

import numpy as np
import pytest
import torch
from botocore.exceptions import ClientError, EndpointConnectionError

from segmentation import tasks
from segmentation.contract import (
    RESULTS_QUEUE,
    SCHEMA_VERSION,
    STATUS_DONE,
    STATUS_FAILED,
    TASK_SAVE_RESULT,
    SegmentationResult,
)
from segmentation.errors import SegmentationError

HOTSPOT = "0b7c8c3e-0000-4000-8000-000000000001"


class Reports:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def __call__(self, name: str, kwargs: dict[str, Any], queue: str) -> None:
        assert (name, queue) == (TASK_SAVE_RESULT, RESULTS_QUEUE)
        self.sent.append(kwargs)


@pytest.fixture
def reports(monkeypatch: pytest.MonkeyPatch) -> Reports:
    sent = Reports()
    monkeypatch.setattr(tasks.celery_app, "send_task", sent)
    monkeypatch.setattr(tasks.inference, "model_version", lambda settings: "classpose-test/consep")
    return sent


def fail_with(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    def segment_and_store(hotspot_id: str, image_key: str, mpp: float | None) -> str:
        raise error

    monkeypatch.setattr(tasks, "_segment_and_store", segment_and_store)


def run(**overrides: Any) -> None:
    arguments: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "hotspot_id": HOTSPOT, "image_key": "k", "mpp": 0.5}
    tasks.run_segmentation(**{**arguments, **overrides})


def test_success_reports_the_result_key(monkeypatch: pytest.MonkeyPatch, reports: Reports) -> None:
    calls: list[tuple[str, str, float | None]] = []

    def segment_and_store(hotspot_id: str, image_key: str, mpp: float | None) -> str:
        calls.append((hotspot_id, image_key, mpp))
        return "hotspots/x/segmentation.json"

    monkeypatch.setattr(tasks, "_segment_and_store", segment_and_store)
    run(image_key="hotspots/x/image.png")
    assert calls == [(HOTSPOT, "hotspots/x/image.png", 0.5)]
    assert reports.sent == [
        {
            "schema_version": SCHEMA_VERSION,
            "hotspot_id": HOTSPOT,
            "status": STATUS_DONE,
            "model_version": "classpose-test/consep",
            "result_key": "hotspots/x/segmentation.json",
        }
    ]


def test_unsupported_job_format_fails_without_running(monkeypatch: pytest.MonkeyPatch, reports: Reports) -> None:
    fail_with(monkeypatch, AssertionError("must not run"))
    run(schema_version=SCHEMA_VERSION + 1)
    [report] = reports.sent
    assert report["status"] == STATUS_FAILED
    assert f"job format {SCHEMA_VERSION + 1}" in report["error"]


def test_permanent_error_is_reported_with_its_message(monkeypatch: pytest.MonkeyPatch, reports: Reports) -> None:
    fail_with(monkeypatch, SegmentationError("The hotspot image no longer exists."))
    run()
    assert reports.sent == [
        {"schema_version": SCHEMA_VERSION, "hotspot_id": HOTSPOT, "status": STATUS_FAILED,
         "error": "The hotspot image no longer exists."}
    ]


def test_out_of_gpu_memory_frees_the_cache(monkeypatch: pytest.MonkeyPatch, reports: Reports) -> None:
    emptied: list[bool] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    fail_with(monkeypatch, torch.cuda.OutOfMemoryError("CUDA out of memory"))
    run()
    assert emptied == [True]
    assert reports.sent[0]["error"] == tasks.OUT_OF_GPU_MEMORY


@pytest.mark.parametrize(
    "error",
    [
        EndpointConnectionError(endpoint_url="http://s3"),
        ClientError({"Error": {"Code": "SlowDown"}}, "GetObject"),
    ],
    ids=["unreachable", "throttled"],
)
def test_storage_errors_leave_the_hotspot_pending(
    monkeypatch: pytest.MonkeyPatch, reports: Reports, error: Exception
) -> None:
    fail_with(monkeypatch, error)
    run()
    assert reports.sent == []


def test_unexpected_error_hides_the_details(monkeypatch: pytest.MonkeyPatch, reports: Reports) -> None:
    fail_with(monkeypatch, ValueError("secret internals"))
    run()
    assert reports.sent[0]["error"] == tasks.UNEXPECTED_ERROR


class FakeStorage:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.stored: dict[str, str] = {}

    def load_rgb(self, key: str, max_pixels: int) -> np.ndarray:
        return self.image

    def put_json(self, key: str, body: str) -> None:
        self.stored[key] = body


def test_segment_and_store_writes_the_result_file(monkeypatch: pytest.MonkeyPatch) -> None:
    storage = FakeStorage(np.zeros((30, 50, 3), dtype=np.uint8))
    masks = np.zeros((30, 50), dtype=np.int32)
    masks[5:15, 5:15] = 1
    masks[20:28, 30:45] = 2
    monkeypatch.setattr(tasks, "_storage", lambda: storage)
    monkeypatch.setattr(tasks.inference, "segment", lambda image, mpp, settings: (masks, [None, "Tumor", None]))
    monkeypatch.setattr(tasks.inference, "model_version", lambda settings: "v")

    key = tasks._segment_and_store(HOTSPOT, "img", None)

    assert key == f"hotspots/{HOTSPOT}/segmentation.json"
    result = SegmentationResult.model_validate(json.loads(storage.stored[key]))
    assert (result.schema_version, result.hotspot_id, result.model_version) == (SCHEMA_VERSION, HOTSPOT, "v")
    assert (result.image_width, result.image_height) == (50, 30)
    assert [cell.cell_type for cell in result.cells] == ["Tumor", None]
    assert all(len(cell.ring) >= 3 for cell in result.cells)
