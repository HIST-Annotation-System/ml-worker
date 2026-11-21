"""The worker and the backend keep separate copies of the job/result contract (see contract.py). Nothing else checks
that they match, so this test reads the backend's copy from source (the backend's dependencies are not installed here)
and compares the two."""

import ast
from pathlib import Path

import pytest

from segmentation import contract

BACKEND_CONTRACT = Path(__file__).resolve().parents[2] / "backend" / "app" / "services" / "segmentation.py"
SHARED_CONSTANTS = [
    "SCHEMA_VERSION",
    "JOBS_QUEUE",
    "RESULTS_QUEUE",
    "TASK_RUN",
    "TASK_SAVE_RESULT",
    "STATUS_DONE",
    "STATUS_FAILED",
]
SHARED_MODELS = ["Cell", "SegmentationResult"]

pytestmark = pytest.mark.skipif(not BACKEND_CONTRACT.exists(), reason="backend sources are not checked out")


WORKER_CONTRACT = Path(contract.__file__)


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def backend_constants() -> dict[str, object]:
    values: dict[str, object] = {}
    for node in parse(BACKEND_CONTRACT).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return values


def fields_of(path: Path, class_name: str) -> dict[str, str]:
    for node in parse(path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.target.id: ast.unparse(item.annotation)
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
    raise AssertionError(f"{class_name} is missing from {path}")


@pytest.mark.parametrize("name", SHARED_CONSTANTS)
def test_constants_match(name: str) -> None:
    assert backend_constants()[name] == getattr(contract, name)


@pytest.mark.parametrize("name", SHARED_MODELS)
def test_result_models_have_the_same_fields(name: str) -> None:
    assert fields_of(BACKEND_CONTRACT, name) == fields_of(WORKER_CONTRACT, name)


def test_result_key_is_under_the_hotspots_prefix() -> None:
    # The backend deletes "hotspots/{id}/" with the hotspot; the result must live there to go with it.
    assert contract.RESULT_KEY.format(hotspot_id="abc").startswith("hotspots/abc/")
