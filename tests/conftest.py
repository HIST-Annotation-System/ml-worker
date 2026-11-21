"""Lets the worker's modules be imported on a machine without a GPU, the Classpose package or a broker.

* Storage settings are required at import time (`segmentation.celery_app` reads them), so dummy values are set
  unless the environment already has real ones.
* Classpose is replaced by a small stand-in when it is not installed; tests that need its behaviour patch
  `segmentation.inference` directly.
"""

import os
import sys
import types
from typing import Any

for name, value in {
    "BUCKET_ENDPOINT": "http://storage.invalid",
    "BUCKET_ACCESS_KEY": "key",
    "BUCKET_SECRET_KEY": "secret",
    "BUCKET_NAME": "bucket",
    "BUCKET_REGION": "eu-north-1",
    # A broker URL that is never contacted: tasks are called directly and `send_task` is patched.
    "CELERY_BROKER_URL": "memory://",
}.items():
    os.environ.setdefault(name, value)


def _install_classpose_stub() -> None:
    try:
        import classpose.model_configs  # noqa: F401
        import classpose.models  # noqa: F401
    except ImportError:
        pass
    else:
        return

    class ModelConfig:
        def __init__(self, path: str, mpp: float, cell_types: list[str], **_: Any) -> None:
            self.path, self.mpp, self.cell_types = path, mpp, cell_types

        def download_if_necessary(self) -> None:
            pass

    class ClassposeModel:  # replaced per test where it matters
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    package = types.ModuleType("classpose")
    configs = types.ModuleType("classpose.model_configs")
    configs.DEFAULT_MODEL_CONFIGS = {}  # type: ignore[attr-defined]
    configs.ModelConfig = ModelConfig  # type: ignore[attr-defined]
    models = types.ModuleType("classpose.models")
    models.ClassposeModel = ClassposeModel  # type: ignore[attr-defined]
    package.model_configs, package.models = configs, models  # type: ignore[attr-defined]
    sys.modules.update({"classpose": package, "classpose.model_configs": configs, "classpose.models": models})


_install_classpose_stub()
