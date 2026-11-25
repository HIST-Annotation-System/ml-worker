import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from segmentation import inference
from segmentation.config import Settings
from segmentation.inference import _head_structure, _scale_to_model

MAX_PIXELS = 16_000_000


class TestScaleToModel:
    @pytest.mark.parametrize("mpp", [None, 0.0, -0.5])
    def test_unknown_resolution_keeps_the_image(self, mpp: float | None) -> None:
        assert _scale_to_model(mpp, 0.5, 1000, MAX_PIXELS) == 1.0

    def test_finer_image_is_shrunk_to_the_models_resolution(self) -> None:
        assert _scale_to_model(0.25, 0.5, 1_000_000, MAX_PIXELS) == pytest.approx(0.5)

    def test_coarser_image_is_enlarged(self) -> None:
        assert _scale_to_model(1.0, 0.5, 1_000_000, MAX_PIXELS) == pytest.approx(2.0)

    def test_enlarging_stops_at_the_pixel_limit(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=inference.__name__):
            scale = _scale_to_model(2.0, 0.5, 4_000_000, 16_000_000)  # wants 4x, may only grow 2x per side
        assert scale == pytest.approx(2.0)
        assert "exceed SEGMENTATION_MAX_PIXELS" in caplog.text

    def test_image_already_at_the_limit_is_not_enlarged(self) -> None:
        assert _scale_to_model(1.0, 0.5, MAX_PIXELS * 2, MAX_PIXELS) == 1.0

    def test_shrinking_is_never_limited(self) -> None:
        assert _scale_to_model(0.1, 0.5, MAX_PIXELS * 4, MAX_PIXELS) == pytest.approx(0.2)


class TestHeadStructure:
    def save(self, tmp_path: Path, weights: dict[str, torch.Tensor]) -> str:
        path = tmp_path / "weights.pt"
        torch.save(weights, path)
        return str(path)

    def test_unet_head_layers_and_class_count(self, tmp_path: Path) -> None:
        path = self.save(
            tmp_path,
            {
                "out_class.encoder_blocks.0.block.conv1.weight": torch.zeros(32, 3, 3, 3),
                "out_class.encoder_blocks.1.block.conv1.weight": torch.zeros(64, 32, 3, 3),
                "out_class.encoder_blocks.1.block.conv2.weight": torch.zeros(99, 1),  # not a head layer size
                "W3": torch.zeros(1, 6),
            },
        )
        assert _head_structure(path) == ([32, 64], 6)

    def test_single_convolution_head(self, tmp_path: Path) -> None:
        assert _head_structure(self.save(tmp_path, {"W3": torch.zeros(1, 4)})) == (None, 4)


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class TestLoadModel:
    @pytest.fixture(autouse=True)
    def fresh_cache(self) -> Any:
        inference._load_model.cache_clear()
        yield
        inference._load_model.cache_clear()

    def test_needs_a_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="No CUDA GPU"):
            inference._load_model("consep", "bf16")

    def test_unknown_model_lists_the_known_ones(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(inference, "DEFAULT_MODEL_CONFIGS", {"b": {}, "a": {}})
        with pytest.raises(RuntimeError, match="'nope'; it must be one of: a, b"):
            inference._load_model("nope", "bf16")

    def test_weights_must_match_the_configured_cell_types(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / "w.pt"
        torch.save({"W3": torch.zeros(1, 3)}, path)  # background + 2 classes
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            inference, "DEFAULT_MODEL_CONFIGS", {"m": {"path": str(path), "mpp": 0.5, "cell_types": ["a", "b", "c"]}}
        )
        monkeypatch.setattr(inference, "ModelConfig", _Config)
        with pytest.raises(RuntimeError, match="predict 2 classes, but its configuration names 3"):
            inference._load_model("m", "bf16")

    def test_loads_the_model_once(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        path = tmp_path / "w.pt"
        torch.save({"W3": torch.zeros(1, 3)}, path)
        created: list[dict[str, Any]] = []

        class Model:
            def __init__(self, **kwargs: Any) -> None:
                created.append(kwargs)
                self.net = SimpleNamespace(eval=lambda: None)

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            inference, "DEFAULT_MODEL_CONFIGS", {"m": {"path": str(path), "mpp": 0.5, "cell_types": ["a", "b"]}}
        )
        monkeypatch.setattr(inference, "ModelConfig", _Config)
        monkeypatch.setattr(inference, "ClassposeModel", Model)
        first = inference._load_model("m", "fp16")
        assert inference._load_model("m", "fp16") is first
        assert len(created) == 1
        assert created[0]["nclasses"] == 3 and created[0]["precision"] == "fp16"
        assert created[0]["feature_transformation_structure"] is None


class _Config:
    def __init__(self, path: str, mpp: float, cell_types: list[str]) -> None:
        self.path, self.mpp, self.cell_types = path, mpp, cell_types

    def download_if_necessary(self) -> None:
        pass


class FakeModel:
    """Labels the left half of the image cell 1 (class 2), a block on the right cell 2 (background class).

    Its cell-probability logit is 2 in cell 1 and 0 elsewhere; class 2 has logit ln 3 in cell 1 and the other classes
    0, so cell 1's type has probability 3 / 5 there.
    """

    def __init__(self, class_logits: bool = True) -> None:
        self.seen_shapes: list[tuple[int, ...]] = []
        self.class_logits = class_logits

    def eval(self, image: np.ndarray, **_: Any) -> tuple[np.ndarray, tuple[Any, ...], np.ndarray, None]:
        self.seen_shapes.append(image.shape)
        height, width = image.shape[:2]
        masks = np.zeros((height, width), dtype=np.int32)
        classes = np.zeros((height, width), dtype=np.int32)
        masks[:, : width // 2] = 1
        classes[:, : width // 2] = 2
        masks[: height // 4, width * 3 // 4 :] = 2
        cellprob = np.where(masks == 1, 2.0, 0.0).astype(np.float32)
        logits = np.zeros((3, height, width), dtype=np.float32)
        logits[2][masks == 1] = np.log(3.0)
        flows = (None, None, cellprob, logits if self.class_logits else None, image.shape)
        return masks, flows, classes, None


class TestSegment:
    def patch(self, monkeypatch: pytest.MonkeyPatch, model: FakeModel, model_mpp: float = 0.5) -> None:
        config = SimpleNamespace(mpp=model_mpp, cell_types=["Tumor", "Lymphocyte"])
        monkeypatch.setattr(inference, "_load_model", lambda name, precision: (model, config))

    def test_labels_their_cell_types_and_confidences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = FakeModel()
        self.patch(monkeypatch, model)
        result = inference.segment(np.zeros((40, 40, 3), dtype=np.uint8), None, settings())
        assert result.masks.shape == (40, 40)
        assert result.cell_types == [None, "Lymphocyte", None]
        assert model.seen_shapes == [(40, 40, 3)]
        # sigmoid(2) = 0.8808, sigmoid(0) = 0.5; the background-class cell has no type, so no type confidence.
        assert result.outline_confidence == [None, 0.8808, 0.5]
        assert result.class_confidence == [None, 0.6, None]

    def test_image_is_resized_for_the_model_and_masks_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = FakeModel()
        self.patch(monkeypatch, model, model_mpp=0.5)
        result = inference.segment(np.zeros((40, 60, 3), dtype=np.uint8), 0.25, settings())
        assert model.seen_shapes == [(20, 30, 3)]
        assert result.masks.shape == (40, 60)
        assert set(np.unique(result.masks)) == {0, 1, 2}
        assert result.cell_types[1] == "Lymphocyte"
        # Measured on the model's own output, before the masks are resized back.
        assert result.outline_confidence == [None, 0.8808, 0.5]
        assert result.class_confidence == [None, 0.6, None]

    def test_without_class_probabilities_the_cells_still_come_back(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        self.patch(monkeypatch, FakeModel(class_logits=False))
        with caplog.at_level(logging.WARNING, logger=inference.__name__):
            result = inference.segment(np.zeros((40, 40, 3), dtype=np.uint8), None, settings())
        assert result.cell_types == [None, "Lymphocyte", None]
        assert result.outline_confidence == [None, 0.8808, 0.5]
        assert result.class_confidence == [None, None, None]
        assert "no usable class probabilities" in caplog.text


def test_model_version_names_the_package_and_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference, "version", lambda package: "1.2.3")
    assert inference.model_version(settings(CLASSPOSE_MODEL="consep")) == "classpose-1.2.3/consep"
