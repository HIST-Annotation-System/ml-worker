import io
from typing import Any

import numpy as np
import pytest
from botocore.exceptions import ClientError
from PIL import Image

from segmentation.config import Settings
from segmentation.errors import SegmentationError
from segmentation.storage import Storage


class FakeBody:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.error_code: str | None = None
        self.puts: list[dict[str, Any]] = []

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803 - boto3's names
        if self.error_code:
            raise ClientError({"Error": {"Code": self.error_code}}, "GetObject")
        return {"Body": FakeBody(self.objects[Key])}

    def put_object(self, **kwargs: Any) -> None:
        self.puts.append(kwargs)


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def storage(s3: FakeS3, monkeypatch: pytest.MonkeyPatch) -> Storage:
    monkeypatch.setattr("segmentation.storage.boto3.client", lambda *args, **kwargs: s3)
    return Storage(Settings(_env_file=None))  # type: ignore[call-arg]


def encoded(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_any_image_mode_becomes_rgb(storage: Storage, s3: FakeS3, mode: str) -> None:
    s3.objects["k"] = encoded(Image.new(mode, (8, 6)))
    array = storage.load_rgb("k", 1000)
    assert array.shape == (6, 8, 3)
    assert array.dtype == np.uint8


def test_pixels_keep_their_values(storage: Storage, s3: FakeS3) -> None:
    s3.objects["k"] = encoded(Image.new("RGB", (2, 2), (10, 20, 30)))
    assert storage.load_rgb("k", 100)[0, 0].tolist() == [10, 20, 30]


def test_oversized_image_is_refused(storage: Storage, s3: FakeS3) -> None:
    s3.objects["k"] = encoded(Image.new("RGB", (100, 100)))
    with pytest.raises(SegmentationError, match="10,000 pixels; segmentation handles at most 9,999"):
        storage.load_rgb("k", 9_999)


@pytest.mark.parametrize("code", ["NoSuchKey", "404"])
def test_missing_image_is_a_permanent_failure(storage: Storage, s3: FakeS3, code: str) -> None:
    s3.error_code = code
    with pytest.raises(SegmentationError, match="no longer exists"):
        storage.load_rgb("k", 100)


@pytest.mark.parametrize("code", ["AccessDenied", "SlowDown", "InternalError"])
def test_other_storage_errors_propagate(storage: Storage, s3: FakeS3, code: str) -> None:
    s3.error_code = code
    with pytest.raises(ClientError):
        storage.load_rgb("k", 100)


def test_put_json(storage: Storage, s3: FakeS3) -> None:
    storage.put_json("r.json", '{"a": "ż"}')
    assert s3.puts == [
        {"Bucket": "bucket", "Key": "r.json", "Body": '{"a": "ż"}'.encode(), "ContentType": "application/json"}
    ]
