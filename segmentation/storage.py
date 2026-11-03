import io

import boto3
import numpy as np
import numpy.typing as npt
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image

from segmentation.config import Settings
from segmentation.errors import SegmentationError

IMAGE_TOO_LARGE = "The image has {pixels:,} pixels; segmentation handles at most {limit:,}."
IMAGE_MISSING = "The hotspot image no longer exists."
MISSING_OBJECT_CODES = {"NoSuchKey", "404"}


class Storage:
    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.BUCKET_NAME
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.BUCKET_ENDPOINT,
            aws_access_key_id=settings.BUCKET_ACCESS_KEY,
            aws_secret_access_key=settings.BUCKET_SECRET_KEY,
            region_name=settings.BUCKET_REGION,
            config=Config(signature_version="s3v4"),
        )

    def load_rgb(self, key: str, max_pixels: int) -> npt.NDArray[np.uint8]:
        try:
            data = self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        except ClientError as exc:
            # A deleted hotspot is final; any other storage error may be temporary, so it propagates.
            if exc.response.get("Error", {}).get("Code") in MISSING_OBJECT_CODES:
                raise SegmentationError(IMAGE_MISSING) from exc
            raise
        with Image.open(io.BytesIO(data)) as image:
            # The size is known before decoding, so an oversized image costs no memory.
            pixels = image.width * image.height
            if pixels > max_pixels:
                raise SegmentationError(IMAGE_TOO_LARGE.format(pixels=pixels, limit=max_pixels))
            return np.asarray(image.convert("RGB"))

    def put_json(self, key: str, body: str) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=body.encode(), ContentType="application/json")
