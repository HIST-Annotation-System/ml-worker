from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Moving to SQS only changes these two (plus AWS credentials); see README.md.
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_BROKER_TRANSPORT_OPTIONS: dict[str, Any] = {"visibility_timeout": 3600}

    BUCKET_ENDPOINT: str
    BUCKET_ACCESS_KEY: str
    BUCKET_SECRET_KEY: str
    BUCKET_NAME: str
    BUCKET_REGION: str

    # One of Classpose's pretrained models; it decides which cell types are predicted (see README.md).
    CLASSPOSE_MODEL: str = "consep"
    # "bf16" (falls back to "fp16" on GPUs without it), "fp16" or "fp32".
    CLASSPOSE_PRECISION: str = "bf16"
    CLASSPOSE_BATCH_SIZE: int = 8
    CLASSPOSE_FLOW_THRESHOLD: float = 0.4
    CLASSPOSE_CELLPROB_THRESHOLD: float = 0.0
    # Larger images are reported as failed instead of risking an out-of-memory crash.
    SEGMENTATION_MAX_PIXELS: int = 16_777_216
    # How far (px) a simplified outline may stray from the mask: raw contours are too dense to edit by hand.
    POLYGON_SIMPLIFY_PX: float = 1.0

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
