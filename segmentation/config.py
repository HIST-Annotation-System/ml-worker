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

    CELLPOSE_MODEL: str = "cpsam_v2"
    CELLPOSE_BATCH_SIZE: int = 8
    CELLPOSE_FLOW_THRESHOLD: float = 0.4
    CELLPOSE_CELLPROB_THRESHOLD: float = 0.0
    # Larger images are reported as failed instead of risking an out-of-memory crash.
    SEGMENTATION_MAX_PIXELS: int = 16_777_216
    # How far (px) a simplified outline may stray from the mask: raw contours are too dense to edit by hand.
    POLYGON_SIMPLIFY_PX: float = 1.0

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
