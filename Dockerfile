FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"
# On a volume, so the weights are downloaded once and not on every rebuild.
ENV CELLPOSE_LOCAL_MODELS_PATH="/models"

# The CUDA packages take gigabytes: with uv's defaults, small downloads time out while competing with them.
ENV UV_HTTP_TIMEOUT="600"
ENV UV_CONCURRENT_DOWNLOADS="4"

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY segmentation ./segmentation

# --pool=solo: one GPU runs one job. Gossip, mingle and heartbeat are worker-to-worker features SQS does not support.
CMD ["celery", "-A", "segmentation.celery_app", "worker", "-Q", "segmentation-jobs", "--pool=solo", "--concurrency=1", "--without-gossip", "--without-mingle", "--without-heartbeat", "--loglevel=INFO"]
