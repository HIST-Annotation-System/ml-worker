# HistAno ml-worker

The GPU worker of HistAno. It runs [Classpose](https://github.com/sohmandal/classpose) on every hotspot and returns
the cell outlines, each with a predicted cell type and the model's confidence, to the backend. The model only
provides a starting point: its cells are saved as annotations that experts edit like any other.

It is **optional**: without a worker, hotspots stay `pending` and the rest of the app works as usual.

Classpose and its weights are licensed CC BY-NC 4.0: non-commercial use only (see [License](#license)).

## HistAno

HistAno is a system for annotating cells in histopathology slides: a model outlines and classifies the cells of each
hotspot first, and experts correct its work instead of drawing every cell from scratch. It is split into three
repositories:

| Repository | What it is |
|---|---|
| [backend](https://github.com/HIST-Annotation-System/backend) | The API, its database and storage; the AWS infrastructure (`infra/`) |
| [frontend](https://github.com/HIST-Annotation-System/frontend) | The React web app |
| [ml-worker](https://github.com/HIST-Annotation-System/ml-worker) | This worker |

Clone them side by side in one folder (`backend/`, `frontend/`, `ml-worker/`). The backend's README describes the
architecture.

## Quick start

Requirements: Docker with access to an NVIDIA GPU (on Windows: Docker Desktop on WSL2 with a current NVIDIA driver).
A worker without a usable GPU exits instead of taking jobs.

This repository is the worker alone: the broker and the bucket belong to the backend it serves. There are two workers,
one per backend, each with its own settings file (both gitignored):

| Worker | Serves | Settings | Start |
|---|---|---|---|
| `worker-local` | The backend's local stack (Redis, S3Mock) | `.env.local`: `cp .env.example .env.local`, works as is | `docker compose --profile local up -d --build` |
| `worker-aws` | The deployed app (SQS, the real bucket) | `.env.aws`: `terraform output -raw ml_worker_env` in `backend/infra`, plus any optional settings | `docker compose --profile aws up -d --build` |

The first build downloads PyTorch (several GB), and the first start the model weights (kept in the `model_cache`
volume). Both profiles can run at once; they share the GPU and the weights.

For the local worker, the backend must send jobs: set `SEGMENTATION_ENABLED=true` in `backend/.env.local` and start
the backend with `docker compose --profile segmentation up -d --build`. The deployed backend always sends them. The
backend and the worker can start in any order.

The AWS worker only dials out (SQS and S3 in `eu-north-1`), so its PC needs no open port and can sit on any network.
`.env.aws` holds an AWS access key: never commit or share it.

### Is it working?

```bash
docker compose logs -f worker-aws      # or worker-local
```

A healthy worker logs `Connected to sqs://` (or `redis://`) and `celery@... ready`, then one line per job:
`Hotspot <id>: <n> cells`. A missing GPU, a model that fails to load, or wrong credentials show up here at startup.
`LOG_LEVEL=DEBUG` in the settings file gives more detail. In the deployed app, jobs that failed five times end up in
the `segmentation-jobs-dead-letter` queue (SQS console).

## How it works

```text
backend API ---- segmentation-jobs ----> worker (this repo)
                                           | reads the hotspot image, writes hotspots/{id}/segmentation.json to S3
backend consumer <-- segmentation-results -+
   | saves the cells to the database
```

The worker has no database access. It takes one job at a time and acknowledges it only when done, so a job of a
worker that dies is delivered again. Several GPU machines can serve the same backend at once: they take jobs from the
same queue.

When the job carries the image's `mpp` (microns per pixel), the image is resized to the model's resolution for
inference and the outlines are scaled back; it is never enlarged beyond `SEGMENTATION_MAX_PIXELS`. Without `mpp` it is
used as it is. Outlines are simplified (`POLYGON_SIMPLIFY_PX`) so they stay editable by hand.

### The contract with the backend

`backend` and `ml-worker` are separate repositories and nothing checks that they match. The contract is defined in
`segmentation/contract.py` here and at the top of `backend/app/services/segmentation.py`. Change both together, and
raise `SCHEMA_VERSION` when a format changes.

| Item | Value |
|---|---|
| Queues | `segmentation-jobs`, `segmentation-results` |
| `segmentation.run` | `schema_version`, `hotspot_id`, `image_key`, `mpp` (microns per pixel of the image, or null), and `rescore` (true when the backend only wants the confidences of a hotspot segmented before they existed; the run is the same) |
| `segmentation.save_result` | `schema_version`, `hotspot_id`, `status` (`done` or `failed`), then `model_version` and `result_key`, or `error` |
| `hotspots/{id}/segmentation.json` | `schema_version`, `hotspot_id`, `model_version`, `image_width`, `image_height`, `cells` (per cell: `ring`, one open ring of `[x, y]` image pixels, `cell_type`, the model's name for it or null, and `outline_confidence` / `class_confidence`, the mean probability of the cell's pixels and of its type over them, each null when unknown) |

### Brokers

Everything used here works on both Redis (local) and SQS (deployed), so switching is configuration only. Keep to the
rules in `segmentation/celery_app.py` (no result backend, no remote control, no events, no ETA tasks), and keep
messages small (ids and S3 keys): SQS limits their size and Redis does not, so an oversized message would only fail on
SQS. The deployed queues, their dead-letter queues and the worker's permissions come from Terraform in
`backend/infra`, and so does `.env.aws`. For reference, it has this shape:

```dotenv
CELERY_BROKER_URL=sqs://
CELERY_BROKER_TRANSPORT_OPTIONS={"region": "eu-north-1", "visibility_timeout": 3600, "wait_time_seconds": 20, "predefined_queues": {"segmentation-jobs": {"url": "https://sqs.eu-north-1.amazonaws.com/<account>/segmentation-jobs"}, "segmentation-results": {"url": "https://sqs.eu-north-1.amazonaws.com/<account>/segmentation-results"}}}
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
BUCKET_ENDPOINT=https://s3.eu-north-1.amazonaws.com
BUCKET_ACCESS_KEY=...
BUCKET_SECRET_KEY=...
BUCKET_NAME=...
BUCKET_REGION=eu-north-1
```

## Configuration

Settings are read from the environment, which compose fills from `.env.local` or `.env.aws` (`segmentation/config.py`).

| Variable | Default | Purpose |
|---|---|---|
| `CELERY_BROKER_URL`, `CELERY_BROKER_TRANSPORT_OPTIONS` | set by the settings file | The backend's broker |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | | SQS credentials (`.env.aws` only) |
| `BUCKET_ENDPOINT`, `BUCKET_ACCESS_KEY`, `BUCKET_SECRET_KEY`, `BUCKET_NAME`, `BUCKET_REGION` | required | The backend's bucket |
| `CLASSPOSE_MODEL` | `consep` | Which pretrained model, and so which cell types; see below |
| `CLASSPOSE_PRECISION` | `bf16` | `bf16` (falls back to `fp16` on GPUs without it), `fp16` or `fp32` |
| `CLASSPOSE_BATCH_SIZE` | `8` | Lower it if the GPU runs out of memory |
| `CLASSPOSE_FLOW_THRESHOLD`, `CLASSPOSE_CELLPROB_THRESHOLD` | `0.4`, `0.0` | Model parameters |
| `SEGMENTATION_MAX_PIXELS` | `16777216` (4096²) | Larger images are reported as failed: they take long and give more cells than anyone edits |
| `POLYGON_SIMPLIFY_PX` | `1.0` | How far a simplified outline may stray from the mask |
| `LOG_LEVEL` | `INFO` | Celery's log level |

After changing a settings file, recreate the worker: `docker compose --profile aws up -d` (`restart` keeps the old
settings).

### Models and cell types

Each pretrained model was trained on one dataset, at one resolution, and predicts that dataset's cell types:

| `CLASSPOSE_MODEL` | Trained at | Cell types |
|---|---|---|
| `conic` | 0.5 µm/px | Neutrophil, Epithelial, Lymphocyte, Plasma cell, Eosinophil, Connective |
| `consep` | 0.25 µm/px | Other, Inflammatory, Healthy epithelial, Malignant epithelial, Stroma, Muscle |
| `glysac` | 0.25 µm/px | Other, Lymphocyte, Epithelial, Ambiguous |
| `monusac` | 0.25 µm/px | Epithelial, Lymphocyte, Macrophage, Neutrophil |
| `nucls` | 0.2 µm/px | Tumor, Stroma, Lymphocyte, Plasma cell, Macrophage, Other |
| `puma` | 0.22 µm/px | Apoptosis, Tumor, Endothelial, Stroma, Lymphocyte, Histocyte, Epithelial, Melanophage, Other |

`glysac` does not load: its weights predict fewer classes than its configuration names, so the worker refuses to guess
which name belongs to which class.

The models differ in how many cells they find, not only in how they name them. On three of our hotspots (a nearly
empty one, pale adipose tissue, dense tissue), `consep` found 9, 43 and 630 cells, `puma` 5, 39 and 592, `nucls` 6, 35
and 507, `conic` 0, 16 and 454, and `monusac` 0, 13 and 272. Hence the default.

The backend matches the names to its cell types, ignoring case, and creates the missing ones; a cell the model gave no
type is saved as `Unclassified`.

## Development

Python 3.13 and [uv](https://docs.astral.sh/uv/). Classpose is not on PyPI: `scripts/fetch_classpose.py` downloads
its source at a pinned commit into `vendor/` (gitignored), which `uv sync` installs from.

```bash
python scripts/fetch_classpose.py   # once, before the first `uv sync`
uv sync
uv run --with pytest python -m pytest tests
```

The tests need no GPU, broker or bucket. PyTorch comes from the CUDA 12.6 index (`pyproject.toml`).

```text
segmentation/
  celery_app.py   the Celery app; the broker rules
  config.py       settings
  contract.py     the message and result formats shared with the backend
  tasks.py        segmentation.run: download, segment, upload, report
  inference.py    loading Classpose, resizing to the model's resolution, confidences
  polygons.py     masks to simplified outlines
  storage.py      the bucket
  errors.py       errors reported to the backend as `failed`
scripts/fetch_classpose.py   downloads Classpose's source into vendor/
tests/                       pytest suite
```

## Contributing

- `main` is what the GPU machine runs; `dev` is the integration branch. There is no CI or automatic deployment: after
  a merge to `main`, pull and rebuild on the GPU machine (`docker compose --profile aws up -d --build`).
- Branch off `dev` (`feat/<topic>`, `bugfix/<topic>`), run the tests, and open a pull request into `dev`.
- Commit messages: `<type>: <summary>`, with `feat`, `bugfix`, `perf`, `test`, `style`, `docs` or `infra`.
- A change to the contract needs the matching change in the backend, released together.

## License

The code in this repository is [MIT](LICENSE). It depends on [Classpose](https://github.com/sohmandal/classpose),
which is not part of the repository: `scripts/fetch_classpose.py` downloads its source at build time, and the worker
downloads its weights on first start. Both are licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), so the built worker, including its Docker image, may
only be used for non-commercial purposes.
