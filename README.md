# ml-worker

Runs [Classpose](https://github.com/sohmandal/classpose) on every hotspot and returns the cell outlines, each with a
predicted cell type, to the backend. The model only provides a starting point: its cells are saved as annotations that
annotators edit like any other.

Classpose and its weights are licensed CC BY-NC 4.0: non-commercial use only.

This stack is **optional**. Without it hotspots stay `pending` and the rest of the app works as usual.

## Running it

Needs Docker with GPU access (on Windows: Docker Desktop on WSL2 with a current NVIDIA driver). This stack is the
worker alone: the broker and the bucket belong to the backend it serves. There are two workers, one per backend, each
with its own settings file (both gitignored):

| Worker | Serves | Settings | Start |
|---|---|---|---|
| `worker-local` | The backend's local stack (`backend/docker-compose.yml`: Redis, S3Mock) | `.env.local`: `cp .env.example .env.local`, works as is | `docker compose --profile local up -d --build` |
| `worker-aws` | The deployed app (SQS, the real bucket) | `.env.aws`: `terraform output -raw ml_worker_env` in `backend/infra`, plus optional settings such as `LOG_LEVEL` | `docker compose --profile aws up -d --build` |

Both profiles at once share the GPU and the downloaded weights. The first build downloads PyTorch, and the first start
the model weights.

For the local one, also set `SEGMENTATION_ENABLED=true` in `backend/.env.local` and start the backend with
`docker compose --profile segmentation up -d --build`. The deployed backend always has it on. The AWS worker only dials
out (SQS and S3 in `eu-north-1`), so its PC needs no open ports and can sit on any network.

The backend and the worker can start in any order. A worker without a usable GPU exits instead of taking jobs, and
`docker compose logs -f worker-local` (or `worker-aws`) shows why.

## How it works

```text
backend API ---- segmentation-jobs ----> worker (this repo)
                                           | reads the image, writes hotspots/{id}/segmentation.json to S3
backend consumer <-- segmentation-results -+
   | saves the cells to the database
```

The worker has no database access. Several GPU machines can serve the same backend at once: they take jobs from
the same queue.

## The contract with the backend

`backend` and `ml-worker` are separate repositories and nothing checks that they match. It is defined in
`segmentation/contract.py` here and at the top of `backend/app/services/segmentation.py`. Change both together, and
raise `SCHEMA_VERSION` when a format changes.

| Item | Value |
|---|---|
| Queues | `segmentation-jobs`, `segmentation-results` |
| `segmentation.run` | `schema_version`, `hotspot_id`, `image_key`, `mpp` (microns per pixel of the image, or null), and `rescore` (true when the backend only wants the confidences of a hotspot segmented before they existed; the run is the same) |
| `segmentation.save_result` | `schema_version`, `hotspot_id`, `status` (`done` or `failed`), then `model_version` and `result_key`, or `error` |
| `hotspots/{id}/segmentation.json` | `schema_version`, `hotspot_id`, `model_version`, `image_width`, `image_height`, `cells` (per cell: `ring`, one open ring of `[x, y]` image pixels, `cell_type`, the model's name for it or null, and `outline_confidence` / `class_confidence`, the mean probability of the cell's pixels and of its type over them, each null when unknown) |

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `CELERY_BROKER_URL`, `CELERY_BROKER_TRANSPORT_OPTIONS` | Redis of this stack | The broker; see below |
| `BUCKET_*` | required | The backend's bucket |
| `CLASSPOSE_MODEL` | `consep` | Which pretrained model, and so which cell types; see below |
| `CLASSPOSE_PRECISION` | `bf16` | `bf16` (falls back to `fp16` on GPUs without it), `fp16` or `fp32` |
| `CLASSPOSE_BATCH_SIZE` | `8` | Lower it if the GPU runs out of memory |
| `CLASSPOSE_FLOW_THRESHOLD`, `CLASSPOSE_CELLPROB_THRESHOLD` | `0.4`, `0.0` | Model parameters |
| `SEGMENTATION_MAX_PIXELS` | `16777216` (4096²) | Larger images are reported as failed |
| `POLYGON_SIMPLIFY_PX` | `1.0` | How far a simplified outline may stray from the mask |

The size limit is there because of the time and the number of cells to edit, not the memory.

## Models and cell types

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

The models differ in how many cells they find, not only in how they name them. On three of our hotspots (a nearly empty
one, pale adipose tissue, dense tissue), `consep` found 9, 43 and 630 cells, `puma` 5, 39 and 592, `nucls` 6, 35 and
507, `conic` 0, 16 and 454, and `monusac` 0, 13 and 272. Hence the default.

The weights are downloaded from Hugging Face on the first start. The backend matches the names to its cell types,
ignoring case, and creates the missing ones; a cell the model gave no type is saved as `Unclassified`.

When the job carries the image's `mpp`, the image is resized to the model's resolution for inference and the outlines
are scaled back. An image is never enlarged beyond `SEGMENTATION_MAX_PIXELS`. Without `mpp` it is used as it is.

## Brokers

Everything used here works on both Redis (local) and SQS (deployed), so switching is configuration only. The deployed
queues, their dead-letter queues and the worker's permissions come from Terraform in `backend/infra`, and so does the
`.env.aws` for them. For reference, it has this shape:

```dotenv
CELERY_BROKER_URL=sqs://
CELERY_BROKER_TRANSPORT_OPTIONS={"region": "eu-north-1", "visibility_timeout": 3600, "wait_time_seconds": 20, "predefined_queues": {"segmentation-jobs": {"url": "https://sqs.eu-north-1.amazonaws.com/<account>/segmentation-jobs"}, "segmentation-results": {"url": "https://sqs.eu-north-1.amazonaws.com/<account>/segmentation-results"}}}
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Keep to the rules in `segmentation/celery_app.py`. Messages stay small (ids and S3 keys), because SQS limits their size
and Redis does not, so an oversized message would only fail on SQS.

## Development

```bash
python scripts/fetch_classpose.py  # Classpose's source, into vendor/; needed before the first `uv sync`
uv sync
uv run --with pytest python -m pytest tests
```
