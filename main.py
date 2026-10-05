"""
Main script for running Cellpose inference on hotspot images.
"""
import time
import numpy as np
import torch
from cellpose import models, io

IMAGE_PATH = "data/image.jpg"
OUTPUT_MASK_PATH = "data/hotspot_masks.tif"
BATCH_SIZE = 8
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0

def run_inference() -> None:
    """Runs the inference using the Cellpose model on the configured image."""
    assert torch.cuda.is_available(), "Brak CUDA"
    print("GPU:", torch.cuda.get_device_name(0))

    model = models.CellposeModel(gpu=True)
    img = io.imread(IMAGE_PATH)
    model.eval(img)

    t = time.time()
    masks, flows, styles = model.eval(
        img,
        batch_size=BATCH_SIZE,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
    )
    torch.cuda.synchronize()
    print(f"Inferencja: {time.time() - t:.2f}s")

    print("Liczba komórek:", masks.max())
    print("Shape:", masks.shape, "dtype:", masks.dtype)
    print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    io.imsave(OUTPUT_MASK_PATH, masks)
    print(f"Zapisano maski do: {OUTPUT_MASK_PATH}")

if __name__ == "__main__":
    run_inference()