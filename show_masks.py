import matplotlib.pyplot as plt
import tifffile
import numpy as np

IMAGE_PATH = "data/image.jpg"
MASK_PATH = "data/hotspot_masks.tif"
OUTPUT_OVERLAY_PATH = "data/hotspot_masks_overlay.png"
ALPHA = 0.5
DPI = 300
FIGSIZE = (12, 12)
COLORMAP = "nipy_spectral"

def visualize_masks() -> None:
    img = plt.imread(IMAGE_PATH)
    masks = tifffile.imread(MASK_PATH)

    plt.figure(figsize=FIGSIZE)
    plt.imshow(img)

    masked_masks = np.ma.masked_where(masks == 0, masks)

    plt.imshow(masked_masks, cmap=COLORMAP, alpha=ALPHA, interpolation='nearest')

    plt.title(f"Maska nałożona na oryginał - znaleziono {masks.max()} obiektów")
    plt.axis('off')
    plt.savefig(OUTPUT_OVERLAY_PATH, bbox_inches='tight', pad_inches=0, dpi=DPI)
    print(f"Zapisano podgląd do: {OUTPUT_OVERLAY_PATH}")

    plt.show()

if __name__ == "__main__":
    visualize_masks()
