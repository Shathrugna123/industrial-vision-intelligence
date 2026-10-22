from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"


def main():
    category_dir = RAW_DIR / "pcb1"
    normal_dir = category_dir / "Data" / "Images" / "Normal"
    anomaly_dir = category_dir / "Data" / "Images" / "Anomaly"

    normal_images = sorted(normal_dir.glob("*.JPG"))[:4]
    anomaly_images = sorted(anomaly_dir.glob("*.JPG"))[:4]

    samples = [
        (path, "Normal") for path in normal_images
    ] + [
        (path, "Anomaly") for path in anomaly_images
    ]

    if not samples:
        raise FileNotFoundError(
            f"No sample images found in {category_dir}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for ax, (path, label) in zip(axes.flat, samples):
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{label}: {path.name}")
        ax.axis("off")

    fig.tight_layout()
    output_path = OUTPUT_DIR / "dataset_sample_grid.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Sample visualization saved to: {output_path}")


if __name__ == "__main__":
    main()