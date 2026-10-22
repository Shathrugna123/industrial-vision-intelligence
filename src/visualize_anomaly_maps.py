
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from mvtec_dataset import MVTecDataset
from feature_extractor import FeatureExtractor, DEVICE
from patchcore_baseline import (
    extract_features,
    build_memory_bank,
    score_feature_maps,
    CATEGORY,
    BATCH_SIZE,
    MAX_MEMORY_FEATURES,
)

from torch.utils.data import DataLoader
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_original_image(image_path):
    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = MVTecDataset(CATEGORY, split="train")
    test_dataset = MVTecDataset(CATEGORY, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = FeatureExtractor(pretrained=True).to(DEVICE)
    model.eval()

    print("Extracting normal training features...")
    train_features, _, _ = extract_features(model, train_loader)

    memory_bank = build_memory_bank(
        train_features, MAX_MEMORY_FEATURES
    )

    print("Extracting test features...")
    test_features, labels, masks = extract_features(model, test_loader)

    anomaly_maps = score_feature_maps(test_features, memory_bank)

    # Pick the first defective test sample.
    anomaly_indices = np.flatnonzero(labels == 1)

    if len(anomaly_indices) == 0:
        raise RuntimeError("No anomalous test images found.")

    index = int(anomaly_indices[0])
    sample = test_dataset[index]

    original = load_original_image(sample["image_path"])
    original = original.resize((224, 224))

    ground_truth = masks[index, 0].numpy()
    anomaly_map = anomaly_maps[index].numpy()

    # Normalize only for visualization, not for evaluation.
    low = float(anomaly_map.min())
    high = float(anomaly_map.max())

    if high > low:
        heatmap = (anomaly_map - low) / (high - low)
    else:
        heatmap = np.zeros_like(anomaly_map)

    original_array = np.asarray(original)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(original_array)
    axes[0].set_title("Original image")

    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-truth mask")

    axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Predicted anomaly heatmap")

    axes[3].imshow(original_array)
    axes[3].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[3].set_title("Anomaly overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"MVTec AD | {CATEGORY} | {sample['defect_type']}"
    )
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{CATEGORY}_anomaly_visualization.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Category: {CATEGORY}")
    print(f"Defect type: {sample['defect_type']}")
    print(f"Ground-truth defect pixels: {int(ground_truth.sum())}")
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()
