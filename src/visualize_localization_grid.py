
from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from mvtec_dataset import MVTecDataset
from feature_extractor import FeatureExtractor, DEVICE
from patchcore_baseline import (
    extract_features,
    build_memory_bank,
    score_feature_maps,
    BATCH_SIZE,
    MAX_MEMORY_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

# Include strong and weak categories for visual comparison.
CATEGORIES = ["bottle", "grid", "tile", "transistor"]
SAMPLES_PER_CATEGORY = 2


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = FeatureExtractor(pretrained=True).to(DEVICE)
    model.eval()

    # One row per category/sample, four columns per row.
    rows = []

    for category in CATEGORIES:
        print(f"\nProcessing {category}...")

        train_dataset = MVTecDataset(category, split="train")
        test_dataset = MVTecDataset(category, split="test")

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

        train_features, _, _ = extract_features(model, train_loader)
        test_features, _, _ = extract_features(model, test_loader)

        memory_bank = build_memory_bank(
            train_features,
            MAX_MEMORY_FEATURES,
        )
        anomaly_maps = score_feature_maps(test_features, memory_bank)

        # Keep dataset order aligned with the extracted feature order.
        chosen_indices = []
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            is_anomaly = int(sample["label"]) == 1

            # Select one normal and one anomalous example when available.
            if is_anomaly not in [
                int(test_dataset[i]["label"]) == 1
                for i in chosen_indices
            ]:
                chosen_indices.append(idx)

            if len(chosen_indices) >= SAMPLES_PER_CATEGORY:
                break

        for idx in chosen_indices:
            sample = test_dataset[idx]

            image = sample["image"].detach().cpu()
            mask = sample["mask"].detach().cpu().squeeze().numpy()
            heatmap = anomaly_maps[idx].detach().cpu()

            if heatmap.ndim == 3:
                heatmap = heatmap.squeeze(0)

            heatmap = heatmap.numpy()

            # Convert normalized tensor image to displayable RGB.
            image = image.permute(1, 2, 0).numpy()
            image = np.clip(image, 0, 1)

            rows.append({
                "category": category,
                "label": "Anomaly" if int(sample["label"]) == 1 else "Normal",
                "image": image,
                "mask": mask,
                "heatmap": heatmap,
            })

        del train_features, test_features, memory_bank, anomaly_maps
        del train_dataset, test_dataset, train_loader, test_loader
        gc.collect()

    fig, axes = plt.subplots(
        len(rows), 4,
        figsize=(14, 3.2 * len(rows)),
        squeeze=False,
    )

    column_titles = [
        "Original Image",
        "Ground-Truth Mask",
        "Predicted Anomaly Heatmap",
        "Heatmap Overlay",
    ]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=11)

    for row_idx, item in enumerate(rows):
        image = item["image"]
        mask = item["mask"]
        heatmap = item["heatmap"]

        axes[row_idx, 0].imshow(image)
        axes[row_idx, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].imshow(heatmap, cmap="jet")
        axes[row_idx, 3].imshow(image)
        axes[row_idx, 3].imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            vmin=np.min(heatmap),
            vmax=np.max(heatmap),
        )

        label = f"{item['category']} — {item['label']}"
        axes[row_idx, 0].set_ylabel(label, fontsize=10)

        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    output_path = OUTPUT_DIR / "localization_comparison_grid.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved visualization to: {output_path}")


if __name__ == "__main__":
    main()
