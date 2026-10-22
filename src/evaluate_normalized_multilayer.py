
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from mvtec_dataset import MVTecDataset
from normalized_multilayer_feature_extractor import (
    NormalizedMultiLayerFeatureExtractor,
)
from patchcore_baseline import build_memory_bank, score_feature_maps


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["grid", "tile", "transistor"]
BATCH_SIZE = 4
MAX_MEMORY_FEATURES = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_features(model, loader):
    feature_batches = []
    labels = []
    masks = []

    for batch in loader:
        images = batch["image"].to(DEVICE)

        with torch.no_grad():
            features = model(images)

        feature_batches.append(features.cpu())
        labels.extend(batch["label"].cpu().tolist())
        masks.extend(batch["mask"].cpu())

    return (
        torch.cat(feature_batches, dim=0),
        np.asarray(labels),
        torch.stack(masks).numpy(),
    )


def main():
    model = NormalizedMultiLayerFeatureExtractor()
    results = {}

    for category in CATEGORIES:
        print(f"\nEvaluating normalized multi-layer: {category}")

        train_dataset = MVTecDataset(category, split="train")
        test_dataset = MVTecDataset(category, split="test")

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        train_features, _, _ = collect_features(model, train_loader)
        test_features, labels, masks = collect_features(model, test_loader)

        memory_bank = build_memory_bank(
            train_features,
            max_features=MAX_MEMORY_FEATURES,
        )

        anomaly_maps = score_feature_maps(test_features, memory_bank)

        # Image score: 95th percentile of the anomaly map.
        image_scores = np.percentile(
            anomaly_maps.reshape(len(anomaly_maps), -1),
            95,
            axis=1,
        )

        image_auc = roc_auc_score(labels, image_scores)

        # Evaluate pixel-level localization only on test images with defects.
        # Normal images have all-zero masks and are included as background pixels.
        pixel_auc = roc_auc_score(
            masks.reshape(-1).astype(np.uint8),
            anomaly_maps.reshape(-1),
        )

        results[category] = {
            "image_auroc_p95": float(image_auc),
            "pixel_auroc": float(pixel_auc),
            "test_images": int(len(labels)),
            "defective_images": int(labels.sum()),
            "memory_bank_features": int(len(memory_bank)),
        }

        print(f"Image AUROC (P95): {image_auc:.4f}")
        print(f"Pixel AUROC:       {pixel_auc:.4f}")

    report_path = REPORT_DIR / "normalized_multilayer_targeted_evaluation.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
