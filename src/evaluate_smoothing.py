
from pathlib import Path
import gc
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"

CATEGORIES = ["grid", "tile", "transistor"]
SIGMAS = [0.0, 1.0, 2.0]


def calculate_auc(labels, scores):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)

    if len(np.unique(labels)) < 2:
        return None

    return float(roc_auc_score(labels, scores))


def smooth_maps(maps, sigma):
    if sigma == 0:
        return maps

    smoothed = [
        gaussian_filter(
            anomaly_map.numpy(),
            sigma=sigma,
        )
        for anomaly_map in maps
    ]

    return torch.from_numpy(np.stack(smoothed)).float()


def evaluate_category(category, model):
    train_ds = MVTecDataset(category, split="train")
    test_ds = MVTecDataset(category, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    train_features, _, _ = extract_features(model, train_loader)
    test_features, labels, masks = extract_features(model, test_loader)

    memory_bank = build_memory_bank(
        train_features,
        MAX_MEMORY_FEATURES,
    )
    anomaly_maps = score_feature_maps(test_features, memory_bank)

    mask_array = masks.numpy().reshape(-1).astype(np.uint8)
    results = {}

    for sigma in SIGMAS:
        current_maps = smooth_maps(anomaly_maps, sigma)

        # Use the previously selected 95th-percentile image score.
        image_scores = torch.quantile(
            current_maps.flatten(1),
            0.95,
            dim=1,
        ).numpy()

        pixel_scores = current_maps.numpy().reshape(-1)

        results[str(sigma)] = {
            "image_auroc_p95": calculate_auc(labels, image_scores),
            "pixel_auroc": calculate_auc(mask_array, pixel_scores),
        }

        print(
            f"  sigma={sigma}: "
            f"image AUROC={results[str(sigma)]['image_auroc_p95']}, "
            f"pixel AUROC={results[str(sigma)]['pixel_auroc']}"
        )

    del train_features, test_features, memory_bank
    del anomaly_maps, masks, train_ds, test_ds
    del train_loader, test_loader
    gc.collect()

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = FeatureExtractor(pretrained=True).to(DEVICE)
    model.eval()

    report = {
        "experiment": "Gaussian smoothing of anomaly maps",
        "categories": {},
        "failures": [],
    }

    for category in CATEGORIES:
        print(f"\nEvaluating category: {category}")

        try:
            report["categories"][category] = evaluate_category(
                category, model
            )
        except Exception as exc:
            report["failures"].append({
                "category": category,
                "error": str(exc),
            })
            print(f"FAILED: {exc}")
            gc.collect()

    output_path = OUTPUT_DIR / "smoothing_evaluation.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved to: {output_path}")
    print(f"Failed categories: {len(report['failures'])}")


if __name__ == "__main__":
    main()
