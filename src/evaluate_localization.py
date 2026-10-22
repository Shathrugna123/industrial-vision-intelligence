
from pathlib import Path
import gc
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from mvtec_dataset import MVTecDataset, DATASET_ROOT
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


def calculate_pixel_auroc(anomaly_maps, masks):
    """Calculate pixel-level AUROC from anomaly maps and binary masks."""
    scores = anomaly_maps.detach().cpu().numpy().reshape(-1)
    targets = masks.detach().cpu().numpy().reshape(-1).astype(np.uint8)

    if len(np.unique(targets)) < 2:
        return None

    return float(roc_auc_score(targets, scores))


def evaluate_category(category, model):
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

    # Build the normal-only reference memory bank.
    memory_bank = build_memory_bank(
        train_features,
        MAX_MEMORY_FEATURES,
    )

    # Ground-truth masks come from the test dataset.
    masks = []
    for batch in tqdm(test_loader, desc=f"Masks: {category}"):
        masks.append(batch["mask"])

    masks = torch.cat(masks, dim=0)

    # Generate anomaly maps with the existing baseline implementation.
    anomaly_maps = score_feature_maps(test_features, memory_bank)

    if anomaly_maps.ndim == 3:
        anomaly_maps = anomaly_maps.unsqueeze(1)

    if masks.ndim == 3:
        masks = masks.unsqueeze(1)

    if anomaly_maps.shape[-2:] != masks.shape[-2:]:
        anomaly_maps = torch.nn.functional.interpolate(
            anomaly_maps.float(),
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    pixel_auroc = calculate_pixel_auroc(anomaly_maps, masks)

    result = {
        "pixel_auroc": pixel_auroc,
        "test_images": len(test_dataset),
        "mask_pixels": int(masks.numel()),
    }

    del train_features, test_features, memory_bank
    del anomaly_maps, masks
    del train_dataset, test_dataset, train_loader, test_loader
    gc.collect()

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    categories = sorted(
        path.name for path in DATASET_ROOT.iterdir()
        if path.is_dir()
    )

    model = FeatureExtractor(pretrained=True).to(DEVICE)
    model.eval()

    report = {
        "experiment": "Pixel-level anomaly localization evaluation",
        "categories": {},
        "failures": [],
    }

    for category in categories:
        print(f"\nEvaluating localization: {category}")

        try:
            result = evaluate_category(category, model)
            report["categories"][category] = result

            print(f"Pixel AUROC: {result['pixel_auroc']}")

        except Exception as exc:
            report["failures"].append({
                "category": category,
                "error": str(exc),
            })
            print(f"FAILED: {exc}")
            gc.collect()

    values = [
        result["pixel_auroc"]
        for result in report["categories"].values()
        if result["pixel_auroc"] is not None
    ]

    report["macro_average_pixel_auroc"] = (
        float(np.mean(values)) if values else None
    )
    report["categories_evaluated"] = len(values)

    output_path = OUTPUT_DIR / "localization_evaluation.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nLOCALIZATION EVALUATION SUMMARY")
    print(f"Categories evaluated: {len(values)}")
    print(f"Mean pixel AUROC: {report['macro_average_pixel_auroc']}")
    print(f"Failed categories: {len(report['failures'])}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
