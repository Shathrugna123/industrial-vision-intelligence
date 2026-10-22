
from pathlib import Path
import json
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
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


def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


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

    print(f"\nCategory: {category}")
    print(
        f"Training images: {len(train_dataset)} | "
        f"Test images: {len(test_dataset)}"
    )

    train_features, _, _ = extract_features(model, train_loader)
    test_features, test_labels, test_masks = extract_features(
        model, test_loader
    )

    memory_bank = build_memory_bank(
        train_features,
        MAX_MEMORY_FEATURES,
    )

    anomaly_maps = score_feature_maps(test_features, memory_bank)

    image_scores = anomaly_maps.flatten(1).max(dim=1).values.numpy()
    image_auc = safe_auc(test_labels, image_scores)

    pixel_scores = anomaly_maps.flatten().numpy()
    pixel_labels = test_masks.flatten().numpy().astype(np.uint8)
    pixel_auc = safe_auc(pixel_labels, pixel_scores)

    result = {
        "category": category,
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "memory_bank_features": len(memory_bank),
        "image_auroc": image_auc,
        "pixel_auroc": pixel_auc,
    }

    print(f"Image AUROC: {image_auc}")
    print(f"Pixel AUROC: {pixel_auc}")

    del train_features, test_features, memory_bank, anomaly_maps
    del train_dataset, test_dataset, train_loader, test_loader
    gc.collect()

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    categories = sorted(
        path.name for path in DATASET_ROOT.iterdir()
        if path.is_dir()
    )

    if not categories:
        raise RuntimeError(
            f"No categories found in {DATASET_ROOT}"
        )

    model = FeatureExtractor(pretrained=True).to(DEVICE)
    model.eval()

    results = []
    failures = []

    print(f"Evaluating {len(categories)} categories on {DEVICE}")

    for category in categories:
        try:
            result = evaluate_category(category, model)
            results.append(result)
        except Exception as exc:
            print(f"FAILED: {category}: {exc}")
            failures.append({
                "category": category,
                "error": str(exc),
            })
            gc.collect()

    report = {
        "method": "Simplified PatchCore-style nearest-neighbor baseline",
        "device": str(DEVICE),
        "categories_attempted": len(categories),
        "categories_completed": len(results),
        "failures": failures,
        "results": results,
    }

    output_path = OUTPUT_DIR / "mvtec_all_categories_evaluation.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 65)
    print("MULTI-CATEGORY EVALUATION SUMMARY")
    print("=" * 65)

    for result in results:
        print(
            f"{result['category']:12} | "
            f"Image AUROC: {result['image_auroc']} | "
            f"Pixel AUROC: {result['pixel_auroc']}"
        )

    print(f"\nCategories completed: {len(results)}/{len(categories)}")
    print(f"Failures: {len(failures)}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
