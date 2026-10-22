
from pathlib import Path
import json
import gc

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

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

# Compare different ways to summarize an anomaly map into one image score.
SCORING_METHODS = {
    "maximum": lambda maps: maps.flatten(1).max(dim=1).values,
    "p99": lambda maps: torch.quantile(
        maps.flatten(1), 0.99, dim=1
    ),
    "p95": lambda maps: torch.quantile(
        maps.flatten(1), 0.95, dim=1
    ),
    "top_1_percent_mean": lambda maps: torch.topk(
        maps.flatten(1),
        k=max(1, int(maps.shape[-2] * maps.shape[-1] * 0.01)),
        dim=1,
    ).values.mean(dim=1),
}


def calculate_auc(labels, scores):
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

    train_features, _, _ = extract_features(model, train_loader)
    test_features, labels, _ = extract_features(model, test_loader)

    memory_bank = build_memory_bank(
        train_features,
        MAX_MEMORY_FEATURES,
    )

    anomaly_maps = score_feature_maps(test_features, memory_bank)

    results = {}

    for method_name, scoring_function in SCORING_METHODS.items():
        scores = scoring_function(anomaly_maps).cpu().numpy()

        results[method_name] = {
            "image_auroc": calculate_auc(labels, scores),
        }

    del train_features, test_features, memory_bank, anomaly_maps
    del train_dataset, test_dataset, train_loader, test_loader
    gc.collect()

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    categories = sorted(
        path.name for path in DATASET_ROOT.iterdir()
        if path.is_dir()
    )

    model = FeatureExtractor(pretrained=True).to(DEVICE)
    model.eval()

    report = {
        "experiment": "Image-level anomaly scoring comparison",
        "categories": {},
        "failures": [],
    }

    for category in categories:
        print(f"\nEvaluating: {category}")

        try:
            results = evaluate_category(category, model)
            report["categories"][category] = results

            for method, values in results.items():
                print(
                    f"  {method:20} "
                    f"Image AUROC: {values['image_auroc']}"
                )

        except Exception as exc:
            report["failures"].append({
                "category": category,
                "error": str(exc),
            })
            print(f"  FAILED: {exc}")
            gc.collect()

    # Average category AUROC equally so large categories do not dominate.
    averages = {}

    for method in SCORING_METHODS:
        values = [
            category_results[method]["image_auroc"]
            for category_results in report["categories"].values()
            if category_results[method]["image_auroc"] is not None
        ]

        averages[method] = {
            "mean_image_auroc": (
                float(np.mean(values)) if values else None
            ),
            "categories_evaluated": len(values),
        }

    report["macro_average"] = averages

    output_path = OUTPUT_DIR / "image_scoring_comparison.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nMACRO-AVERAGE IMAGE AUROC")
    for method, values in averages.items():
        print(
            f"{method:20} "
            f"{values['mean_image_auroc']}"
        )

    print(f"\nReport saved to: {output_path}")
    print(f"Failed categories: {len(report['failures'])}")


if __name__ == "__main__":
    main()
