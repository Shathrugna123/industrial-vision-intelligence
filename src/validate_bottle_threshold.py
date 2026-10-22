
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvtec_dataset import MVTecDataset
from feature_extractor import FeatureExtractor, DEVICE
from patchcore_baseline import build_memory_bank, score_feature_maps


CATEGORY = "bottle"
BATCH_SIZE = 4
MAX_MEMORY_FEATURES = 2000


def extract_features(model, loader):
    feature_maps = []
    labels = []
    paths = []
    defect_types = []

    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            features = model(images)

            feature_maps.append(features.cpu())
            labels.extend(batch["label"].numpy().tolist())
            paths.extend(batch["image_path"])
            defect_types.extend(batch["defect_type"])

    return (
        torch.cat(feature_maps, dim=0),
        np.asarray(labels, dtype=np.int64),
        paths,
        defect_types,
    )


def main():
    report_dir = ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = MVTecDataset(CATEGORY, split="train")
    test_dataset = MVTecDataset(CATEGORY, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    model = FeatureExtractor(pretrained=True).to(DEVICE)
    model.eval()

    print("Extracting normal training features...")
    train_features, _, _, _ = extract_features(model, train_loader)

    print("Extracting test features...")
    test_features, labels, paths, defect_types = extract_features(
        model, test_loader
    )

    memory_bank = build_memory_bank(
        train_features, MAX_MEMORY_FEATURES
    )

    print("Scoring test images...")
    anomaly_maps = score_feature_maps(test_features, memory_bank)

    scores = np.percentile(
        anomaly_maps.numpy().reshape(len(labels), -1),
        95,
        axis=1,
    )

    if len(np.unique(labels)) < 2:
        raise ValueError("Test set must contain normal and defective images.")

    image_auc = float(roc_auc_score(labels, scores))

    # Exploratory threshold analysis on the labelled test set.
    # These thresholds are NOT independent validation thresholds.
    candidate_thresholds = np.unique(
        np.quantile(scores, np.linspace(0, 1, 101))
    )

    threshold_results = []

    for threshold in candidate_thresholds:
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).ravel()

        threshold_results.append({
            "threshold": float(threshold),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "defect_recall": float(recall_score(labels, predictions, zero_division=0)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
        })

    # Candidate threshold maximizes F1 on this test set.
    # This is for exploration only; don't treat it as final calibration.
    best = max(
        threshold_results,
        key=lambda row: (row["f1"], row["defect_recall"]),
    )

    image_rows = []
    for path, defect_type, label, score in zip(
        paths, defect_types, labels, scores
    ):
        image_rows.append({
            "image_path": str(path),
            "defect_type": str(defect_type),
            "ground_truth": "defective" if label else "normal",
            "p95_anomaly_score": float(score),
        })

    image_df = pd.DataFrame(image_rows)
    threshold_df = pd.DataFrame(threshold_results)

    image_csv = report_dir / "bottle_threshold_image_scores.csv"
    threshold_csv = report_dir / "bottle_threshold_sweep.csv"
    report_json = report_dir / "bottle_threshold_validation.json"

    image_df.to_csv(image_csv, index=False)
    threshold_df.to_csv(threshold_csv, index=False)

    report = {
        "category": CATEGORY,
        "device": str(DEVICE),
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "normal_test_images": int((labels == 0).sum()),
        "defective_test_images": int((labels == 1).sum()),
        "image_auroc_p95": image_auc,
        "exploratory_best_threshold": best,
        "important_limitation": (
            "The labelled test set was used to explore thresholds. "
            "The best threshold is optimistic and must not be presented "
            "as independently validated. Calibrate using a separate "
            "normal validation set, then evaluate once on held-out test data."
        ),
    }

    with report_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nBOTTLE VALIDATION RESULTS")
    print(f"Normal test images:    {(labels == 0).sum()}")
    print(f"Defective test images: {(labels == 1).sum()}")
    print(f"Image AUROC (P95):     {image_auc:.4f}")
    print("\nExploratory threshold with highest F1:")
    for key, value in best.items():
        print(f"{key}: {value}")

    print("\nFiles saved:")
    print(image_csv)
    print(threshold_csv)
    print(report_json)


if __name__ == "__main__":
    main()
