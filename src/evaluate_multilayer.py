
from pathlib import Path
import gc
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from mvtec_dataset import MVTecDataset, DATASET_ROOT
from multilayer_feature_extractor import MultiLayerFeatureExtractor
from feature_extractor import DEVICE
from patchcore_baseline import BATCH_SIZE, MAX_MEMORY_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"


def extract_features(model, loader):
    feature_maps = []
    labels = []
    masks = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            images = batch["image"].to(DEVICE)
            feature_maps.append(model(images).cpu())
            labels.extend(batch["label"].tolist())
            masks.append(batch["mask"].cpu())

    return (
        torch.cat(feature_maps),
        np.asarray(labels, dtype=np.int64),
        torch.cat(masks),
    )


def build_memory_bank(features, max_features):
    vectors = features.permute(0, 2, 3, 1).reshape(
        -1, features.shape[1]
    )

    if len(vectors) > max_features:
        indices = torch.linspace(
            0, len(vectors) - 1, steps=max_features
        ).long()
        vectors = vectors[indices]

    return vectors.contiguous()


def score_maps(features, memory_bank):
    maps = []

    for feature_map in tqdm(features, desc="Scoring test images"):
        vectors = feature_map.permute(1, 2, 0).reshape(
            -1, feature_map.shape[0]
        )

        distances = torch.cdist(vectors, memory_bank)
        nearest = distances.min(dim=1).values

        anomaly_map = nearest.reshape(
            feature_map.shape[1], feature_map.shape[2]
        )

        # Convert to the 224 x 224 input resolution.
        anomaly_map = torch.nn.functional.interpolate(
            anomaly_map[None, None],
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        maps.append(anomaly_map)

    return torch.stack(maps)


def auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def evaluate_category(category, model):
    train_ds = MVTecDataset(category, split="train")
    test_ds = MVTecDataset(category, split="test")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )

    train_features, _, _ = extract_features(model, train_loader)
    test_features, labels, masks = extract_features(model, test_loader)

    memory = build_memory_bank(train_features, MAX_MEMORY_FEATURES)
    maps = score_maps(test_features, memory)

    # Use the 95th percentile image score from our earlier experiment.
    image_scores = torch.quantile(
        maps.flatten(1), 0.95, dim=1
    ).numpy()

    pixel_auc = auc(
        masks.numpy().reshape(-1).astype(np.uint8),
        maps.numpy().reshape(-1),
    )

    result = {
        "image_auroc_p95": auc(labels, image_scores),
        "pixel_auroc": pixel_auc,
        "train_images": len(train_ds),
        "test_images": len(test_ds),
        "feature_shape": list(train_features.shape[1:]),
    }

    del train_features, test_features, memory, maps, masks
    del train_ds, test_ds, train_loader, test_loader
    gc.collect()

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    categories = sorted(
        p.name for p in DATASET_ROOT.iterdir() if p.is_dir()
    )

    model = MultiLayerFeatureExtractor(pretrained=True).to(DEVICE)
    model.eval()

    report = {
        "experiment": "Multi-layer ResNet18 anomaly detection",
        "categories": {},
        "failures": [],
    }

    for category in categories:
        print(f"\nEvaluating: {category}")

        try:
            result = evaluate_category(category, model)
            report["categories"][category] = result
            print("Image AUROC (P95):", result["image_auroc_p95"])
            print("Pixel AUROC:", result["pixel_auroc"])
        except Exception as exc:
            report["failures"].append({
                "category": category,
                "error": str(exc),
            })
            print("FAILED:", exc)
            gc.collect()

    image_values = [
        r["image_auroc_p95"]
        for r in report["categories"].values()
        if r["image_auroc_p95"] is not None
    ]
    pixel_values = [
        r["pixel_auroc"]
        for r in report["categories"].values()
        if r["pixel_auroc"] is not None
    ]

    report["macro_average"] = {
        "image_auroc_p95": (
            float(np.mean(image_values)) if image_values else None
        ),
        "pixel_auroc": (
            float(np.mean(pixel_values)) if pixel_values else None
        ),
    }

    output_path = OUTPUT_DIR / "multilayer_evaluation.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nMULTI-LAYER EVALUATION SUMMARY")
    print("Categories evaluated:", len(report["categories"]))
    print("Mean image AUROC (P95):",
          report["macro_average"]["image_auroc_p95"])
    print("Mean pixel AUROC:",
          report["macro_average"]["pixel_auroc"])
    print("Failed categories:", len(report["failures"]))
    print("Report saved to:", output_path)


if __name__ == "__main__":
    main()
