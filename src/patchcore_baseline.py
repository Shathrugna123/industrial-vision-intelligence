
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from mvtec_dataset import MVTecDataset
from feature_extractor import FeatureExtractor, DEVICE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"

CATEGORY = "bottle"
BATCH_SIZE = 4
MAX_MEMORY_FEATURES = 2000
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_features(model, loader):
    feature_maps = []
    labels = []
    masks = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            images = batch["image"].to(DEVICE)
            features = model(images)

            feature_maps.append(features.cpu())
            labels.extend(batch["label"].tolist())
            masks.extend(batch["mask"].cpu())

    return (
        torch.cat(feature_maps, dim=0),
        np.asarray(labels, dtype=np.int64),
        torch.stack(masks),
    )


def build_memory_bank(feature_maps, max_features):
    # Convert [N, C, H, W] into a collection of feature vectors.
    features = feature_maps.permute(0, 2, 3, 1).reshape(-1, feature_maps.shape[1])

    if len(features) > max_features:
        indices = torch.linspace(
            0, len(features) - 1, steps=max_features
        ).long()
        features = features[indices]

    return features.contiguous()


def score_feature_maps(feature_maps, memory_bank):
    anomaly_maps = []

    for feature_map in tqdm(feature_maps, desc="Scoring test images"):
        # [H, W, C] -> [H*W, C]
        vectors = feature_map.permute(1, 2, 0).reshape(-1, feature_map.shape[0])

        # Find the nearest normal feature vector for each spatial location.
        distances = torch.cdist(vectors, memory_bank)
        nearest_distances = distances.min(dim=1).values

        anomaly_map = nearest_distances.reshape(
            feature_map.shape[1], feature_map.shape[2]
        )

        # Resize the feature-level map to the original model input size.
        anomaly_map = F.interpolate(
            anomaly_map[None, None],
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        anomaly_maps.append(anomaly_map)

    return torch.stack(anomaly_maps)


def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def main():
    set_seed(SEED)
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

    print(f"Category: {CATEGORY}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Device: {DEVICE}")

    train_features, _, _ = extract_features(model, train_loader)
    test_features, test_labels, test_masks = extract_features(
        model, test_loader
    )

    memory_bank = build_memory_bank(
        train_features, MAX_MEMORY_FEATURES
    )

    print(f"\nMemory bank shape: {memory_bank.shape}")

    anomaly_maps = score_feature_maps(test_features, memory_bank)

    # Image-level score: largest anomaly value in each image.
    image_scores = anomaly_maps.flatten(1).max(dim=1).values.numpy()

    image_auc = safe_auc(test_labels, image_scores)

    # Pixel AUROC compares anomaly scores at pixels with binary masks.
    pixel_scores = anomaly_maps.flatten().numpy()
    pixel_labels = test_masks.flatten().numpy().astype(np.uint8)
    pixel_auc = safe_auc(pixel_labels, pixel_scores)

    results = {
        "method": "PatchCore-style nearest-neighbor baseline",
        "category": CATEGORY,
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "memory_bank_features": len(memory_bank),
        "image_auroc": image_auc,
        "pixel_auroc": pixel_auc,
        "notes": (
            "Simplified baseline using ResNet18 layer3 features, "
            "subsampled normal memory bank, nearest-neighbor scoring, "
            "and maximum pixel score for image-level anomaly score."
        ),
    }

    import json

    report_path = OUTPUT_DIR / f"{CATEGORY}_patchcore_baseline.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nBASELINE RESULTS")
    print(f"Image AUROC: {image_auc}")
    print(f"Pixel AUROC: {pixel_auc}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
