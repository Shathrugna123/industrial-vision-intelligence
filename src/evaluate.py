
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from model import IndustrialVisionModel
from dataset_preparation import IMAGE_TRANSFORM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_FILE = PROJECT_ROOT / "outputs" / "reports" / "dataset_split.json"
CHECKPOINT_FILE = PROJECT_ROOT / "models" / "resnet18_best.pth"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

BATCH_SIZE = 8
DEVICE = torch.device("cpu")


class SplitDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = PROJECT_ROOT / sample["path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = IMAGE_TRANSFORM(image)

        return image_tensor, int(sample["numeric_label"]), sample["path"]


def main():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_FILE}"
        )

    with SPLIT_FILE.open("r", encoding="utf-8") as file:
        split_data = json.load(file)

    test_dataset = SplitDataset(split_data["test"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = IndustrialVisionModel(pretrained=False).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []
    error_cases = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()

            actual_labels = labels.tolist()

            all_labels.extend(actual_labels)
            all_predictions.extend(predictions)

            for path, actual, predicted in zip(
                paths, actual_labels, predictions
            ):
                if actual != predicted:
                    error_cases.append({
                        "path": path,
                        "actual": "anomaly" if actual == 1 else "normal",
                        "predicted": (
                            "anomaly" if predicted == 1 else "normal"
                        ),
                        "error_type": (
                            "false_negative" if actual == 1
                            else "false_positive"
                        ),
                    })

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels, all_predictions, zero_division=0
    )
    recall = recall_score(
        all_labels, all_predictions, zero_division=0
    )
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    matrix = confusion_matrix(
        all_labels, all_predictions, labels=[0, 1]
    )

    report = {
        "dataset": "VisA PCB1 (temporary test dataset)",
        "checkpoint": str(CHECKPOINT_FILE.relative_to(PROJECT_ROOT)),
        "test_images": len(all_labels),
        "metrics": {
            "accuracy": float(accuracy),
            "precision_anomaly": float(precision),
            "recall_anomaly": float(recall),
            "f1_anomaly": float(f1),
        },
        "confusion_matrix": {
            "labels": ["normal", "anomaly"],
            "matrix": matrix.tolist(),
        },
        "classification_report": classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=["normal", "anomaly"],
            output_dict=True,
            zero_division=0,
        ),
        "misclassified_images": error_cases,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "evaluation_report.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("TEST EVALUATION")
    print(f"Test images: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Anomaly precision: {precision:.4f}")
    print(f"Anomaly recall: {recall:.4f}")
    print(f"Anomaly F1-score: {f1:.4f}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("Labels: normal, anomaly")
    print(matrix)
    print(f"\nMisclassified images: {len(error_cases)}")
    print(f"Report saved to: {output_path}")
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
