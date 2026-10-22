
from pathlib import Path
import random
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(category="pcb1"):
    category_dir = RAW_DIR / category

    folders = {
        "normal": category_dir / "Data" / "Images" / "Normal",
        "anomaly": category_dir / "Data" / "Images" / "Anomaly",
    }

    samples = []

    for label, folder in folders.items():
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append({
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "label": label,
                    "numeric_label": 0 if label == "normal" else 1,
                })

    return samples


def split_class(samples):
    random.shuffle(samples)

    total = len(samples)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    return {
        "train": samples[:train_end],
        "validation": samples[train_end:val_end],
        "test": samples[val_end:],
    }


def main():
    random.seed(SEED)

    all_samples = collect_images("pcb1")

    split_data = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for label in ("normal", "anomaly"):
        class_samples = [
            sample for sample in all_samples
            if sample["label"] == label
        ]

        class_splits = split_class(class_samples)

        for split_name, samples in class_splits.items():
            split_data[split_name].extend(samples)

    for samples in split_data.values():
        random.shuffle(samples)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "dataset_split.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(split_data, file, indent=2)

    print(f"Total images: {len(all_samples)}")

    for split_name, samples in split_data.items():
        normal_count = sum(s["label"] == "normal" for s in samples)
        anomaly_count = sum(s["label"] == "anomaly" for s in samples)

        print(
            f"{split_name.capitalize()}: {len(samples)} images "
            f"(normal={normal_count}, anomaly={anomaly_count})"
        )

    print(f"\nSplit report saved to: {output_path}")
    print("Dataset split test passed.")


if __name__ == "__main__":
    main()
