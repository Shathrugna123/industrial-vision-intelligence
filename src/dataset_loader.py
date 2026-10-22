
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_visa_samples(category="pcb1", samples_per_class=5):
    """Load sample normal and anomalous images from the VisA dataset."""

    category_dir = RAW_DIR / category

    normal_dir = category_dir / "Data" / "Images" / "Normal"
    anomaly_dir = category_dir / "Data" / "Images" / "Anomaly"

    if not normal_dir.exists() or not anomaly_dir.exists():
        raise FileNotFoundError(
            f"Dataset folders not found for category: {category}"
        )

    normal_images = sorted(
        path for path in normal_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:samples_per_class]

    anomaly_images = sorted(
        path for path in anomaly_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:samples_per_class]

    samples = []

    for path in normal_images:
        samples.append({
            "path": path,
            "label": "normal",
            "image": Image.open(path).convert("RGB"),
        })

    for path in anomaly_images:
        samples.append({
            "path": path,
            "label": "anomaly",
            "image": Image.open(path).convert("RGB"),
        })

    if not samples:
        raise ValueError(f"No images found for category: {category}")

    print(f"Category: {category}")
    print(f"Normal samples loaded: {len(normal_images)}")
    print(f"Anomaly samples loaded: {len(anomaly_images)}")
    print(f"Total samples loaded: {len(samples)}")

    return samples


def main():
    samples = load_visa_samples(
        category="pcb1",
        samples_per_class=5,
    )

    for sample in samples:
        print(
            f"Label: {sample['label']} | "
            f"File: {sample['path'].name} | "
            f"Size: {sample['image'].size}"
        )

        sample["image"].close()


if __name__ == "__main__":
    main()
