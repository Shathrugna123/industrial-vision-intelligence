from pathlib import Path
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"


def summarize_visa(category_dir):
    annotation_file = category_dir / "image_anno.csv"

    with annotation_file.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    labels = {}
    missing_images = []
    missing_masks = []
    mask_count = 0

    for row in rows:
        label = row["label"].strip().lower()
        labels[label] = labels.get(label, 0) + 1

        image_path = RAW_DIR / row["image"]
        if not image_path.is_file():
            missing_images.append(row["image"])

        mask_value = row.get("mask", "").strip()
        if mask_value:
            mask_count += 1
            if not (RAW_DIR / mask_value).is_file():
                missing_masks.append(mask_value)

    return {
        "dataset": "VisA",
        "category": category_dir.name,
        "annotation_rows": len(rows),
        "labels": labels,
        "annotated_masks": mask_count,
        "missing_images": len(missing_images),
        "missing_masks": len(missing_masks),
    }


def summarize_mvtec(category_dir):
    train_good = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    ground_truth = category_dir / "ground_truth"

    if not train_good.is_dir() or not test_dir.is_dir():
        return None

    count_images = lambda folder: (
        sum(
            1 for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if folder.is_dir() else 0
    )

    test_counts = {}
    for folder in sorted(test_dir.iterdir()):
        if folder.is_dir():
            test_counts[folder.name] = count_images(folder)

    mask_counts = {}
    if ground_truth.is_dir():
        for folder in sorted(ground_truth.iterdir()):
            if folder.is_dir():
                mask_counts[folder.name] = count_images(folder)

    return {
        "dataset": "MVTec AD",
        "category": category_dir.name,
        "train_good_images": count_images(train_good),
        "test_images_by_label": test_counts,
        "ground_truth_masks_by_label": mask_counts,
    }


def main():
    results = []

    visa_categories = [
        folder for folder in RAW_DIR.iterdir()
        if folder.is_dir() and (folder / "image_anno.csv").is_file()
    ]

    mvtec_categories = [
        folder for folder in RAW_DIR.iterdir()
        if folder.is_dir() and (folder / "train" / "good").is_dir()
    ]

    for category in visa_categories:
        results.append(summarize_visa(category))

    for category in mvtec_categories:
        result = summarize_mvtec(category)
        if result:
            results.append(result)

    if not results:
        print("No supported dataset structure found in:", RAW_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "dataset_summary.json"
    output_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    main()
