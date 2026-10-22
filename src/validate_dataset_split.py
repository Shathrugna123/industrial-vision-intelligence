
from pathlib import Path
import json
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
SPLIT_FILE = REPORT_DIR / "dataset_split.json"

VALID_SPLITS = {"train", "validation", "test"}
VALID_LABELS = {"normal", "anomaly"}


def main():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {SPLIT_FILE}"
        )

    with SPLIT_FILE.open("r", encoding="utf-8") as file:
        split_data = json.load(file)

    if set(split_data.keys()) != VALID_SPLITS:
        raise ValueError(
            f"Expected splits: {sorted(VALID_SPLITS)}"
        )

    all_paths = []
    total_images = 0
    errors = []

    for split_name, samples in split_data.items():
        if not isinstance(samples, list):
            errors.append(f"{split_name} must contain a list.")
            continue

        labels = Counter()

        for sample in samples:
            relative_path = sample.get("path")
            label = sample.get("label")

            if label not in VALID_LABELS:
                errors.append(
                    f"Invalid label in {split_name}: {label}"
                )
                continue

            if not relative_path:
                errors.append(
                    f"Missing image path in {split_name}."
                )
                continue

            image_path = PROJECT_ROOT / relative_path

            if not image_path.is_file():
                errors.append(
                    f"Image does not exist: {relative_path}"
                )

            all_paths.append(relative_path)
            labels[label] += 1
            total_images += 1

        print(
            f"{split_name.capitalize()}: {len(samples)} images | "
            f"Normal: {labels['normal']} | "
            f"Anomaly: {labels['anomaly']}"
        )

        if not samples:
            errors.append(f"{split_name} split is empty.")

        if not labels["normal"] or not labels["anomaly"]:
            errors.append(
                f"{split_name} does not contain both classes."
            )

    duplicate_paths = {
        path for path, count in Counter(all_paths).items()
        if count > 1
    }

    if duplicate_paths:
        errors.append(
            f"Duplicate paths across splits: {len(duplicate_paths)}"
        )

    print(f"\nTotal manifest entries: {total_images}")
    print(f"Unique image paths: {len(set(all_paths))}")

    if errors:
        print("\nVALIDATION FAILED:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nDataset split validation passed.")
    print("All image paths exist; no duplicate paths were found.")


if __name__ == "__main__":
    main()
