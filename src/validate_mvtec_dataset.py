
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def count_images(folder):
    if not folder.exists():
        return 0

    return sum(
        1 for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_category(category_dir):
    train_dir = category_dir / "train"
    test_dir = category_dir / "test"
    mask_dir = category_dir / "ground_truth"

    result = {
        "category": category_dir.name,
        "train": {},
        "test": {},
        "ground_truth": {},
        "total_train_images": 0,
        "total_test_images": 0,
        "total_mask_images": 0,
        "issues": [],
    }

    for split_name, split_dir in [
        ("train", train_dir),
        ("test", test_dir),
        ("ground_truth", mask_dir),
    ]:
        if not split_dir.is_dir():
            result["issues"].append(
                f"Missing {split_name} directory"
            )
            continue

        counts = {}

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            counts[class_dir.name] = count_images(class_dir)

        result[split_name] = counts

    result["total_train_images"] = sum(result["train"].values())
    result["total_test_images"] = sum(result["test"].values())
    result["total_mask_images"] = sum(
        result["ground_truth"].values()
    )

    if result["total_train_images"] == 0:
        result["issues"].append("No training images found")

    if result["total_test_images"] == 0:
        result["issues"].append("No test images found")

    test_defect_classes = {
        name for name, count in result["test"].items()
        if name != "good" and count > 0
    }

    mask_classes = {
        name for name, count in result["ground_truth"].items()
        if count > 0
    }

    missing_mask_classes = test_defect_classes - mask_classes

    if missing_mask_classes:
        result["issues"].append(
            "No masks found for defect classes: "
            + ", ".join(sorted(missing_mask_classes))
        )

    return result


def main():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"MVTec dataset folder not found: {DATASET_ROOT}"
        )

    category_dirs = sorted(
        path for path in DATASET_ROOT.iterdir()
        if path.is_dir()
    )

    if not category_dirs:
        raise RuntimeError("No category folders found")

    results = [
        inspect_category(category_dir)
        for category_dir in category_dirs
    ]

    print(f"MVTec categories found: {len(results)}")
    print("-" * 75)

    for result in results:
        print(
            f"{result['category']:12} | "
            f"Train: {result['total_train_images']:4} | "
            f"Test: {result['total_test_images']:4} | "
            f"Masks: {result['total_mask_images']:4} | "
            f"Issues: {len(result['issues'])}"
        )

        for issue in result["issues"]:
            print(f"    WARNING: {issue}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "mvtec_dataset_validation.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("-" * 75)
    print(f"Total categories: {len(results)}")
    print(
        "Total training images:",
        sum(item["total_train_images"] for item in results),
    )
    print(
        "Total test images:",
        sum(item["total_test_images"] for item in results),
    )
    print(
        "Total masks:",
        sum(item["total_mask_images"] for item in results),
    )
    print(f"Report saved to: {report_path}")

    if any(item["issues"] for item in results):
        print("Validation finished with warnings. Review the report.")
    else:
        print("Dataset structure validation passed.")


if __name__ == "__main__":
    main()
