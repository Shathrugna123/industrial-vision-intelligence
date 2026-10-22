
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def images_in(folder):
    if not folder.exists():
        return []

    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    audit = []
    total_missing_masks = 0
    total_defect_images = 0

    for category_dir in sorted(DATASET_ROOT.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        train_dir = category_dir / "train"
        test_dir = category_dir / "test"
        gt_dir = category_dir / "ground_truth"

        train_good = images_in(train_dir / "good")
        test_images = images_in(test_dir)

        defect_images = [
            path for path in test_images
            if path.parent.name != "good"
        ]

        missing_masks = []
        matched_masks = []

        for image_path in defect_images:
            defect_type = image_path.parent.name
            mask_dir = gt_dir / defect_type

            # Standard MVTec masks use the image stem plus "_mask".
            mask_path = mask_dir / f"{image_path.stem}_mask.png"

            if mask_path.is_file():
                matched_masks.append({
                    "image": str(image_path.relative_to(PROJECT_ROOT)),
                    "mask": str(mask_path.relative_to(PROJECT_ROOT)),
                    "defect_type": defect_type,
                })
            else:
                missing_masks.append(
                    str(image_path.relative_to(PROJECT_ROOT))
                )

        category_result = {
            "category": category,
            "train_good_images": len(train_good),
            "test_images": len(test_images),
            "test_defect_images": len(defect_images),
            "matched_masks": len(matched_masks),
            "missing_masks": missing_masks,
            "defect_types": sorted({
                path.parent.name for path in defect_images
            }),
        }

        audit.append(category_result)
        total_defect_images += len(defect_images)
        total_missing_masks += len(missing_masks)

        print(
            f"{category:12} | "
            f"train_good={len(train_good):4} | "
            f"test={len(test_images):4} | "
            f"defects={len(defect_images):4} | "
            f"matched_masks={len(matched_masks):4} | "
            f"missing_masks={len(missing_masks):3}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "mvtec_dataset_audit.json"

    report = {
        "categories_audited": len(audit),
        "total_test_defect_images": total_defect_images,
        "total_matched_masks": sum(
            item["matched_masks"] for item in audit
        ),
        "total_missing_masks": total_missing_masks,
        "categories": audit,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nDataset audit completed.")
    print(f"Categories audited: {len(audit)}")
    print(f"Test defect images: {total_defect_images}")
    print(f"Matched masks: {report['total_matched_masks']}")
    print(f"Missing masks: {total_missing_masks}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
