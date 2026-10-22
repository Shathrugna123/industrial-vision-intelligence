
from pathlib import Path
import json

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def load_json(filename):
    path = REPORT_DIR / filename
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    original_image = load_json("image_scoring_comparison.json")
    original_pixel = load_json("localization_evaluation.json")
    multilayer = load_json("multilayer_evaluation.json")

    image_scores = original_image["categories"]
    pixel_scores = original_pixel["categories"]
    multilayer_scores = multilayer["categories"]

    rows = []

    for category in sorted(multilayer_scores):
        original_image_auc = image_scores[category]["p95"]["image_auroc"]
        original_pixel_auc = pixel_scores[category]["pixel_auroc"]

        experimental = multilayer_scores[category]

        rows.append({
            "category": category,
            "original_image_auroc_p95": original_image_auc,
            "multilayer_image_auroc_p95": experimental["image_auroc_p95"],
            "image_auc_change": (
                experimental["image_auroc_p95"] - original_image_auc
            ),
            "original_pixel_auroc": original_pixel_auc,
            "multilayer_pixel_auroc": experimental["pixel_auroc"],
            "pixel_auc_change": (
                experimental["pixel_auroc"] - original_pixel_auc
            ),
        })

    df = pd.DataFrame(rows)

    output_csv = REPORT_DIR / "model_comparison.csv"
    output_json = REPORT_DIR / "model_comparison.json"

    df.to_csv(output_csv, index=False)

    summary = {
        "categories_compared": len(df),
        "original_mean_image_auroc_p95": float(
            df["original_image_auroc_p95"].mean()
        ),
        "multilayer_mean_image_auroc_p95": float(
            df["multilayer_image_auroc_p95"].mean()
        ),
        "original_mean_pixel_auroc": float(
            df["original_pixel_auroc"].mean()
        ),
        "multilayer_mean_pixel_auroc": float(
            df["multilayer_pixel_auroc"].mean()
        ),
        "categories_image_improved": int(
            (df["image_auc_change"] > 0).sum()
        ),
        "categories_pixel_improved": int(
            (df["pixel_auc_change"] > 0).sum()
        ),
    }

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(
            {"summary": summary, "categories": rows},
            file,
            indent=2,
        )

    print("\nMODEL COMPARISON")
    print(df.round(4).to_string(index=False))

    print("\nSUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nCSV saved to: {output_csv}")
    print(f"JSON saved to: {output_json}")


if __name__ == "__main__":
    main()
