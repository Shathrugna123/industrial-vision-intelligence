
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "image_scoring_comparison.json"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    averages = report["macro_average"]

    methods = list(averages.keys())
    scores = [
        averages[method]["mean_image_auroc"]
        for method in methods
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, scores)

    plt.title("Image-Level Anomaly Scoring Comparison")
    plt.xlabel("Scoring Method")
    plt.ylabel("Mean Image AUROC")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{score:.4f}",
            ha="center",
        )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "image_scoring_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Chart saved to: {output_path}")

if __name__ == "__main__":
    main()
