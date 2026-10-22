
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
OUTPUTS = ROOT / "outputs"

FILES = {
    "image_scoring": "image_scoring_comparison.json",
    "localization": "localization_evaluation.json",
    "all_categories": "mvtec_all_categories_evaluation.json",
    "model_comparison": "model_comparison.json",
    "multilayer": "multilayer_evaluation.json",
    "smoothing": "smoothing_evaluation.json",
}


def read_json(filename):
    path = REPORTS / filename
    if not path.exists():
        print(f"Skipping missing file: {filename}")
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def flatten(data, prefix=""):
    """Flatten nested dictionaries into readable key-value rows."""
    rows = []
    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else key
            rows.extend(flatten(value, name))
    elif isinstance(data, list):
        rows.append((prefix, json.dumps(data)))
    else:
        rows.append((prefix, data))
    return rows


def main():
    summary_rows = []
    details = {}

    for report_name, filename in FILES.items():
        data = read_json(filename)
        if data is None:
            continue

        details[report_name] = data
        for key, value in flatten(data):
            summary_rows.append({
                "report": report_name,
                "metric": key,
                "value": value,
            })

    if not summary_rows:
        raise RuntimeError("No existing reports were found.")

    csv_path = OUTPUTS / "project_evaluation_summary.csv"
    json_path = REPORTS / "project_evaluation_summary.json"

    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(details, file, indent=2)

    print("\nPROJECT REPORT SUMMARY CREATED")
    print(f"Reports combined: {len(details)}")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("\nNo model inference or retraining was performed.")


if __name__ == "__main__":
    main()
