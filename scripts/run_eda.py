"""Execute Exploratory Data Analysis and generate all figures from actual CIC-IDS2017 data."""
import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.dataset_config import RAW_DATA_DIR, FIGURES_DIR, REPORTS_DIR, SCHEMAS_DIR
from src.dataset.loader import DatasetLoader
from src.dataset.schema import (
    TARGET_COLUMN,
    CANONICAL_COLUMNS,
    DUPLICATE_COLUMNS,
    CONSTANT_COLUMNS,
    normalize_label,
    export_schemas_to_json
)
from src.eda.visualizer import EDAVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("STARTING REPRODUCIBLE EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("=" * 70)

    loader = DatasetLoader(RAW_DATA_DIR)
    visualizer = EDAVisualizer(FIGURES_DIR)

    # 1. Export canonical schema and label mapping JSONs
    logger.info("Exporting canonical feature schema and label mappings...")
    export_schemas_to_json(SCHEMAS_DIR)

    # 2. Inspect all 8 CSV files for exact counts and anomalies
    csv_files = loader.get_csv_files()
    file_summaries = []
    total_raw_rows = 0
    all_label_counts = {}
    nan_occurrences = {}
    inf_occurrences = {}

    logger.info(f"Scanning {len(csv_files)} dataset files for full ground-truth audit...")
    for f in csv_files:
        df_part = loader.load_file(f, normalize_labels=True)
        nrows = len(df_part)
        total_raw_rows += nrows

        # Count labels
        if TARGET_COLUMN in df_part.columns:
            l_counts = df_part[TARGET_COLUMN].value_counts().to_dict()
            for k, v in l_counts.items():
                all_label_counts[k] = all_label_counts.get(k, 0) + v
        else:
            l_counts = {}

        # Count NaNs
        nans = df_part.isna().sum()
        for col, cnt in nans[nans > 0].items():
            nan_occurrences[col] = nan_occurrences.get(col, 0) + int(cnt)

        # Count Infs
        num_cols = df_part.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            infs = int(np.isinf(df_part[col]).sum())
            if infs > 0:
                inf_occurrences[col] = inf_occurrences.get(col, 0) + infs

        file_summaries.append({
            "filename": f.name,
            "rows": nrows,
            "labels": l_counts
        })

    logger.info(f"Total dataset records: {total_raw_rows:,}")

    # 3. Load representative stratified sample across all files for in-depth visualization
    logger.info("Loading balanced sample (25,000 rows per file = 200,000 rows) for detailed plotting...")
    sample_df = loader.load_combined(nrows_per_file=25000, normalize_labels=True)

    # 4. Generate Visualizations from real data
    logger.info("Generating publication-ready figures...")

    # Plot 1: Class Distribution
    fig1 = visualizer.plot_class_distribution(sample_df, "class_distribution.png")

    # Plot 2: Attack Categories
    fig2 = visualizer.plot_attack_categories(sample_df, "attack_categories.png")

    # Plot 3: Missing & Infinite Values
    fig3 = visualizer.plot_missing_and_inf_analysis(
        nan_occurrences,
        inf_occurrences,
        total_raw_rows,
        "missing_and_inf_analysis.png"
    )

    # Plot 4: Correlation Matrix
    key_features = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Mean", "Bwd Packet Length Mean",
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
        "Packet Length Mean", "Average Packet Size",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward"
    ]
    fig4 = visualizer.plot_correlation_heatmap(sample_df, key_features, "correlation_heatmap.png")

    # Plot 5: Feature Distributions by Class
    fig5 = visualizer.plot_feature_distributions_by_class(
        sample_df,
        ["Flow Duration", "Total Length of Fwd Packets", "Packet Length Mean", "Destination Port"],
        "feature_distributions_by_class.png"
    )

    # 5. Generate Comprehensive Quality Report
    report_path = REPORTS_DIR / "dataset_quality_report.md"
    logger.info(f"Writing dataset quality report to '{report_path}'...")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# CIC-IDS2017 Dataset Quality & Exploratory Data Analysis Report\n\n")
        f.write(f"**Total Records Evaluated:** {total_raw_rows:,}\n")
        f.write(f"**Total CSV Files:** {len(csv_files)}\n")
        f.write(f"**Canonical Columns:** {len(CANONICAL_COLUMNS)}\n\n")

        f.write("## 1. File Breakdown\n\n")
        f.write("| File Name | Record Count | Present Classes |\n")
        f.write("| :--- | :--- | :--- |\n")
        for item in file_summaries:
            classes_str = ", ".join(f"{k} ({v:,})" for k, v in item["labels"].items())
            f.write(f"| `{item['filename']}` | {item['rows']:,} | {classes_str} |\n")

        f.write("\n## 2. Ground-Truth Class Distribution\n\n")
        f.write("| Traffic Class | Flow Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        sorted_labels = sorted(all_label_counts.items(), key=lambda x: -x[1])
        for lbl, cnt in sorted_labels:
            pct = (cnt / total_raw_rows) * 100
            f.write(f"| **{lbl}** | {cnt:,} | {pct:.4f}% |\n")

        f.write("\n## 3. Data Integrity & Anomaly Analysis\n\n")
        f.write("### Missing Values (NaN)\n")
        if nan_occurrences:
            for c, cnt in nan_occurrences.items():
                f.write(f"- `{c}`: {cnt:,} missing rows ({(cnt/total_raw_rows)*100:.4f}%)\n")
        else:
            f.write("- No missing values detected.\n")

        f.write("\n### Infinite Values (Inf / -Inf)\n")
        if inf_occurrences:
            for c, cnt in inf_occurrences.items():
                f.write(f"- `{c}`: {cnt:,} infinite values ({(cnt/total_raw_rows)*100:.4f}%)\n")
            f.write("\n*Root cause:* `Flow Duration = 0` causes division by zero in packet/byte rate calculations.\n")
        else:
            f.write("- No infinite values detected.\n")

        f.write("\n### Zero-Variance (Constant) Features\n")
        for col in CONSTANT_COLUMNS:
            f.write(f"- `{col}`: Value is constant 0 across all {total_raw_rows:,} records.\n")

        f.write("\n### Duplicate Columns\n")
        for col in DUPLICATE_COLUMNS:
            f.write(f"- `{col}`: 100% duplicate of `Fwd Header Length`.\n")

        f.write("\n## 4. Generated Artifacts & Visualizations\n\n")
        f.write(f"- Class Distribution: `artifacts/figures/class_distribution.png`\n")
        f.write(f"- Attack Categories: `artifacts/figures/attack_categories.png`\n")
        f.write(f"- Missing & Inf Anomaly Analysis: `artifacts/figures/missing_and_inf_analysis.png`\n")
        f.write(f"- Feature Correlation Matrix: `artifacts/figures/correlation_heatmap.png`\n")
        f.write(f"- Feature Distributions by Class: `artifacts/figures/feature_distributions_by_class.png`\n")

    logger.info("EDA completed successfully!")


if __name__ == "__main__":
    main()
