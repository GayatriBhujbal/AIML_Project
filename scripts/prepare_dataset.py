"""End-to-end dataset preparation and preprocessor fitting script."""
import os
import sys
import json
import argparse
from pathlib import Path
import logging
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.dataset_config import RAW_DATA_DIR, SCHEMAS_DIR, MODELS_DIR, REPORTS_DIR, DEFAULT_SAMPLE_PER_FILE, SAMPLE_STRATEGY
from src.dataset.loader import DatasetLoader
from src.dataset.schema import TARGET_COLUMN, export_schemas_to_json
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.splitter import DataSplitter
from src.preprocessing.pipeline import PreprocessingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="End-to-end dataset preprocessing and pipeline fitting.")
    parser.add_argument(
        "--sample-per-file", type=int, default=DEFAULT_SAMPLE_PER_FILE,
        help=(
            f"Rows to load per CSV file (default: {DEFAULT_SAMPLE_PER_FILE:,}; "
            "0 = full dataset). Shared default set in config/dataset_config.py."
        ),
    )
    parser.add_argument(
        "--sample-strategy", default=SAMPLE_STRATEGY, choices=["random", "head"],
        help="'random' (reproducible, avoids temporal bias) or 'head' (first N rows).",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("STARTING DATASET PREPARATION PIPELINE")
    logger.info("=" * 70)

    # 1. Export schemas
    export_schemas_to_json(SCHEMAS_DIR)

    # 2. Load dataset
    loader = DatasetLoader(RAW_DATA_DIR)
    nrows = None if args.sample_per_file == 0 else args.sample_per_file
    logger.info(
        f"Loading data (limit per file: {nrows or 'FULL'}, strategy: {args.sample_strategy})..."
    )
    raw_df = loader.load_combined(
        nrows_per_file=nrows,
        normalize_labels=True,
        sample_strategy=args.sample_strategy,
    )

    # 3. Clean dataset
    cleaner = DataCleaner()
    cleaned_df, audit = cleaner.clean_dataframe(raw_df, is_training=True)

    # 4. Stratified Split
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    train_df, val_df, test_df, split_report = splitter.split_dataframe(cleaned_df, stratify_col=TARGET_COLUMN)

    # 5. Load selected features if already generated, else use default top features
    selected_features_file = SCHEMAS_DIR / "selected_features.json"
    if selected_features_file.exists():
        with open(selected_features_file, "r", encoding="utf-8") as f:
            selected_features = json.load(f)["selected_features"]
        logger.info(f"Loaded {len(selected_features)} selected features from '{selected_features_file}'.")
    else:
        selected_features = None
        logger.info("No selected_features.json found; pipeline will retain all non-constant features.")

    # 6. Fit PreprocessingPipeline strictly on training data
    pipeline = PreprocessingPipeline(
        scaler_type="robust",
        selected_features=selected_features
    )
    X_train, y_train_str = cleaner.separate_features_and_target(train_df)
    pipeline.fit(X_train)

    # 7. Transform validation and test data
    X_val, y_val_str = cleaner.separate_features_and_target(val_df)
    X_test, y_test_str = cleaner.separate_features_and_target(test_df)

    X_train_trans = pipeline.transform(X_train)
    X_val_trans = pipeline.transform(X_val)
    X_test_trans = pipeline.transform(X_test)

    logger.info(f"Transformed shapes: Train={X_train_trans.shape}, Val={X_val_trans.shape}, Test={X_test_trans.shape}")

    # 8. Save fitted pipeline
    pipeline_path = pipeline.save(MODELS_DIR / "preprocessor.joblib")

    # 9. Save sample inference record for Members 2, 3, and 4
    sample_raw = X_test.iloc[:5].to_dict(orient="records")
    with open(MODELS_DIR / "inference_samples.json", "w", encoding="utf-8") as f:
        json.dump(sample_raw, f, indent=2)

    logger.info(f"Saved inference test samples to '{MODELS_DIR / 'inference_samples.json'}'.")
    logger.info("=" * 70)
    logger.info("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    logger.info(f"Fitted Pipeline Artifact: {pipeline_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
