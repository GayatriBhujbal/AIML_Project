"""Systematic feature selection execution on training data with validation benchmarking."""
import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.dataset_config import RAW_DATA_DIR, SCHEMAS_DIR, REPORTS_DIR, FIGURES_DIR
from src.dataset.loader import DatasetLoader
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.splitter import DataSplitter
from src.feature_selection.selector import FeatureSelector
from src.eda.visualizer import EDAVisualizer
from src.dataset.schema import TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("STARTING SYSTEMATIC FEATURE SELECTION PIPELINE")
    logger.info("=" * 70)

    # 1. Load balanced sample from actual dataset files
    loader = DatasetLoader(RAW_DATA_DIR)
    logger.info("Loading dataset sample (25,000 rows per file = 200,000 records)...")
    raw_df = loader.load_combined(nrows_per_file=25000, normalize_labels=True)

    # 2. Clean data (dropping duplicate/constant columns and handling infinite/negative values)
    cleaner = DataCleaner()
    cleaned_df, audit = cleaner.clean_dataframe(raw_df, is_training=True)
    logger.info(f"Cleaned dataset: {len(cleaned_df):,} rows, {len(cleaned_df.columns)} columns.")

    # 3. Stratified Train-Val-Test Split
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    train_df, val_df, test_df, split_report = splitter.split_dataframe(cleaned_df, stratify_col=TARGET_COLUMN)
    logger.info(f"Split sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # 4. Separate X and y
    X_train, y_train_str = cleaner.separate_features_and_target(train_df)
    X_val, y_val_str = cleaner.separate_features_and_target(val_df)

    y_train = (y_train_str != "BENIGN").astype(int).values
    y_val = (y_val_str != "BENIGN").astype(int).values

    # Impute missing values using training medians strictly fitted on X_train
    X_train_num = X_train.replace([np.inf, -np.inf], np.nan)
    train_medians = X_train_num.median()
    X_train_imp = X_train_num.fillna(train_medians)
    X_val_imp = X_val.replace([np.inf, -np.inf], np.nan).fillna(train_medians)

    # 5. Fit FeatureSelector on Training Set Only
    selector = FeatureSelector(correlation_threshold=0.95, top_k=25, random_state=42)
    selected_features = selector.fit_and_select(X_train_imp, y_train, X_val_imp, y_val)

    # 6. Save Artifacts
    selector.save_artifacts(SCHEMAS_DIR, REPORTS_DIR)

    # 7. Generate Feature Importance Plot
    visualizer = EDAVisualizer(FIGURES_DIR)
    visualizer.plot_rf_feature_importance(selector.feature_importances_, top_n=25, filename="feature_importance.png")

    logger.info("\n" + "=" * 70)
    logger.info("FEATURE SELECTION RESULTS:")
    logger.info(f"Selected {len(selected_features)} features:")
    for idx, f in enumerate(selected_features, 1):
        imp = selector.feature_importances_.get(f, 0.0)
        logger.info(f"  {idx:02d}. {f:<32} (RF Importance: {imp:.4f})")

    logger.info("\nBENCHMARK VALIDATION RESULTS:")
    for k, v in selector.benchmark_results_.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
