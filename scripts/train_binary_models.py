"""Production training script for Decision Tree baseline and Random Forest binary detector."""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Tuple, Any, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.dataset_config import (
    RAW_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    SCHEMAS_DIR,
    DEFAULT_RANDOM_STATE
)
from src.dataset.loader import DatasetLoader
from src.dataset.schema import TARGET_COLUMN
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.splitter import DataSplitter
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.binary_detector import DecisionTreeBaseline, RandomForestBinaryDetector
from src.models.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calibrate_threshold(model: Any, X_val: pd.DataFrame, y_val: np.ndarray) -> Tuple[float, float]:
    """
    Calibrate decision threshold on validation data to maximize F1-score for threat detection.
    """
    logger.info("Calibrating decision threshold on validation set...")
    val_probs = model.predict_proba(X_val)[:, 1]

    best_thresh = 0.50
    best_f1 = 0.0

    thresholds = np.linspace(0.1, 0.9, 17)
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        score = f1_score(y_val, preds, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)

    logger.info(f"Optimal validation threshold: {best_thresh:.2f} (Val F1: {best_f1:.5f})")
    return best_thresh, best_f1


def main():
    parser = argparse.ArgumentParser(description="Train Decision Tree baseline and Random Forest binary threat detector.")
    parser.add_argument("--sample-per-file", type=int, default=25000,
                        help="Number of rows to load per file (default: 25,000 for balanced 200k records). Use 0 for full dataset.")
    parser.add_argument("--n-estimators", type=int, default=100, help="Random Forest estimators (default: 100).")
    parser.add_argument("--max-depth-rf", type=int, default=16, help="Random Forest max depth (default: 16).")
    parser.add_argument("--max-depth-dt", type=int, default=12, help="Decision Tree max depth (default: 12).")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed.")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("STARTING BINARY THREAT DETECTION TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Load actual dataset
    loader = DatasetLoader(RAW_DATA_DIR)
    nrows = None if args.sample_per_file == 0 else args.sample_per_file
    logger.info(f"Loading data (limit per file: {nrows or 'FULL'})...")
    raw_df = loader.load_combined(nrows_per_file=nrows, normalize_labels=True)

    # 2. Clean data
    cleaner = DataCleaner()
    cleaned_df, audit = cleaner.clean_dataframe(raw_df, is_training=True)
    logger.info(f"Dataset cleaned: {len(cleaned_df):,} records remaining.")

    # 3. Stratified Split (70% Train, 15% Validation, 15% Test)
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=args.random_state)
    train_df, val_df, test_df, split_report = splitter.split_dataframe(cleaned_df, stratify_col=TARGET_COLUMN)
    logger.info(f"Split sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # 4. Separate X and y
    X_train_raw, y_train_str = cleaner.separate_features_and_target(train_df)
    X_val_raw, y_val_str = cleaner.separate_features_and_target(val_df)
    X_test_raw, y_test_str = cleaner.separate_features_and_target(test_df)

    # Encode binary labels (0: BENIGN, 1: ATTACK)
    y_train = PreprocessingPipeline.encode_labels(y_train_str, mode="binary")
    y_val = PreprocessingPipeline.encode_labels(y_val_str, mode="binary")
    y_test = PreprocessingPipeline.encode_labels(y_test_str, mode="binary")

    logger.info(
        f"Train Class Distribution: Benign={np.sum(y_train == 0):,} ({(np.mean(y_train == 0)*100):.1f}%), "
        f"Attack={np.sum(y_train == 1):,} ({(np.mean(y_train == 1)*100):.1f}%)"
    )

    # 5. Load selected features and fit PreprocessingPipeline strictly on Train
    selected_features_path = SCHEMAS_DIR / "selected_features.json"
    if selected_features_path.exists():
        with open(selected_features_path, "r", encoding="utf-8") as f:
            selected_features = json.load(f)["selected_features"]
        logger.info(f"Loaded {len(selected_features)} features from '{selected_features_path}'.")
    else:
        selected_features = None

    pipeline = PreprocessingPipeline(scaler_type="robust", selected_features=selected_features)
    pipeline.fit(X_train_raw)
    pipeline.save(MODELS_DIR / "preprocessor.joblib")

    # Transform Train, Val, Test using fitted statistics
    X_train = pipeline.transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    # 6. Model 1: Train Decision Tree Baseline
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING MODEL 1: DECISION TREE BASELINE")
    logger.info("=" * 50)
    dt_model = DecisionTreeBaseline(
        max_depth=args.max_depth_dt,
        class_weight="balanced",
        random_state=args.random_state
    )
    dt_model.fit(X_train, y_train)

    # 7. Model 2: Train Random Forest Binary Detector
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING MODEL 2: RANDOM FOREST BINARY DETECTOR")
    logger.info("=" * 50)
    rf_model = RandomForestBinaryDetector(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth_rf,
        class_weight="balanced",
        random_state=args.random_state
    )
    rf_model.fit(X_train, y_train)

    # Calibrate threshold on validation data
    opt_thresh, val_f1 = calibrate_threshold(rf_model, X_val, y_val)
    rf_model.set_threshold(opt_thresh)

    # 8. Save Trained Models
    dt_path = dt_model.save(MODELS_DIR / "binary_decision_tree.joblib")
    rf_path = rf_model.save(MODELS_DIR / "binary_random_forest.joblib")

    # 9. Comprehensive Evaluation on Untouched Held-Out Test Set
    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION ON HELD-OUT TEST SET")
    logger.info("=" * 50)
    evaluator = ModelEvaluator(figures_dir=FIGURES_DIR, reports_dir=REPORTS_DIR)

    dt_results = evaluator.evaluate_model("Decision Tree Baseline", dt_model, X_test, y_test)
    rf_results = evaluator.evaluate_model("Random Forest Detector", rf_model, X_test, y_test, threshold=opt_thresh)

    # 10. Generate Evaluation Figures
    models_dict = {
        "Decision Tree Baseline": dt_model,
        "Random Forest Detector": rf_model
    }
    evaluator.plot_roc_curves(models_dict, X_test, y_test, "binary_roc_curve.png")
    evaluator.plot_precision_recall_curves(models_dict, X_test, y_test, "binary_precision_recall_curve.png")
    evaluator.plot_confusion_matrices(models_dict, X_test, y_test, "binary_confusion_matrices.png")

    # 11. Generate Markdown and JSON Reports
    report_path = evaluator.generate_markdown_report(dt_results, rf_results, "binary_model_evaluation_report.md")

    json_report = {
        "decision_tree_baseline": dt_results,
        "random_forest_detector": rf_results,
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test)
        },
        "calibrated_threshold": opt_thresh
    }
    with open(REPORTS_DIR / "binary_model_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING & EVALUATION COMPLETE!")
    logger.info(f"Decision Tree Artifact: {dt_path}")
    logger.info(f"Random Forest Artifact: {rf_path}")
    logger.info(f"Evaluation Report: {report_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
