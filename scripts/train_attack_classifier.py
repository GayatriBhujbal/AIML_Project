#!/usr/bin/env python
"""
Train XGBoost multiclass attack classifier (Member 3).

Usage:
    python scripts/train_attack_classifier.py [--sample-per-file N] [--n-estimators N]

This script:
1. Loads raw CIC-IDS2017 CSV files from MachineLearningCVE/
2. Applies the SHARED PreprocessingPipeline (fitted by Member 1 -- NOT re-fitted here)
3. Filters to attack-only records (BENIGN excluded -- two-stage design)
4. Uses the SAME 70/15/15 stratified split as Members 1 and 2 (random_state=42)
5. Trains XGBoostAttackClassifier with inverse-frequency sample_weight
6. Evaluates on the held-out test set
7. Saves model, class mapping, evaluation JSON, and figures

Strict rules:
- The preprocessor.joblib is loaded read-only -- never re-fitted.
- Class mapping derived from actual labels in the dataset files.
- No fabricated metrics or SHAP values.
- If dataset CSVs are absent, the script exits with a clear error message.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.dataset_config import (
    MODELS_DIR, FIGURES_DIR, REPORTS_DIR, SCHEMAS_DIR,
    DEFAULT_RANDOM_STATE,
)
from src.dataset.loader import DatasetLoader
from src.dataset.schema import normalize_label, TARGET_COLUMN
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.splitter import DataSplitter
from src.models.attack_classifier import XGBoostAttackClassifier
from src.models.attack_evaluator import AttackClassifierEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost attack classifier")
    parser.add_argument(
        "--sample-per-file", type=int, default=None,
        help="Max rows per CSV file for memory-constrained training (default: all rows)"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=300,
        help="Number of XGBoost estimators (default: 300)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=6,
        help="XGBoost max_depth (default: 6)"
    )
    parser.add_argument(
        "--shap-samples", type=int, default=500,
        help="Max rows for SHAP global explanation (default: 500)"
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP computation (faster training run)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load dataset -- fail clearly if CSVs are absent
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading CIC-IDS2017 dataset...")
    try:
        loader = DatasetLoader()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(
            "Place CIC-IDS2017 CSV files in MachineLearningCVE/ and re-run.\n"
            "See README.md Section 3 for download instructions."
        )
        sys.exit(1)

    raw_df = loader.load_combined(nrows_per_file=args.sample_per_file)
    logger.info(f"Loaded {len(raw_df):,} records. Label distribution:")
    logger.info(raw_df[TARGET_COLUMN].value_counts().to_string())

    # ------------------------------------------------------------------
    # 2. Load shared preprocessor (fitted by Member 1 -- read-only)
    # ------------------------------------------------------------------
    logger.info("Step 2: Loading shared PreprocessingPipeline (Member 1 artifact)...")
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    if not preprocessor_path.exists():
        logger.error(
            f"preprocessor.joblib not found at {preprocessor_path}. "
            "Run: python scripts/prepare_dataset.py --sample-per-file 30000"
        )
        sys.exit(1)
    pipeline = PreprocessingPipeline.load(preprocessor_path)
    logger.info(
        f"Loaded. Feature count: {len(pipeline.feature_names_out_)}. "
        f"Features: {pipeline.feature_names_out_}"
    )

    # ------------------------------------------------------------------
    # 3. Split FIRST (same 70/15/15 as Members 1 & 2), then preprocess
    #    This matches the pipeline used in Member 1's prepare_dataset.py
    # ------------------------------------------------------------------
    logger.info("Step 3: Stratified 70/15/15 split (random_state=42)...")
    splitter = DataSplitter(random_state=DEFAULT_RANDOM_STATE)
    train_df, val_df, test_df, split_report = splitter.split_dataframe(
        raw_df, reset_index=True
    )
    logger.info(
        f"Split: train={split_report['train_size']:,} | "
        f"val={split_report['val_size']:,} | test={split_report['test_size']:,}"
    )

    # ------------------------------------------------------------------
    # 4. Separate X and y, transform features (shared pipeline)
    # ------------------------------------------------------------------
    logger.info("Step 4: Applying shared preprocessor to each split...")

    def extract_xy(df: pd.DataFrame):
        y = df[TARGET_COLUMN].apply(normalize_label)
        X_raw = df.drop(columns=[TARGET_COLUMN])
        X_proc = pipeline.transform(X_raw)
        return X_proc, y

    X_train_all, y_train_all = extract_xy(train_df)
    X_val_all, y_val_all = extract_xy(val_df)
    X_test_all, y_test_all = extract_xy(test_df)

    # ------------------------------------------------------------------
    # 5. Filter to attack-only records (two-stage design)
    # ------------------------------------------------------------------
    logger.info("Step 5: Filtering to attack-only records (BENIGN excluded)...")

    def filter_attacks(X: pd.DataFrame, y: pd.Series):
        mask = y.str.strip().str.upper() != "BENIGN"
        return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    X_train, y_train = filter_attacks(X_train_all, y_train_all)
    X_val, y_val = filter_attacks(X_val_all, y_val_all)
    X_test, y_test = filter_attacks(X_test_all, y_test_all)

    logger.info(f"Attack-only sizes: train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")
    logger.info(f"Attack classes in training:\n{y_train.value_counts().to_string()}")

    # Guard: minimum viable dataset
    if len(X_train) < 50:
        logger.error("Training set has fewer than 50 attack records. Cannot train reliably.")
        sys.exit(1)

    n_attack_classes = y_train.nunique()
    if n_attack_classes < 2:
        logger.error(f"Only {n_attack_classes} attack class(es) found. Multiclass requires >= 2.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Train XGBoost
    # ------------------------------------------------------------------
    logger.info(
        f"Step 6: Training XGBoostAttackClassifier "
        f"(n_estimators={args.n_estimators}, max_depth={args.max_depth})..."
    )
    classifier = XGBoostAttackClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=DEFAULT_RANDOM_STATE,
        early_stopping_rounds=20,
    )
    classifier.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # Validation quick-check
    val_preds = classifier.predict_label_names(X_val)
    from sklearn.metrics import f1_score as skf1
    val_f1 = skf1(y_val, val_preds, average="macro", zero_division=0)
    logger.info(f"Validation F1-macro: {val_f1:.4f}")

    # ------------------------------------------------------------------
    # 7. Test evaluation (held-out -- used ONLY here)
    # ------------------------------------------------------------------
    logger.info("Step 7: Evaluating on held-out test set...")
    test_preds = classifier.predict_label_names(X_test)
    test_proba = classifier.predict_proba(X_test)

    evaluator = AttackClassifierEvaluator()
    results = evaluator.evaluate(
        y_true_labels=list(y_test),
        y_pred_labels=test_preds,
        y_proba=test_proba,
        class_names=classifier.get_class_names(),
    )
    evaluator.print_report(results)

    # ------------------------------------------------------------------
    # 8. SHAP global explanation
    # ------------------------------------------------------------------
    if not args.skip_shap:
        logger.info(
            f"Step 8: Computing SHAP global explanations "
            f"(max_samples={args.shap_samples} from real test data)..."
        )
        shap_vals, X_shap_sample = classifier.explain_global(
            X_test, max_samples=args.shap_samples
        )
        shap_fig = evaluator.plot_shap_summary(
            shap_vals, X_shap_sample,
            class_names=classifier.get_class_names(),
            output_dir=FIGURES_DIR,
        )
        per_class_fig = evaluator.plot_per_class_shap(
            shap_vals, X_shap_sample,
            class_names=classifier.get_class_names(),
            output_dir=FIGURES_DIR,
        )
        logger.info(f"SHAP plots saved: {shap_fig}, {per_class_fig}")
    else:
        logger.info("Step 8: SHAP skipped (--skip-shap flag set).")

    # ------------------------------------------------------------------
    # 9. Save model and artifacts
    # ------------------------------------------------------------------
    logger.info("Step 9: Saving model artifacts...")
    model_save_path = MODELS_DIR / "attack_xgboost_classifier.joblib"
    class_map_path = MODELS_DIR / "attack_class_mapping.json"
    classifier.save(model_save_path, class_map_path)

    # Save confusion matrix plot
    cm_fig = evaluator.plot_confusion_matrix(results, FIGURES_DIR)
    logger.info(f"Confusion matrix saved: {cm_fig}")

    # Save evaluation JSON
    evaluator.save_results(results, REPORTS_DIR, prefix="attack_model")

    # Save feature importance
    importance_df = classifier.get_global_feature_importance()
    importance_path = REPORTS_DIR / "attack_xgboost_feature_importance.json"
    importance_df.to_json(importance_path, orient="records", indent=2)
    logger.info(f"Feature importance saved: {importance_path}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE. Summary:")
    logger.info(f"  Model:        {model_save_path}")
    logger.info(f"  Class map:    {class_map_path}")
    logger.info(f"  Test accuracy:  {results['accuracy']:.4f}")
    logger.info(f"  Test F1-macro:  {results['f1_macro']:.4f}")
    logger.info(f"  Test F1-wt:     {results['f1_weighted']:.4f}")
    roc = results.get("roc_auc_macro_ovr")
    if roc:
        logger.info(f"  ROC-AUC (OvR):  {roc:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
