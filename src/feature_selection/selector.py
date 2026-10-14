"""Systematic feature selection and justification for CIC-IDS2017."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import time

from config.dataset_config import (
    SCHEMAS_DIR,
    REPORTS_DIR,
    DEFAULT_NUM_SELECTED_FEATURES,
    CORRELATION_THRESHOLD,
    DEFAULT_RANDOM_STATE
)

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Performs data-driven feature selection using training data only.
    Combines:
    1. Zero-variance and near-zero variance elimination.
    2. Collinearity filtering (Pearson |r| > threshold).
    3. Random Forest Gini impurity importance ranking.
    4. Validation F1 score benchmarking before and after selection.
    """

    def __init__(
        self,
        correlation_threshold: float = CORRELATION_THRESHOLD,
        top_k: int = DEFAULT_NUM_SELECTED_FEATURES,
        random_state: int = DEFAULT_RANDOM_STATE
    ):
        self.correlation_threshold = correlation_threshold
        self.top_k = top_k
        self.random_state = random_state

        self.selected_features_: List[str] = []
        self.excluded_features_: Dict[str, str] = {}
        self.feature_importances_: Dict[str, float] = {}
        self.collinear_pairs_: List[Dict[str, Any]] = []
        self.benchmark_results_: Dict[str, Any] = {}

    def fit_and_select(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ) -> List[str]:
        """
        Execute feature selection process on training set.
        
        Args:
            X_train: Preprocessed numeric training features
            y_train: Binary or multiclass target labels
            X_val: Optional preprocessed validation features for benchmarking
            y_val: Optional validation target labels
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Starting feature selection on training set with {X_train.shape[1]} initial features...")
        current_features = list(X_train.columns)

        # 1. Zero and Near-Zero Variance Filtering
        variances = X_train.var()
        zero_var = variances[variances == 0].index.tolist()
        for col in zero_var:
            self.excluded_features_[col] = "Zero variance (constant feature on training data)"
            current_features.remove(col)
        logger.info(f"Removed {len(zero_var)} zero-variance features.")

        X_filtered = X_train[current_features].copy()

        # 2. Correlation Analysis and Redundancy Filtering
        corr_matrix = X_filtered.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        for col in upper.columns:
            high_corrs = upper[col][upper[col] > self.correlation_threshold]
            for partner, val in high_corrs.items():
                self.collinear_pairs_.append({
                    "feature_1": partner,
                    "feature_2": col,
                    "correlation": float(val)
                })

        logger.info(f"Identified {len(self.collinear_pairs_)} highly collinear feature pairs (|r| > {self.correlation_threshold}).")

        # 3. Random Forest Feature Importance Ranking
        logger.info("Training Random Forest on training data to compute Gini feature importances...")
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_filtered, y_train)

        importances = pd.Series(rf.feature_importances_, index=X_filtered.columns).sort_values(ascending=False)
        self.feature_importances_ = {k: float(v) for k, v in importances.items()}

        # 4. Filter Redundant Collinear Features (retain the one with higher RF importance)
        features_to_drop_collinear = set()
        for pair in sorted(self.collinear_pairs_, key=lambda x: -x["correlation"]):
            f1 = pair["feature_1"]
            f2 = pair["feature_2"]
            if f1 not in features_to_drop_collinear and f2 not in features_to_drop_collinear:
                # Keep feature with higher importance
                imp1 = self.feature_importances_.get(f1, 0.0)
                imp2 = self.feature_importances_.get(f2, 0.0)
                dropped = f2 if imp1 >= imp2 else f1
                kept = f1 if imp1 >= imp2 else f2
                features_to_drop_collinear.add(dropped)
                self.excluded_features_[dropped] = f"Collinear with '{kept}' (|r| = {pair['correlation']:.4f}) with lower tree importance"

        candidates = [f for f in importances.index if f not in features_to_drop_collinear]
        self.selected_features_ = candidates[:self.top_k]

        # Record remaining excluded features
        for f in importances.index:
            if f not in self.selected_features_ and f not in self.excluded_features_:
                self.excluded_features_[f] = "Excluded by top-k importance cutoff after collinearity pruning"

        logger.info(f"Selected {len(self.selected_features_)} features out of {X_train.shape[1]}.")

        # 5. Validation Benchmark (if validation data provided)
        if X_val is not None and y_val is not None:
            self._benchmark_selection(X_train, y_train, X_val, y_val)

        return self.selected_features_

    def _benchmark_selection(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ) -> None:
        """Benchmark validation performance with full features vs selected features."""
        logger.info("Benchmarking validation performance: Full Feature Set vs Selected Subset...")

        # Benchmark 1: Full features
        t0 = time.time()
        rf_full = RandomForestClassifier(n_estimators=30, max_depth=12, random_state=self.random_state, n_jobs=-1)
        rf_full.fit(X_train, y_train)
        train_time_full = time.time() - t0
        y_pred_full = rf_full.predict(X_val)

        full_metrics = {
            "num_features": X_train.shape[1],
            "train_time_sec": round(train_time_full, 3),
            "accuracy": round(float(accuracy_score(y_val, y_pred_full)), 5),
            "f1_score": round(float(f1_score(y_val, y_pred_full, average="macro")), 5),
            "precision": round(float(precision_score(y_val, y_pred_full, average="macro")), 5),
            "recall": round(float(recall_score(y_val, y_pred_full, average="macro")), 5)
        }

        # Benchmark 2: Selected features
        t0 = time.time()
        rf_sel = RandomForestClassifier(n_estimators=30, max_depth=12, random_state=self.random_state, n_jobs=-1)
        rf_sel.fit(X_train[self.selected_features_], y_train)
        train_time_sel = time.time() - t0
        y_pred_sel = rf_sel.predict(X_val[self.selected_features_])

        sel_metrics = {
            "num_features": len(self.selected_features_),
            "train_time_sec": round(train_time_sel, 3),
            "accuracy": round(float(accuracy_score(y_val, y_pred_sel)), 5),
            "f1_score": round(float(f1_score(y_val, y_pred_sel, average="macro")), 5),
            "precision": round(float(precision_score(y_val, y_pred_sel, average="macro")), 5),
            "recall": round(float(recall_score(y_val, y_pred_sel, average="macro")), 5)
        }

        self.benchmark_results_ = {
            "full_feature_model": full_metrics,
            "selected_feature_model": sel_metrics,
            "f1_improvement": round(sel_metrics["f1_score"] - full_metrics["f1_score"], 5),
            "training_speedup_ratio": round(full_metrics["train_time_sec"] / max(sel_metrics["train_time_sec"], 0.001), 2)
        }

        logger.info(
            f"Benchmark Results: Full ({full_metrics['num_features']} feats) F1={full_metrics['f1_score']} | "
            f"Selected ({sel_metrics['num_features']} feats) F1={sel_metrics['f1_score']} "
            f"(Speedup: {self.benchmark_results_['training_speedup_ratio']}x)"
        )

    def save_artifacts(
        self,
        schemas_dir: Optional[Path] = None,
        reports_dir: Optional[Path] = None
    ) -> None:
        """Save selected feature list and justification reports to disk."""
        s_dir = schemas_dir or SCHEMAS_DIR
        r_dir = reports_dir or REPORTS_DIR
        s_dir.mkdir(parents=True, exist_ok=True)
        r_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save selected feature list JSON
        selected_data = {
            "num_selected_features": len(self.selected_features_),
            "selected_features": self.selected_features_,
            "compatible_models": ["RandomForestClassifier", "XGBoostClassifier", "SHAP_Explainer"]
        }
        with open(s_dir / "selected_features.json", "w", encoding="utf-8") as f:
            json.dump(selected_data, f, indent=2)

        # 2. Save feature selection report JSON
        report_data = {
            "num_selected": len(self.selected_features_),
            "num_excluded": len(self.excluded_features_),
            "selected_features": self.selected_features_,
            "excluded_features_with_reasons": self.excluded_features_,
            "collinear_pairs_found": self.collinear_pairs_,
            "top_feature_importances": dict(list(self.feature_importances_.items())[:30]),
            "benchmark_comparison": self.benchmark_results_
        }
        with open(r_dir / "feature_selection_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Saved selected features and reports to '{s_dir}' and '{r_dir}'.")
