"""
Attack classifier evaluation module (Member 3).
Computes metrics from real held-out predictions only -- no fabricated results.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class AttackClassifierEvaluator:
    """Evaluate XGBoostAttackClassifier on held-out test data.

    All metrics are computed from real model predictions on the held-out
    test set. No metrics are fabricated or assumed.
    """

    def evaluate(
        self,
        y_true_labels: List[str],
        y_pred_labels: List[str],
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute full evaluation metrics from real held-out predictions.

        Args:
            y_true_labels: Ground-truth attack class names.
            y_pred_labels: Model-predicted attack class names.
            y_proba:       Probability matrix (N, n_classes). Used for ROC-AUC.
                           Pass None to skip ROC-AUC computation.
            class_names:   Ordered list of class names (must match y_proba columns).

        Returns:
            Dict with accuracy, precision, recall, f1 (macro/weighted),
            per-class report, confusion matrix, and ROC-AUC if applicable.
        """
        y_true = np.array(y_true_labels)
        y_pred = np.array(y_pred_labels)

        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        report_str = classification_report(y_true, y_pred, zero_division=0)
        report_dict = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )

        labels_in_test = sorted(list(set(y_true_labels) | set(y_pred_labels)))
        cm = confusion_matrix(y_true, y_pred, labels=labels_in_test).tolist()

        # ROC-AUC (multiclass OvR): only if proba and class_names provided
        roc_auc: Optional[float] = None
        roc_auc_note = ""
        if y_proba is not None and class_names is not None:
            try:
                # Filter to classes that appear in y_true
                present = sorted(set(y_true_labels))
                present_idx = [class_names.index(c) for c in present if c in class_names]
                y_proba_filt = y_proba[:, present_idx]
                # Normalize rows so they sum to 1 after filtering
                row_sums = y_proba_filt.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                y_proba_norm = y_proba_filt / row_sums

                if len(present) >= 2:
                    roc_auc = float(
                        roc_auc_score(
                            y_true,
                            y_proba_norm,
                            multi_class="ovr",
                            labels=present,
                            average="macro",
                        )
                    )
                    roc_auc_note = f"OvR macro, {len(present)} classes"
                else:
                    roc_auc_note = "Skipped: fewer than 2 classes in test set"
            except Exception as e:
                roc_auc_note = f"ROC-AUC computation failed: {e}"
                logger.warning(f"[evaluate] ROC-AUC error: {e}")

        results = {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "roc_auc_macro_ovr": roc_auc,
            "roc_auc_note": roc_auc_note,
            "classification_report_str": report_str,
            "classification_report_dict": report_dict,
            "confusion_matrix_labels": labels_in_test,
            "confusion_matrix": cm,
            "n_test_samples": len(y_true),
            "n_classes_in_test": len(labels_in_test),
        }

        logger.info(
            f"[evaluate] Test results: "
            f"acc={acc:.4f} | prec={prec_macro:.4f} | rec={rec_macro:.4f} | "
            f"f1_macro={f1_macro:.4f} | f1_weighted={f1_weighted:.4f} | "
            f"roc_auc={roc_auc}"
        )
        return results

    def print_report(self, results: Dict[str, Any]) -> None:
        """Print a formatted evaluation summary to stdout."""
        print("\n" + "=" * 72)
        print("ATTACK CLASSIFIER EVALUATION REPORT (XGBoost, Test Set)")
        print("=" * 72)
        print(f"  Test samples:     {results['n_test_samples']:,}")
        print(f"  Classes in test:  {results['n_classes_in_test']}")
        print(f"  Accuracy:         {results['accuracy']:.4f}")
        print(f"  Precision (macro):{results['precision_macro']:.4f}")
        print(f"  Recall (macro):   {results['recall_macro']:.4f}")
        print(f"  F1 (macro):       {results['f1_macro']:.4f}")
        print(f"  F1 (weighted):    {results['f1_weighted']:.4f}")
        if results["roc_auc_macro_ovr"] is not None:
            print(f"  ROC-AUC (OvR):   {results['roc_auc_macro_ovr']:.4f}  [{results['roc_auc_note']}]")
        elif results["roc_auc_note"]:
            print(f"  ROC-AUC:         N/A -- {results['roc_auc_note']}")
        print("\nPer-Class Classification Report:")
        print(results["classification_report_str"])

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        prefix: str = "attack_model",
    ) -> None:
        """Save evaluation JSON to artifacts/reports/."""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{prefix}_evaluation.json"
        save_data = {k: v for k, v in results.items() if k != "classification_report_str"}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=str)
        logger.info(f"[save_results] Evaluation JSON -> {json_path}")

    def plot_confusion_matrix(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        prefix: str = "attack_model",
        figsize: Tuple = (14, 11),
    ) -> Path:
        """Plot and save confusion matrix to artifacts/figures/."""
        cm = np.array(results["confusion_matrix"])
        labels = results["confusion_matrix_labels"]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            ylabel="True Label",
            xlabel="Predicted Label",
            title=f"XGBoost Attack Classifier -- Confusion Matrix (Test Set)\n"
                  f"Accuracy: {results['accuracy']:.4f}  |  F1-macro: {results['f1_macro']:.4f}",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)

        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7,
                )

        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{prefix}_confusion_matrix.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[plot_confusion_matrix] Saved -> {out_path}")
        return out_path

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X_sample: pd.DataFrame,
        class_names: List[str],
        output_dir: Path,
        prefix: str = "attack_model",
        top_n: int = 15,
    ) -> Path:
        """Plot and save mean absolute SHAP feature importance bar chart.

        Uses mean absolute SHAP values summed across all classes.
        This is a global summary -- per-class breakdowns are available via
        shap_values[:, :, class_idx].

        Args:
            shap_values: np.ndarray (N, n_features, n_classes)
            X_sample:    pd.DataFrame (N, 25) -- actual rows used for SHAP
            class_names: List of class names
            output_dir:  Directory to save the plot
            prefix:      File name prefix
            top_n:       Number of top features to display
        """
        import shap
        output_dir.mkdir(parents=True, exist_ok=True)

        # Mean absolute SHAP across all samples and classes
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))  # (n_features,)
        feat_names = list(X_sample.columns)

        sorted_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
        top_feats = [feat_names[i] for i in sorted_idx]
        top_vals = mean_abs_shap[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)))
        bars = ax.barh(top_feats[::-1], top_vals[::-1], color="#2196F3", edgecolor="white")
        ax.set_xlabel("Mean |SHAP value| (summed over classes)")
        ax.set_title(
            f"Global SHAP Feature Importance -- XGBoost Attack Classifier\n"
            f"Top {top_n} features (N={len(X_sample)} samples from real test data)"
        )
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        fig.tight_layout()
        out_path = output_dir / f"{prefix}_shap_global_importance.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[plot_shap_summary] Saved -> {out_path}")
        return out_path

    def plot_per_class_shap(
        self,
        shap_values: np.ndarray,
        X_sample: pd.DataFrame,
        class_names: List[str],
        output_dir: Path,
        prefix: str = "attack_model",
        top_n: int = 10,
    ) -> Path:
        """Plot per-class SHAP importance bar charts in a grid layout."""
        output_dir.mkdir(parents=True, exist_ok=True)
        n_classes = len(class_names)
        ncols = min(3, n_classes)
        nrows = (n_classes + ncols - 1) // ncols
        feat_names = list(X_sample.columns)

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.array(axes).flatten() if n_classes > 1 else [axes]

        for c_idx, cname in enumerate(class_names):
            ax = axes[c_idx]
            if shap_values.ndim == 3:
                sv_class = shap_values[:, :, c_idx]
            else:
                sv_class = shap_values
            mean_abs = np.abs(sv_class).mean(axis=0)
            sorted_idx = np.argsort(mean_abs)[::-1][:top_n]
            top_feats = [feat_names[i] for i in sorted_idx]
            top_vals = mean_abs[sorted_idx]

            ax.barh(top_feats[::-1], top_vals[::-1], color="#E91E63", edgecolor="white")
            ax.set_title(cname, fontsize=9, fontweight="bold")
            ax.set_xlabel("Mean |SHAP|", fontsize=8)
            ax.tick_params(labelsize=7)

        # Hide unused subplots
        for j in range(n_classes, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Per-Class SHAP Feature Importance -- XGBoost Attack Classifier",
            fontsize=12, y=1.01
        )
        fig.tight_layout()
        out_path = output_dir / f"{prefix}_shap_per_class.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[plot_per_class_shap] Saved -> {out_path}")
        return out_path
