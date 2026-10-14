"""Visualization module for Exploratory Data Analysis on CIC-IDS2017."""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.dataset_config import FIGURES_DIR
from src.dataset.schema import (
    TARGET_COLUMN,
    ATTACK_CATEGORY_MAPPING,
    normalize_label
)

logger = logging.getLogger(__name__)

# Configure matplotlib styling for clean, professional academic presentation
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class EDAVisualizer:
    """
    Generates real, publication-grade figures from actual CIC-IDS2017 data.
    Never fabricates values or placeholder charts.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_class_distribution(
        self,
        df: pd.DataFrame,
        filename: str = "class_distribution.png"
    ) -> Path:
        """Plot actual class counts across all 15 classes with log scale to expose minority classes."""
        out_path = self.output_dir / filename
        logger.info(f"Generating class distribution plot -> '{out_path}'...")

        counts = df[TARGET_COLUMN].apply(normalize_label).value_counts()

        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.barh(counts.index[::-1], counts.values[::-1], color="#1f77b4", edgecolor="#0e3a59", alpha=0.85)

        ax.set_xscale("log")
        ax.set_xlabel("Flow Count (Logarithmic Scale)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Network Traffic Class", fontsize=12, fontweight="bold")
        ax.set_title("CIC-IDS2017 Network Threat Class Distribution (Actual Data)", fontsize=14, fontweight="bold", pad=15)

        # Label counts on bars
        for bar in bars:
            w = bar.get_width()
            ax.text(w * 1.15, bar.get_y() + bar.get_height() / 2, f"{int(w):,}",
                    va="center", ha="left", fontsize=9, color="#222222")

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_attack_categories(
        self,
        df: pd.DataFrame,
        filename: str = "attack_categories.png"
    ) -> Path:
        """Plot distribution aggregated into high-level attack families."""
        out_path = self.output_dir / filename
        logger.info(f"Generating attack category plot -> '{out_path}'...")

        norm_labels = df[TARGET_COLUMN].apply(normalize_label)
        categories = norm_labels.map(ATTACK_CATEGORY_MAPPING).value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
        bars = ax.bar(categories.index, categories.values, color=colors[:len(categories)], edgecolor="#333333", alpha=0.85)

        ax.set_yscale("log")
        ax.set_xlabel("Threat Category", fontsize=12, fontweight="bold")
        ax.set_ylabel("Total Flow Records (Log Scale)", fontsize=12, fontweight="bold")
        ax.set_title("Distribution by Attack Family (CIC-IDS2017)", fontsize=14, fontweight="bold", pad=15)
        plt.xticks(rotation=25, ha="right")

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.2, f"{int(h):,}",
                    ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_missing_and_inf_analysis(
        self,
        nan_counts: Dict[str, int],
        inf_counts: Dict[str, int],
        total_rows: int,
        filename: str = "missing_and_inf_analysis.png"
    ) -> Path:
        """Plot actual missing (NaN) and infinite (Inf) values detected in the dataset."""
        out_path = self.output_dir / filename
        logger.info(f"Generating missing and infinite values plot -> '{out_path}'...")

        labels = ["Flow Bytes/s (NaN)", "Flow Bytes/s (Inf)", "Flow Packets/s (Inf)"]
        values = [
            nan_counts.get("Flow Bytes/s", 0),
            inf_counts.get("Flow Bytes/s", 0),
            inf_counts.get("Flow Packets/s", 0)
        ]
        pcts = [(v / total_rows) * 100 for v in values]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, values, color=["#e74c3c", "#e67e22", "#f39c12"], edgecolor="#333333", alpha=0.85)

        ax.set_ylabel("Occurrence Count", fontsize=12, fontweight="bold")
        ax.set_title(f"Missing (NaN) & Infinite (Inf) Anomaly Analysis\n(Total Records: {total_rows:,})",
                     fontsize=13, fontweight="bold", pad=15)

        for bar, pct in zip(bars, pcts):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 30, f"{int(h):,} ({pct:.3f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylim(0, max(values) * 1.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        features: List[str],
        filename: str = "correlation_heatmap.png"
    ) -> Path:
        """Plot correlation matrix of selected features from actual data."""
        out_path = self.output_dir / filename
        logger.info(f"Generating correlation heatmap -> '{out_path}'...")

        valid_feats = [f for f in features if f in df.columns]
        sub_df = df[valid_feats].replace([np.inf, -np.inf], np.nan).dropna()
        corr = sub_df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr,
            cmap="vlag",
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Pearson Correlation Coefficient"}
        )
        ax.set_title("Correlation Matrix of Key Network Flow Features", fontsize=14, fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_feature_distributions_by_class(
        self,
        df: pd.DataFrame,
        features: List[str],
        filename: str = "feature_distributions_by_class.png"
    ) -> Path:
        """Plot comparative distributions of key features for Benign vs Attack traffic."""
        out_path = self.output_dir / filename
        logger.info(f"Generating feature distributions plot -> '{out_path}'...")

        valid_feats = [f for f in features if f in df.columns][:4]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        is_attack = (df[TARGET_COLUMN].apply(normalize_label) != "BENIGN")

        for i, feat in enumerate(valid_feats):
            ax = axes[i]
            benign_vals = df.loc[~is_attack, feat].replace([np.inf, -np.inf], np.nan).dropna()
            attack_vals = df.loc[is_attack, feat].replace([np.inf, -np.inf], np.nan).dropna()

            # Robust log-transform for highly skewed network metrics
            q99 = max(benign_vals.quantile(0.99), attack_vals.quantile(0.99))
            b_plot = benign_vals[benign_vals <= q99]
            a_plot = attack_vals[attack_vals <= q99]

            sns.histplot(b_plot, ax=ax, color="#1f77b4", label="BENIGN", kde=True, stat="density", alpha=0.4, bins=35)
            sns.histplot(a_plot, ax=ax, color="#d62728", label="ATTACK", kde=True, stat="density", alpha=0.4, bins=35)

            ax.set_title(f"Distribution: {feat}", fontsize=11, fontweight="bold")
            ax.set_xlabel(f"{feat} (capped at 99th percentile)", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.legend()

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_rf_feature_importance(
        self,
        feature_importances: Dict[str, float],
        top_n: int = 25,
        filename: str = "feature_importance.png"
    ) -> Path:
        """Plot top N features by Random Forest Gini impurity importance."""
        out_path = self.output_dir / filename
        logger.info(f"Generating feature importance plot -> '{out_path}'...")

        sorted_items = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [x[0] for x in sorted_items][::-1]
        scores = [x[1] for x in sorted_items][::-1]

        fig, ax = plt.subplots(figsize=(12, 9))
        bars = ax.barh(features, scores, color="#2ca02c", edgecolor="#1a5e1a", alpha=0.85)

        ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Flow Feature", fontsize=12, fontweight="bold")
        ax.set_title(f"Top {top_n} Network Flow Features (Random Forest on Training Data)",
                     fontsize=14, fontweight="bold", pad=15)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f}",
                    va="center", ha="left", fontsize=9)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path
