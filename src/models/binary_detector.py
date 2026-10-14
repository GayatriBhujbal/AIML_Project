"""Binary threat detection models: Decision Tree baseline and Random Forest detector."""
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config.dataset_config import DEFAULT_RANDOM_STATE

logger = logging.getLogger(__name__)


class DecisionTreeBaseline:
    """
    Interpretable Decision Tree baseline classifier for binary threat detection (BENIGN vs ATTACK).
    Serves as an empirical baseline to justify ensemble complexity.
    """

    def __init__(
        self,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        criterion: str = "gini",
        class_weight: Optional[str] = "balanced",
        random_state: int = DEFAULT_RANDOM_STATE
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.class_weight = class_weight
        self.random_state = random_state

        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        self.is_fitted_: bool = False
        self.feature_names_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "DecisionTreeBaseline":
        """Fit Decision Tree baseline on training features and binary labels."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        logger.info(f"Fitting DecisionTreeBaseline (max_depth={self.max_depth}, class_weight={self.class_weight})...")
        self.model.fit(X, y)
        self.is_fitted_ = True
        logger.info(f"DecisionTree fitted: depth={self.model.get_depth()}, leaves={self.model.get_n_leaves()}.")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict binary class labels (0: BENIGN, 1: ATTACK)."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict().")
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities: [P(BENIGN), P(ATTACK)]."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict_proba().")
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> Dict[str, float]:
        """Return dictionary of feature importances."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted to access feature importances.")
        return dict(zip(self.feature_names_, self.model.feature_importances_))

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize trained model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved DecisionTreeBaseline to '{path}'.")
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DecisionTreeBaseline":
        """Load trained model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at '{path}'.")
        loaded = joblib.load(path)
        logger.info(f"Loaded DecisionTreeBaseline from '{path}'.")
        return loaded


class RandomForestBinaryDetector:
    """
    Production-grade Random Forest binary classifier for network threat detection.
    Supports balanced class weighting, parallel training, and calibrated decision thresholding.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 16,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        class_weight: Optional[str] = "balanced",
        decision_threshold: float = 0.50,
        n_jobs: int = -1,
        random_state: int = DEFAULT_RANDOM_STATE
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.decision_threshold = decision_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.is_fitted_: bool = False
        self.feature_names_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "RandomForestBinaryDetector":
        """Fit Random Forest on training features and binary labels."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        logger.info(
            f"Fitting RandomForestBinaryDetector (n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, class_weight={self.class_weight})..."
        )
        self.model.fit(X, y)
        self.is_fitted_ = True
        logger.info("RandomForestBinaryDetector fitted successfully.")
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities: [P(BENIGN), P(ATTACK)]."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict_proba().")
        return self.model.predict_proba(X)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict binary labels applying the decision threshold to P(ATTACK).
        
        Args:
            X: Input feature matrix
            threshold: Optional custom decision threshold (defaults to self.decision_threshold)
            
        Returns:
            np.ndarray of binary predictions (0: BENIGN, 1: ATTACK)
        """
        thresh = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X)
        # Class 1 is ATTACK
        attack_probs = probs[:, 1]
        return (attack_probs >= thresh).astype(int)

    def set_threshold(self, threshold: float) -> None:
        """Set decision threshold calibrated on validation data."""
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"Threshold must be strictly between 0 and 1. Given: {threshold}")
        self.decision_threshold = threshold
        logger.info(f"Decision threshold updated to {threshold:.4f}.")

    @property
    def feature_importances_(self) -> Dict[str, float]:
        """Return dictionary of Gini feature importances."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted to access feature importances.")
        return dict(zip(self.feature_names_, self.model.feature_importances_))

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize trained model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved RandomForestBinaryDetector to '{path}'.")
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "RandomForestBinaryDetector":
        """Load trained model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at '{path}'.")
        loaded = joblib.load(path)
        logger.info(f"Loaded RandomForestBinaryDetector from '{path}'.")
        return loaded
