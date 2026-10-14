"""
XGBoost Multiclass Attack-Category Classifier for CIC-IDS2017 (Member 3).

Design coordination:
- Feature schema: 25 features from artifacts/schemas/selected_features.json (Member 1).
- Input X: pre-processed DataFrame from PreprocessingPipeline.transform() -- do NOT re-fit.
- Input source: ThreatDetectionInferenceEngine.route_to_attack_classifier() (Member 2).
- Splits: 70/15/15 stratified, random_state=42 (coordinated with Members 1 and 2).
- BENIGN excluded from training: two-stage pipeline design.
- Class mapping derived dynamically from labels present in the actual dataset files.
- Class imbalance: inverse-frequency sample_weight (documented in fit() docstring).
- SHAP: TreeExplainer on actual trained model; sampling documented in explain_global().
- Interface for Member 4: get_xgb_booster(), explain_global(), explain_local().
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class XGBoostAttackClassifier:
    """Multiclass XGBoost classifier for fine-grained attack-category prediction.

    Input contract (from Member 2):
        X: pd.DataFrame (N, 25) -- preprocessed attack records only.
        y: pd.Series of raw string attack labels.

    Output contract (for Member 4):
        get_xgb_booster()     -- xgb.Booster for shap.TreeExplainer
        get_feature_names()   -- ordered 25-feature list
        get_class_names()     -- ordered attack class name list
        explain_global()      -- SHAP values + sample DataFrame
        explain_local()       -- per-record SHAP values
        predict_with_scores() -- (preds, proba, class_names)
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        eval_metric: str = "mlogloss",
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int = 20,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds

        self.model = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.class_names_: List[str] = []
        self.feature_names_: List[str] = []
        self.n_classes_: int = 0
        self.class_weight_map_: Dict[int, float] = {}
        self.is_fitted_: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostAttackClassifier":
        """Train XGBoost multiclass classifier on attack-only records.

        Class imbalance approach (documented):
            Uses inverse-frequency sample_weight = total / (n_classes * class_count).
            Rationale: XGBClassifier does not support class_weight for multiclass;
            scale_pos_weight is binary-only. Per-sample weights give XGBoost
            gradient-level control, allowing ultra-rare classes (Heartbleed=11 rows,
            Infiltration=36 rows) to receive proportionally higher gradient signals.
            Weight range is logged for audit.
        """
        import xgboost as xgb

        self.feature_names_ = list(X_train.columns)

        benign_mask = y_train.str.strip().str.upper() == "BENIGN"
        if benign_mask.any():
            logger.warning(
                f"[fit] {benign_mask.sum()} BENIGN rows in training data -- excluding. "
                "Should be filtered by binary stage."
            )
            X_train = X_train[~benign_mask]
            y_train = y_train[~benign_mask]

        if X_val is not None and y_val is not None:
            val_benign = y_val.str.strip().str.upper() == "BENIGN"
            if val_benign.any():
                X_val = X_val[~val_benign]
                y_val = y_val[~val_benign]

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y_train)
        self.class_names_ = list(self.label_encoder_.classes_)
        self.n_classes_ = len(self.class_names_)
        y_train_enc = self.label_encoder_.transform(y_train)

        logger.info(
            f"[fit] {self.n_classes_} attack classes in training: {self.class_names_}"
        )

        class_counts = np.bincount(y_train_enc, minlength=self.n_classes_)
        inv_freq = len(y_train_enc) / (self.n_classes_ * class_counts.astype(float))
        self.class_weight_map_ = {i: float(w) for i, w in enumerate(inv_freq)}
        sample_weight = np.array([inv_freq[c] for c in y_train_enc])
        logger.info(
            f"[fit] Inverse-freq sample_weight range: "
            f"[{sample_weight.min():.2f}, {sample_weight.max():.2f}]"
        )

        objective = "multi:softprob" if self.n_classes_ > 2 else "binary:logistic"

        # XGBoost 3.x: early_stopping_rounds is a constructor param, not fit() kwarg.
        # use_label_encoder was removed in XGBoost 1.6+.
        # num_class is auto-detected from num_class in the objective for softprob.
        xgb_params: Dict[str, Any] = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=objective,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )

        # Build early_stopping model if validation set is provided
        has_val = X_val is not None and y_val is not None
        y_val_known = None
        X_val_k = None
        y_val_enc = None

        if has_val:
            y_val_known = y_val[y_val.isin(self.class_names_)]
            X_val_k = X_val.loc[y_val_known.index]
            if len(y_val_known) > 0:
                y_val_enc = self.label_encoder_.transform(y_val_known)
                xgb_params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = xgb.XGBClassifier(**xgb_params)

        fit_kwargs: Dict[str, Any] = {"sample_weight": sample_weight}
        if y_val_enc is not None:
            fit_kwargs["eval_set"] = [(X_val_k, y_val_enc)]
            fit_kwargs["verbose"] = False

        self.model.fit(X_train, y_train_enc, **fit_kwargs)
        self.is_fitted_ = True
        best = getattr(self.model, "best_iteration", self.n_estimators)
        logger.info(f"[fit] Fitted. Best iteration: {best}. n_classes: {self.n_classes_}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict contiguous integer class codes [0, n_attack_classes-1]."""
        self._check_fitted()
        return self.model.predict(self._align(X))

    def predict_label_names(self, X: pd.DataFrame) -> List[str]:
        """Predict attack category as human-readable string labels."""
        return self.label_encoder_.inverse_transform(self.predict(X)).tolist()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probability matrix (N, n_attack_classes)."""
        self._check_fitted()
        return self.model.predict_proba(self._align(X))

    def predict_with_scores(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return (predictions, probabilities, class_names) for Member 4 integration."""
        self._check_fitted()
        Xa = self._align(X)
        return self.model.predict(Xa), self.model.predict_proba(Xa), self.class_names_

    def explain_global(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute global SHAP feature importance values.

        Sampling policy (documented):
            If len(X) > max_samples, a random subsample of max_samples rows is
            drawn WITHOUT replacement from the actual evaluated data (real records,
            no synthetic data). Seed is fixed and logged. Default max_samples=500
            balances SHAP accuracy vs compute time for the ~87K CIC-IDS2017 attack rows.

        Returns:
            shap_values: np.ndarray (N_sample, n_features, n_classes)
            X_sample:    pd.DataFrame (N_sample, 25) -- rows used for SHAP
        """
        import shap
        self._check_fitted()
        Xa = self._align(X)

        if len(Xa) > max_samples:
            logger.info(
                f"[explain_global] Subsampling {max_samples} of {len(Xa)} real rows "
                f"(seed={random_state}). No synthetic data used."
            )
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(Xa), size=max_samples, replace=False)
            X_sample = Xa.iloc[idx].reset_index(drop=True)
        else:
            X_sample = Xa.reset_index(drop=True)

        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        logger.info(f"[explain_global] SHAP shape: {sv.shape} (N, features, classes)")
        return sv, X_sample

    def explain_local(
        self, X_single: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate SHAP explanation for a single record.

        SHAP values show how each feature influences the model prediction --
        they do NOT claim features caused the attack.

        Returns:
            shap_vals:   np.ndarray (n_features, n_classes)
            base_values: np.ndarray (n_classes,)
            class_names: List[str]
        """
        import shap
        self._check_fitted()
        if len(X_single) != 1:
            raise ValueError(f"explain_local() needs 1 row, got {len(X_single)}.")
        Xa = self._align(X_single)
        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(Xa)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)[0]
        else:
            sv = sv[0]
        base = explainer.expected_value
        if not hasattr(base, "__len__"):
            base = np.array([base])
        return sv, np.asarray(base), self.class_names_

    def get_global_feature_importance(self) -> pd.DataFrame:
        """XGBoost built-in gain-based importance, sorted descending."""
        self._check_fitted()
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def get_xgb_booster(self):
        """Return underlying xgb.Booster for shap.TreeExplainer."""
        self._check_fitted()
        return self.model.get_booster()

    def get_feature_names(self) -> List[str]:
        self._check_fitted()
        return list(self.feature_names_)

    def get_class_names(self) -> List[str]:
        self._check_fitted()
        return list(self.class_names_)

    def get_class_mapping(self) -> Dict[int, str]:
        self._check_fitted()
        return {i: n for i, n in enumerate(self.class_names_)}

    def save(
        self,
        model_path: Union[str, Path],
        class_map_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Serialize classifier and class mapping to disk."""
        self._check_fitted()
        p = Path(model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info(f"[save] XGBoostAttackClassifier -> {p}")

        cmp = Path(class_map_path) if class_map_path else p.parent / "attack_class_mapping.json"
        with open(cmp, "w", encoding="utf-8") as f:
            json.dump({
                "n_classes": self.n_classes_,
                "class_names": self.class_names_,
                "int_to_class": {str(k): v for k, v in self.get_class_mapping().items()},
                "class_to_int": {v: k for k, v in self.get_class_mapping().items()},
                "feature_names": self.feature_names_,
                "model_path": str(p),
            }, f, indent=2)
        logger.info(f"[save] class mapping -> {cmp}")
        return p

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "XGBoostAttackClassifier":
        """Load fitted classifier from disk."""
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return joblib.load(p)

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("XGBoostAttackClassifier must be fitted first.")

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.feature_names_ if f not in X.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} required features: {missing[:5]}...")
        return X[self.feature_names_]
