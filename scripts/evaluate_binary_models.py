"""CLI script to load saved binary threat detection artifacts and evaluate on input data."""
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.dataset_config import MODELS_DIR, RAW_DATA_DIR
from src.dataset.loader import DatasetLoader
from src.models.binary_detector import RandomForestBinaryDetector, DecisionTreeBaseline
from src.models.evaluator import ModelEvaluator
from src.preprocessing.pipeline import PreprocessingPipeline
from src.dataset.schema import TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained binary models on a CSV file.")
    parser.add_argument("--test-file", type=str, default="", help="Path to CSV file. Defaults to Friday DDoS file.")
    parser.add_argument("--nrows", type=int, default=10000, help="Number of rows to test (default: 10,000).")
    args = parser.parse_args()

    # Determine input file
    if args.test_file:
        test_path = Path(args.test_file)
    else:
        test_path = RAW_DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

    logger.info(f"Loading test data from '{test_path}' (nrows={args.nrows})...")
    loader = DatasetLoader(RAW_DATA_DIR)
    raw_df = loader.load_file(test_path, nrows=args.nrows, normalize_labels=True)

    # Separate features and target
    y_raw = raw_df[TARGET_COLUMN]
    X_raw = raw_df.drop(columns=[TARGET_COLUMN])
    y_test = PreprocessingPipeline.encode_labels(y_raw, mode="binary")

    # Load fitted pipeline and models
    pipeline = PreprocessingPipeline.load(MODELS_DIR / "preprocessor.joblib")
    dt_model = DecisionTreeBaseline.load(MODELS_DIR / "binary_decision_tree.joblib")
    rf_model = RandomForestBinaryDetector.load(MODELS_DIR / "binary_random_forest.joblib")

    X_test = pipeline.transform(X_raw)

    evaluator = ModelEvaluator()
    dt_res = evaluator.evaluate_model("Decision Tree Baseline", dt_model, X_test, y_test)
    rf_res = evaluator.evaluate_model("Random Forest Detector", rf_model, X_test, y_test)

    print("\n" + "=" * 60)
    print("EVALUATION ON INPUT TEST DATA:")
    print("=" * 60)
    print(f"File: {test_path.name} ({len(X_test):,} records)")
    print(f"Decision Tree -> Acc: {dt_res['accuracy']:.4f}, F1: {dt_res['f1_attack']:.4f}, Rec: {dt_res['recall_attack']:.4f}")
    print(f"Random Forest -> Acc: {rf_res['accuracy']:.4f}, F1: {rf_res['f1_attack']:.4f}, Rec: {rf_res['recall_attack']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
