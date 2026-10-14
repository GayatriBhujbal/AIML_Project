"""
Adapter script: converts AIML Project-network CSVs (snake_case, 122 cols)
to the canonical 79-col CIC-IDS2017 format and runs the full pipeline.

Run:  python scripts/run_with_network_data.py
"""
import sys, logging, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Column mapping: snake_case -> canonical CIC-IDS2017 ──────────────────────
COL_MAP = {
    "dst_port":                   "Destination Port",
    "duration":                   "Flow Duration",
    "fwd_packets_count":          "Total Fwd Packets",
    "bwd_packets_count":          "Total Backward Packets",
    "fwd_total_payload_bytes":    "Total Length of Fwd Packets",
    "bwd_total_payload_bytes":    "Total Length of Bwd Packets",
    "fwd_payload_bytes_max":      "Fwd Packet Length Max",
    "fwd_payload_bytes_min":      "Fwd Packet Length Min",
    "fwd_payload_bytes_mean":     "Fwd Packet Length Mean",
    "fwd_payload_bytes_std":      "Fwd Packet Length Std",
    "bwd_payload_bytes_max":      "Bwd Packet Length Max",
    "bwd_payload_bytes_min":      "Bwd Packet Length Min",
    "bwd_payload_bytes_mean":     "Bwd Packet Length Mean",
    "bwd_payload_bytes_std":      "Bwd Packet Length Std",
    "bytes_rate":                 "Flow Bytes/s",
    "packets_rate":               "Flow Packets/s",
    "packets_IAT_mean":           "Flow IAT Mean",
    "packet_IAT_std":             "Flow IAT Std",
    "packet_IAT_max":             "Flow IAT Max",
    "packet_IAT_min":             "Flow IAT Min",
    "fwd_packets_IAT_total":      "Fwd IAT Total",
    "fwd_packets_IAT_mean":       "Fwd IAT Mean",
    "fwd_packets_IAT_std":        "Fwd IAT Std",
    "fwd_packets_IAT_max":        "Fwd IAT Max",
    "fwd_packets_IAT_min":        "Fwd IAT Min",
    "bwd_packets_IAT_total":      "Bwd IAT Total",
    "bwd_packets_IAT_mean":       "Bwd IAT Mean",
    "bwd_packets_IAT_std":        "Bwd IAT Std",
    "bwd_packets_IAT_max":        "Bwd IAT Max",
    "bwd_packets_IAT_min":        "Bwd IAT Min",
    "fwd_total_header_bytes":     "Fwd Header Length",
    "bwd_total_header_bytes":     "Bwd Header Length",
    "fwd_packets_rate":           "Fwd Packets/s",
    "bwd_packets_rate":           "Bwd Packets/s",
    "payload_bytes_min":          "Min Packet Length",
    "payload_bytes_max":          "Max Packet Length",
    "payload_bytes_mean":         "Packet Length Mean",
    "payload_bytes_std":          "Packet Length Std",
    "payload_bytes_variance":     "Packet Length Variance",
    "fin_flag_counts":            "FIN Flag Count",
    "syn_flag_counts":            "SYN Flag Count",
    "rst_flag_counts":            "RST Flag Count",
    "psh_flag_counts":            "PSH Flag Count",
    "ack_flag_counts":            "ACK Flag Count",
    "urg_flag_counts":            "URG Flag Count",
    "cwr_flag_counts":            "CWE Flag Count",
    "ece_flag_counts":            "ECE Flag Count",
    "down_up_rate":               "Down/Up Ratio",
    "avg_segment_size":           "Average Packet Size",
    "fwd_avg_segment_size":       "Avg Fwd Segment Size",
    "bwd_avg_segment_size":       "Avg Bwd Segment Size",
    "subflow_fwd_packets":        "Subflow Fwd Packets",
    "subflow_fwd_bytes":          "Subflow Fwd Bytes",
    "subflow_bwd_packets":        "Subflow Bwd Packets",
    "subflow_bwd_bytes":          "Subflow Bwd Bytes",
    "fwd_init_win_bytes":         "Init_Win_bytes_forward",
    "bwd_init_win_bytes":         "Init_Win_bytes_backward",
    "active_mean":                "Active Mean",
    "active_std":                 "Active Std",
    "active_max":                 "Active Max",
    "active_min":                 "Active Min",
    "idle_mean":                  "Idle Mean",
    "idle_std":                   "Idle Std",
    "idle_max":                   "Idle Max",
    "idle_min":                   "Idle Min",
    "label":                      "Label",
}

# Label normalisation: network CSV uses underscores, pipeline expects spaces/dashes
LABEL_MAP = {
    "DoS_Hulk":               "DoS Hulk",
    "PortScan":                "PortScan",
    "DDoS":                    "DDoS",
    "DoS_GoldenEye":           "DoS GoldenEye",
    "FTP-Patator":             "FTP-Patator",
    "SSH-Patator":             "SSH-Patator",
    "DoS_slowloris":           "DoS slowloris",
    "DoS_Slowhttptest":        "DoS Slowhttptest",
    "Bot":                     "Bot",
    "Web Attack-Brute Force":  "Web Attack - Brute Force",
    "Web Attack-XSS":          "Web Attack - XSS",
    "Web Attack-Sql Injection":"Web Attack - Sql Injection",
    "Infiltration":            "Infiltration",
    "Heartbleed":              "Heartbleed",
    "BENIGN":                  "BENIGN",
    "Benign":                  "BENIGN",
}

DATA_DIR = Path(r"C:\Users\Anuja Dhope\Downloads\AIML Project-network")

FILES = {
    "botnet_ares.csv":      "Bot",
    "ddos_loit.csv":        "DDoS",
    "dos_golden_eye.csv":   "DoS GoldenEye",
    "dos_hulk.csv":         "DoS Hulk",
    "dos_slowhttptest.csv": "DoS Slowhttptest",
    "dos_slowloris.csv":    "DoS slowloris",
    "friday_benign.csv":    "BENIGN",
    "ftp_patator.csv":      "FTP-Patator",
    "heartbleed.csv":       "Heartbleed",
    "monday_benign.csv":    "BENIGN",
    "portscan.csv":         "PortScan",
    "ssh_patator-new.csv":  "SSH-Patator",
    "thursday_benign.csv":  "BENIGN",
    "tuesday_benign.csv":   "BENIGN",
    "web_brute_force.csv":  "Web Attack - Brute Force",
    "web_sql_injection.csv":"Web Attack - Sql Injection",
    "web_xss.csv":          "Web Attack - XSS",
    "wednesday_benign.csv": "BENIGN",
}

SAMPLE_PER_FILE = 5000   # rows per file (memory-safe)


def load_and_remap(filepath: Path, expected_label: str, nrows: int) -> pd.DataFrame:
    df = pd.read_csv(filepath, nrows=nrows, low_memory=False)
    # Keep only mappable cols + label
    keep = {k: v for k, v in COL_MAP.items() if k in df.columns}
    df = df[list(keep.keys())].rename(columns=keep)
    # Normalize label
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).map(
            lambda x: LABEL_MAP.get(x, LABEL_MAP.get(x.strip(), expected_label))
        )
    else:
        df["Label"] = expected_label
    return df


def main():
    from config.dataset_config import MODELS_DIR, FIGURES_DIR, REPORTS_DIR
    from src.preprocessing.pipeline import PreprocessingPipeline
    from src.preprocessing.splitter import DataSplitter
    from src.dataset.schema import normalize_label, TARGET_COLUMN
    from src.models.attack_classifier import XGBoostAttackClassifier
    from src.models.attack_evaluator import AttackClassifierEvaluator

    # ── 1. Load and combine ────────────────────────────────────────────────
    logger.info("Step 1: Loading network CSV files...")
    dfs = []
    for fname, lbl in FILES.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            logger.warning(f"Skipping missing file: {fname}")
            continue
        df_part = load_and_remap(fpath, lbl, SAMPLE_PER_FILE)
        dfs.append(df_part)
        logger.info(f"  {fname}: {len(df_part):,} rows | label={lbl}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined: {len(combined):,} rows | {len(combined.columns)} cols")
    logger.info("Label distribution:\n" + combined["Label"].value_counts().to_string())

    # ── 2. Load shared preprocessor ────────────────────────────────────────
    logger.info("\nStep 2: Loading shared PreprocessingPipeline...")
    pipeline = PreprocessingPipeline.load(MODELS_DIR / "preprocessor.joblib")
    logger.info(f"  Output features: {pipeline.feature_names_out_}")

    # ── 3. Split 70/15/15 ──────────────────────────────────────────────────
    logger.info("\nStep 3: Splitting 70/15/15 (stratified, random_state=42)...")
    splitter = DataSplitter(random_state=42)
    train_df, val_df, test_df, report = splitter.split_dataframe(combined, reset_index=True)
    logger.info(f"  Train={report['train_size']:,} | Val={report['val_size']:,} | Test={report['test_size']:,}")

    # ── 4. Preprocess ──────────────────────────────────────────────────────
    logger.info("\nStep 4: Preprocessing each split...")
    def preproc(df):
        y = df["Label"].apply(normalize_label)
        X = pipeline.transform(df.drop(columns=["Label"]))
        return X, y

    X_train_all, y_train_all = preproc(train_df)
    X_val_all,   y_val_all   = preproc(val_df)
    X_test_all,  y_test_all  = preproc(test_df)

    # ── 5. Filter attack-only ──────────────────────────────────────────────
    logger.info("\nStep 5: Filtering attack-only records...")
    def attacks(X, y):
        m = y.str.upper() != "BENIGN"
        return X[m].reset_index(drop=True), y[m].reset_index(drop=True)

    X_train, y_train = attacks(X_train_all, y_train_all)
    X_val,   y_val   = attacks(X_val_all,   y_val_all)
    X_test,  y_test  = attacks(X_test_all,  y_test_all)

    logger.info(f"  Attack train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")
    logger.info("  Attack class counts (train):\n" + y_train.value_counts().to_string())

    # ── 6. Train XGBoost ──────────────────────────────────────────────────
    logger.info("\nStep 6: Training XGBoostAttackClassifier...")
    clf = XGBoostAttackClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=15,
    )
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # ── 7. Evaluate on test set ────────────────────────────────────────────
    logger.info("\nStep 7: Evaluating on held-out test set...")
    test_preds  = clf.predict_label_names(X_test)
    test_proba  = clf.predict_proba(X_test)

    evaluator = AttackClassifierEvaluator()
    results = evaluator.evaluate(
        list(y_test), test_preds,
        y_proba=test_proba,
        class_names=clf.get_class_names(),
    )
    evaluator.print_report(results)

    # ── 8. SHAP global ────────────────────────────────────────────────────
    logger.info("\nStep 8: Computing SHAP global explanations (500 real samples)...")
    shap_vals, X_shap = clf.explain_global(X_test, max_samples=500)
    evaluator.plot_shap_summary(shap_vals, X_shap, clf.get_class_names(), FIGURES_DIR)
    evaluator.plot_per_class_shap(shap_vals, X_shap, clf.get_class_names(), FIGURES_DIR)

    # ── 9. SHAP local (first attack record) ───────────────────────────────
    logger.info("\nStep 9: Local SHAP for first test record...")
    sv, base, names = clf.explain_local(X_test.iloc[:1])
    pred_class = clf.predict_label_names(X_test.iloc[:1])[0]
    pred_code  = clf.predict(X_test.iloc[:1])[0]
    logger.info(f"  Predicted class: {pred_class} (code {pred_code})")
    logger.info(f"  Top 5 features by |SHAP| for predicted class:")
    sv_pred = sv[:, pred_code]
    top5_idx = np.argsort(np.abs(sv_pred))[::-1][:5]
    feats = clf.get_feature_names()
    for i in top5_idx:
        logger.info(f"    {feats[i]}: SHAP={sv_pred[i]:.4f}  (influences '{pred_class}' prediction)")

    # ── 10. Save all artifacts ────────────────────────────────────────────
    logger.info("\nStep 10: Saving artifacts...")
    clf.save(MODELS_DIR / "attack_xgboost_classifier.joblib",
             MODELS_DIR / "attack_class_mapping.json")
    evaluator.plot_confusion_matrix(results, FIGURES_DIR)
    evaluator.save_results(results, REPORTS_DIR, prefix="attack_model")
    clf.get_global_feature_importance().to_json(
        REPORTS_DIR / "attack_xgboost_feature_importance.json",
        orient="records", indent=2
    )

    logger.info("\n" + "="*60)
    logger.info("PROJECT RUN COMPLETE")
    logger.info(f"  Classes trained: {clf.get_class_names()}")
    logger.info(f"  Test Accuracy:   {results['accuracy']:.4f}")
    logger.info(f"  Test F1-macro:   {results['f1_macro']:.4f}")
    logger.info(f"  Test F1-weighted:{results['f1_weighted']:.4f}")
    roc = results.get("roc_auc_macro_ovr")
    if roc: logger.info(f"  ROC-AUC (OvR):  {roc:.4f}")
    logger.info(f"  Model saved:     {MODELS_DIR}/attack_xgboost_classifier.joblib")
    logger.info(f"  Figures saved:   {FIGURES_DIR}")
    logger.info(f"  Reports saved:   {REPORTS_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
