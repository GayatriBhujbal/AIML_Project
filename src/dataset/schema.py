"""Canonical dataset schema, column definitions, and label mappings for CIC-IDS2017."""
import json
from pathlib import Path
from typing import Dict, List, Any

TARGET_COLUMN = "Label"

# The 85 columns in TrafficLabelling (GeneratedLabelledFlows)
RAW_TRAFFIC_LABELLING_COLUMNS: List[str] = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port",
    "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min", "Label"
]

# Raw flow identifiers that cause severe data leakage and memorization
IDENTIFIER_COLUMNS: List[str] = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Timestamp",
    "Protocol"
]

# The canonical 79 columns in MachineLearningCVE (whitespace stripped)
CANONICAL_COLUMNS: List[str] = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min", "Label"
]

# Exact duplicate columns in raw CICFlowMeter output
DUPLICATE_COLUMNS: List[str] = [
    "Fwd Header Length.1"  # 100% duplicate of 'Fwd Header Length' across all 2.83M rows
]

# Columns with zero variance (all 0s across all 2.83M rows)
CONSTANT_COLUMNS: List[str] = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate"
]

# Default excluded features prior to modeling
EXCLUDED_COLUMNS: List[str] = DUPLICATE_COLUMNS + CONSTANT_COLUMNS

# All numeric features after excluding target and duplicated/constant columns
NUMERIC_COLUMNS: List[str] = [
    c for c in CANONICAL_COLUMNS if c not in EXCLUDED_COLUMNS and c != TARGET_COLUMN
]

# Canonical multiclass label mapping (0 to 14)
LABEL_MAPPING: Dict[str, int] = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "PortScan": 2,
    "DDoS": 3,
    "DoS GoldenEye": 4,
    "FTP-Patator": 5,
    "SSH-Patator": 6,
    "DoS slowloris": 7,
    "DoS Slowhttptest": 8,
    "Bot": 9,
    "Web Attack - Brute Force": 10,
    "Web Attack - XSS": 11,
    "Infiltration": 12,
    "Web Attack - Sql Injection": 13,
    "Heartbleed": 14
}

INVERSE_LABEL_MAPPING: Dict[int, str] = {v: k for k, v in LABEL_MAPPING.items()}

# Binary classification mapping
BINARY_MAPPING: Dict[str, int] = {
    "BENIGN": 0,
    "ATTACK": 1
}

# High-level attack category mapping
ATTACK_CATEGORY_MAPPING: Dict[str, str] = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS/DDoS",
    "DoS GoldenEye": "DoS/DDoS",
    "DoS slowloris": "DoS/DDoS",
    "DoS Slowhttptest": "DoS/DDoS",
    "DDoS": "DoS/DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Bot": "Botnet",
    "Infiltration": "Infiltration/Exploit",
    "Heartbleed": "Infiltration/Exploit"
}


def clean_column_name(col: str) -> str:
    """Strip leading and trailing whitespace from column name."""
    return col.strip()


def normalize_label(label: str) -> str:
    """
    Standardize raw label string to canonical label name.
    Handles Unicode replacement character (U+FFFD) and cp1252 corrupted bytes in Web Attack labels.
    """
    if not isinstance(label, str):
        return str(label)
    
    clean = label.strip()
    
    # Handle Unicode replacement character and variants
    replacements = {
        "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
        "Web Attack \ufffd XSS": "Web Attack - XSS",
        "Web Attack \ufffd Sql Injection": "Web Attack - Sql Injection",
        "Web Attack ï¿½ Brute Force": "Web Attack - Brute Force",
        "Web Attack ï¿½ XSS": "Web Attack - XSS",
        "Web Attack ï¿½ Sql Injection": "Web Attack - Sql Injection",
        "Web Attack - Brute Force": "Web Attack - Brute Force",
        "Web Attack - XSS": "Web Attack - XSS",
        "Web Attack - Sql Injection": "Web Attack - Sql Injection"
    }
    
    return replacements.get(clean, clean)


def export_schemas_to_json(output_dir: Path) -> None:
    """Export schemas, label mappings, and feature metadata to JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    schema_metadata = {
        "dataset_name": "CIC-IDS2017",
        "total_raw_columns": len(CANONICAL_COLUMNS),
        "target_column": TARGET_COLUMN,
        "canonical_columns": CANONICAL_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "constant_columns": CONSTANT_COLUMNS,
        "duplicate_columns": DUPLICATE_COLUMNS,
        "identifier_columns_in_traffic_labelling": IDENTIFIER_COLUMNS,
        "numeric_features_count": len(NUMERIC_COLUMNS),
        "numeric_columns": NUMERIC_COLUMNS
    }
    
    label_metadata = {
        "binary_mapping": BINARY_MAPPING,
        "multiclass_mapping": LABEL_MAPPING,
        "inverse_multiclass_mapping": INVERSE_LABEL_MAPPING,
        "attack_category_mapping": ATTACK_CATEGORY_MAPPING,
        "total_classes": len(LABEL_MAPPING)
    }
    
    with open(output_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema_metadata, f, indent=2)
        
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_metadata, f, indent=2)


# Backwards compatibility alias
RAW_COLUMNS = CANONICAL_COLUMNS
