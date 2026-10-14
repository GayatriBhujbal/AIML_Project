"""Project configuration and path definitions for CIC-IDS2017 pipeline."""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directories (configurable via environment variables)
RAW_DATA_DIR = Path(os.getenv("CIC_DATA_DIR", PROJECT_ROOT / "MachineLearningCVE"))
TRAFFIC_LABELLING_DIR = Path(os.getenv("CIC_RAW_FLOWS_DIR", PROJECT_ROOT / "TrafficLabelling"))

# Artifact directories
ARTIFACTS_DIR = Path(os.getenv("CIC_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
FIGURES_DIR = ARTIFACTS_DIR / "figures"
SCHEMAS_DIR = ARTIFACTS_DIR / "schemas"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Ensure artifact output directories exist
for d in [FIGURES_DIR, SCHEMAS_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Splitting Configuration
DEFAULT_RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.70"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.15"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.15"))

# ── Sampling configuration (BUG-3 / BUG-4 fix) ──────────────────────────────
# A single shared default used by prepare_dataset, run_feature_selection,
# train_binary_models, and train_attack_classifier so all scripts work on
# a consistent dataset sample.
#
# Taking the first N rows of each CSV creates temporal / class-distribution
# bias (e.g. Monday's file is 100 % BENIGN; Friday-DDos concentrates DDoS
# traffic in the first rows).  Set SAMPLE_STRATEGY to 'random' (default)
# to draw a reproducible random sample instead.  Use 'head' only if you
# need deterministic ordering for debugging.  Set to 0 to load the full file.
DEFAULT_SAMPLE_PER_FILE = int(os.getenv("SAMPLE_PER_FILE", "30000"))
SAMPLE_RANDOM_SEED      = int(os.getenv("SAMPLE_RANDOM_SEED", "42"))
SAMPLE_STRATEGY         = os.getenv("SAMPLE_STRATEGY", "random")  # 'random' | 'head'

# Feature Selection Thresholds
VARIANCE_THRESHOLD = 0.0  # Drops constant features
CORRELATION_THRESHOLD = 0.95  # Filters collinear feature pairs
DEFAULT_NUM_SELECTED_FEATURES = 25

# CSV Files Expected in MachineLearningCVE
EXPECTED_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]
