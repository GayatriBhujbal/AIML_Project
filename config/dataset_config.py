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
