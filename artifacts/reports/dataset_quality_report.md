# CIC-IDS2017 Dataset Quality & Exploratory Data Analysis Report

**Total Records Evaluated:** 2,830,743
**Total CSV Files:** 8
**Canonical Columns:** 79

## 1. File Breakdown

| File Name | Record Count | Present Classes |
| :--- | :--- | :--- |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | DDoS (128,027), BENIGN (97,718) |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 | PortScan (158,930), BENIGN (127,537) |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 | BENIGN (189,067), Bot (1,966) |
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | BENIGN (529,918) |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 | BENIGN (288,566), Infiltration (36) |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | BENIGN (168,186), Web Attack - Brute Force (1,507), Web Attack - XSS (652), Web Attack - Sql Injection (21) |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | BENIGN (432,074), FTP-Patator (7,938), SSH-Patator (5,897) |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | BENIGN (440,031), DoS Hulk (231,073), DoS GoldenEye (10,293), DoS slowloris (5,796), DoS Slowhttptest (5,499), Heartbleed (11) |

## 2. Ground-Truth Class Distribution

| Traffic Class | Flow Count | Percentage |
| :--- | :--- | :--- |
| **BENIGN** | 2,273,097 | 80.3004% |
| **DoS Hulk** | 231,073 | 8.1630% |
| **PortScan** | 158,930 | 5.6144% |
| **DDoS** | 128,027 | 4.5227% |
| **DoS GoldenEye** | 10,293 | 0.3636% |
| **FTP-Patator** | 7,938 | 0.2804% |
| **SSH-Patator** | 5,897 | 0.2083% |
| **DoS slowloris** | 5,796 | 0.2048% |
| **DoS Slowhttptest** | 5,499 | 0.1943% |
| **Bot** | 1,966 | 0.0695% |
| **Web Attack - Brute Force** | 1,507 | 0.0532% |
| **Web Attack - XSS** | 652 | 0.0230% |
| **Infiltration** | 36 | 0.0013% |
| **Web Attack - Sql Injection** | 21 | 0.0007% |
| **Heartbleed** | 11 | 0.0004% |

## 3. Data Integrity & Anomaly Analysis

### Missing Values (NaN)
- `Flow Bytes/s`: 1,358 missing rows (0.0480%)

### Infinite Values (Inf / -Inf)
- `Flow Bytes/s`: 1,509 infinite values (0.0533%)
- `Flow Packets/s`: 2,867 infinite values (0.1013%)

*Root cause:* `Flow Duration = 0` causes division by zero in packet/byte rate calculations.

### Zero-Variance (Constant) Features
- `Bwd PSH Flags`: Value is constant 0 across all 2,830,743 records.
- `Bwd URG Flags`: Value is constant 0 across all 2,830,743 records.
- `Fwd Avg Bytes/Bulk`: Value is constant 0 across all 2,830,743 records.
- `Fwd Avg Packets/Bulk`: Value is constant 0 across all 2,830,743 records.
- `Fwd Avg Bulk Rate`: Value is constant 0 across all 2,830,743 records.
- `Bwd Avg Bytes/Bulk`: Value is constant 0 across all 2,830,743 records.
- `Bwd Avg Packets/Bulk`: Value is constant 0 across all 2,830,743 records.
- `Bwd Avg Bulk Rate`: Value is constant 0 across all 2,830,743 records.

### Duplicate Columns
- `Fwd Header Length.1`: 100% duplicate of `Fwd Header Length`.

## 4. Generated Artifacts & Visualizations

- Class Distribution: `artifacts/figures/class_distribution.png`
- Attack Categories: `artifacts/figures/attack_categories.png`
- Missing & Inf Anomaly Analysis: `artifacts/figures/missing_and_inf_analysis.png`
- Feature Correlation Matrix: `artifacts/figures/correlation_heatmap.png`
- Feature Distributions by Class: `artifacts/figures/feature_distributions_by_class.png`
