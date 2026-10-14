"""Robust dataset loader for CIC-IDS2017 CSV files."""
import os
import glob
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Generator, Any
import pandas as pd
import numpy as np

from config.dataset_config import RAW_DATA_DIR, EXPECTED_FILES, SAMPLE_RANDOM_SEED
from src.dataset.schema import (
    CANONICAL_COLUMNS,
    TARGET_COLUMN,
    clean_column_name,
    normalize_label
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Production-grade loader for the CIC-IDS2017 dataset.
    Supports memory-efficient loading, stratified sampling, schema validation,
    and automatic label normalization.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self._validate_directory()

    def _validate_directory(self) -> None:
        """Validate that the dataset directory exists and contains CSV files."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory '{self.data_dir}' does not exist!\n"
                f"Please ensure the CIC-IDS2017 dataset is placed at '{self.data_dir}'.\n"
                f"If using MachineLearningCSV.zip, extract the files directly into this directory."
            )

        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in '{self.data_dir}'!\n"
                f"Expected CSV files from CIC-IDS2017:\n" +
                "\n".join(f" - {f}" for f in EXPECTED_FILES)
            )

    def get_csv_files(self) -> List[Path]:
        """Return sorted list of all CSV files in the data directory."""
        return sorted(list(self.data_dir.glob("*.csv")))

    def load_file(
        self,
        filepath: Union[str, Path],
        nrows: Optional[int] = None,
        usecols: Optional[List[str]] = None,
        normalize_labels: bool = True
    ) -> pd.DataFrame:
        """
        Load a single CSV file, strip whitespace from columns, and normalize labels.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File '{path}' does not exist.")

        # Read CSV with robust encoding
        try:
            df = pd.read_csv(
                path,
                nrows=nrows,
                encoding="utf-8",
                encoding_errors="replace",
                low_memory=False
            )
        except Exception as e:
            logger.warning(f"UTF-8 read error for {path.name}: {e}. Retrying with cp1252.")
            df = pd.read_csv(
                path,
                nrows=nrows,
                encoding="cp1252",
                low_memory=False
            )

        # Standardize column names (strip whitespace)
        df.columns = [clean_column_name(c) for c in df.columns]

        # Filter usecols if requested
        if usecols:
            stripped_usecols = [clean_column_name(c) for c in usecols]
            available_cols = [c for c in stripped_usecols if c in df.columns]
            df = df[available_cols]

        # Normalize label strings
        if normalize_labels and TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(normalize_label)

        return df

    def load_combined(
        self,
        nrows_per_file: Optional[int] = None,
        usecols: Optional[List[str]] = None,
        normalize_labels: bool = True,
        sample_strategy: str = "random",
        sample_seed: int = SAMPLE_RANDOM_SEED,
    ) -> pd.DataFrame:
        """
        Load all CSV files and concatenate into a single DataFrame.

        Args:
            nrows_per_file: Maximum rows per file.  None = load full file.
            usecols:        Subset of columns to retain.
            normalize_labels: Normalize label strings.
            sample_strategy: 'random' (default, reproducible stratified sampling)
                             or 'head' (first N rows -- WARNING: temporal bias).
            sample_seed:     Random seed used when sample_strategy='random'.

        Notes on temporal bias (BUG-4)
        --------------------------------
        Using sample_strategy='head' takes the first N rows of each CSV.  Because
        CIC-IDS2017 CSVs are ordered by capture time, Monday's CSV is 100 % BENIGN
        and Friday-DDos concentrates DDoS traffic near the start.  'random' avoids
        this by drawing a reproducible random sample across the full file.
        """
        files = self.get_csv_files()
        logger.info(
            f"Loading {len(files)} files from '{self.data_dir}' "
            f"(limit={nrows_per_file or 'ALL'}, strategy={sample_strategy})..."
        )

        dfs = []
        total_rows = 0
        for f in files:
            logger.info(f"Loading '{f.name}'...")
            df_full = self.load_file(
                f,
                nrows=None,           # always load full file first for random sampling
                usecols=usecols,
                normalize_labels=normalize_labels,
            )
            if nrows_per_file and len(df_full) > nrows_per_file:
                if sample_strategy == "random":
                    df_part = df_full.sample(
                        n=nrows_per_file, random_state=sample_seed, replace=False
                    ).reset_index(drop=True)
                    logger.info(
                        f"  Sampled {nrows_per_file:,} / {len(df_full):,} rows (random, seed={sample_seed})"
                    )
                else:  # 'head'
                    df_part = df_full.head(nrows_per_file)
                    logger.warning(
                        f"  Took first {nrows_per_file:,} rows (head -- temporal bias risk)."
                    )
            else:
                df_part = df_full
                logger.info(f"  Loaded all {len(df_part):,} rows.")
            dfs.append(df_part)
            total_rows += len(df_part)

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {total_rows:,} rows from {len(files)} files.")
        return combined_df

    def load_stream_chunks(
        self,
        chunksize: int = 50000,
        usecols: Optional[List[str]] = None
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Stream all CSV files chunk-by-chunk for memory-constrained workflows.
        """
        files = self.get_csv_files()
        for f in files:
            logger.info(f"Streaming '{f.name}' in chunks of {chunksize}...")
            reader = pd.read_csv(
                f,
                chunksize=chunksize,
                encoding="utf-8",
                encoding_errors="replace",
                low_memory=False
            )
            for chunk in reader:
                chunk.columns = [clean_column_name(c) for c in chunk.columns]
                if usecols:
                    avail = [c for c in usecols if c in chunk.columns]
                    chunk = chunk[avail]
                if TARGET_COLUMN in chunk.columns:
                    chunk[TARGET_COLUMN] = chunk[TARGET_COLUMN].apply(normalize_label)
                yield chunk

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a DataFrame against canonical schema requirements.
        Returns a dictionary summarizing validation status.
        """
        cols = list(df.columns)
        missing_canonical = [c for c in CANONICAL_COLUMNS if c not in cols]
        unexpected_cols = [c for c in cols if c not in CANONICAL_COLUMNS]

        has_target = TARGET_COLUMN in cols
        is_valid = len(missing_canonical) == 0 and has_target

        return {
            "is_valid": is_valid,
            "total_columns": len(cols),
            "missing_canonical_columns": missing_canonical,
            "unexpected_columns": unexpected_cols,
            "has_target": has_target
        }
