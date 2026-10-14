"""Unit tests for dataset schema, column definitions, and label mappings."""
import pytest
from src.dataset.schema import (
    CANONICAL_COLUMNS,
    CONSTANT_COLUMNS,
    DUPLICATE_COLUMNS,
    TARGET_COLUMN,
    LABEL_MAPPING,
    INVERSE_LABEL_MAPPING,
    BINARY_MAPPING,
    ATTACK_CATEGORY_MAPPING,
    clean_column_name,
    normalize_label
)


def test_clean_column_name():
    assert clean_column_name(" Destination Port") == "Destination Port"
    assert clean_column_name(" Flow Duration ") == "Flow Duration"
    assert clean_column_name("Label") == "Label"


def test_normalize_label():
    assert normalize_label("BENIGN") == "BENIGN"
    assert normalize_label("DDoS") == "DDoS"
    assert normalize_label("Web Attack \ufffd Brute Force") == "Web Attack - Brute Force"
    assert normalize_label("Web Attack ï¿½ XSS") == "Web Attack - XSS"
    assert normalize_label("Web Attack ï¿½ Sql Injection") == "Web Attack - Sql Injection"
    assert normalize_label(" Web Attack - Brute Force ") == "Web Attack - Brute Force"


def test_schema_column_counts():
    assert len(CANONICAL_COLUMNS) == 79
    assert TARGET_COLUMN in CANONICAL_COLUMNS
    assert len(CONSTANT_COLUMNS) == 8
    assert len(DUPLICATE_COLUMNS) == 1
    assert "Fwd Header Length.1" in DUPLICATE_COLUMNS


def test_label_mappings_consistency():
    assert len(LABEL_MAPPING) == 15
    assert LABEL_MAPPING["BENIGN"] == 0
    assert len(INVERSE_LABEL_MAPPING) == 15

    for label, code in LABEL_MAPPING.items():
        assert INVERSE_LABEL_MAPPING[code] == label
        assert label in ATTACK_CATEGORY_MAPPING

    assert BINARY_MAPPING["BENIGN"] == 0
    assert BINARY_MAPPING["ATTACK"] == 1
