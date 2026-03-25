"""
tests/test_preprocessing.py
Unit tests for data loading, feature engineering, and preprocessing.

Run:  pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import pathlib
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "processed" / "crm_churn_ml_ready.csv"

CAT_COLS = [
    "streaming_movies", "multiple_lines", "paperless_billing", "phone_service",
    "internet_service", "streaming_tv", "gender", "online_backup", "online_security",
    "internet_type", "device_protection_plan", "contract", "unlimited_data",
    "married", "streaming_music", "payment_method",
]
NUM_COLS = [
    "number_of_referrals", "total_extra_data_charges", "total_revenue",
    "total_charges", "tenure_in_months", "age", "monthly_charge",
    "total_refunds", "total_long_distance_charges",
]


@pytest.fixture(scope="module")
def raw_df():
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["customer_id", "state"], errors="ignore")


@pytest.fixture(scope="module")
def clean_df(raw_df):
    df = raw_df.copy()
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    return df


@pytest.fixture(scope="module")
def engineered_df(clean_df):
    df = clean_df.copy()
    df["is_m2m"]        = (df["contract"] == "Month-to-Month").astype(float)
    df["is_fiber"]      = (df["internet_type"] == "Fiber Optic").astype(float)
    df["has_security"]  = (df["online_security"] == "Yes").astype(float)
    df["addon_count"]   = sum((df[c] == "Yes").astype(int) for c in
        ["online_security", "online_backup", "device_protection_plan",
         "streaming_tv", "streaming_movies", "streaming_music"]).astype(float)
    df["high_m2m"]      = (df["is_m2m"] * (df["monthly_charge"] > 70)).astype(float)
    df["new_fiber"]     = (df["is_fiber"] * (df["tenure_in_months"] < 12)).astype(float)
    df["zero_ref_m2m"]  = (df["is_m2m"] * (df["number_of_referrals"] == 0)).astype(float)
    df["m2m_fiber"]     = (df["is_m2m"] * df["is_fiber"]).astype(float)
    df["m2m_no_sec"]    = (df["is_m2m"] * (df["has_security"] == 0)).astype(float)
    df["low_addon_m2m"] = (df["is_m2m"] * (df["addon_count"] <= 1)).astype(float)
    df["rev_tenure"]    = (df["total_revenue"] / (df["tenure_in_months"] + 1)).clip(0, 500)
    return df


# ════════════════════════════════════════════════════════════
# 1. RAW DATA
# ════════════════════════════════════════════════════════════

class TestRawData:
    def test_dataset_loads(self, raw_df):
        assert raw_df is not None
        assert len(raw_df) > 0

    def test_dataset_row_count(self, raw_df):
        assert len(raw_df) >= 10000, f"Expected ≥10,000 rows, got {len(raw_df)}"

    def test_churn_column_exists(self, raw_df):
        assert "churn" in raw_df.columns

    def test_churn_is_binary(self, raw_df):
        assert set(raw_df["churn"].dropna().unique()).issubset({0, 1})

    def test_churn_rate_reasonable(self, raw_df):
        rate = raw_df["churn"].mean()
        assert 0.10 <= rate <= 0.50, f"Churn rate {rate:.1%} is outside expected 10–50% range"

    def test_required_columns_present(self, raw_df):
        required = CAT_COLS + NUM_COLS + ["churn"]
        missing = [c for c in required if c not in raw_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_duplicate_rows(self, raw_df):
        dupes = raw_df.duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate rows found"

    def test_tenure_positive(self, raw_df):
        assert (raw_df["tenure_in_months"].dropna() >= 0).all(), \
            "tenure_in_months contains negative values"

    def test_monthly_charge_reasonable(self, raw_df):
        """monthly_charge allows negatives (billing credits) but no extreme outliers."""
        assert (raw_df["monthly_charge"].dropna() > -50).all(), \
            f"monthly_charge has extreme values: min={raw_df['monthly_charge'].min()}"

    def test_age_in_valid_range(self, raw_df):
        ages = raw_df["age"].dropna()
        assert (ages >= 18).all() and (ages <= 100).all(), \
            "age values outside expected range 18–100"


# ════════════════════════════════════════════════════════════
# 2. CLEANING
# ════════════════════════════════════════════════════════════

class TestCleaning:
    def test_no_nulls_in_cat_after_fill(self, clean_df):
        for c in CAT_COLS:
            if c in clean_df.columns:
                nulls = clean_df[c].isna().sum()
                assert nulls == 0, f"Column '{c}' has {nulls} nulls after fillna"

    def test_no_nulls_in_num_after_fill(self, clean_df):
        for c in NUM_COLS:
            if c in clean_df.columns:
                nulls = clean_df[c].isna().sum()
                assert nulls == 0, f"Column '{c}' has {nulls} nulls after fillna"

    def test_unknown_fill_used_for_cat(self, raw_df, clean_df):
        for c in CAT_COLS:
            if c in raw_df.columns and raw_df[c].isna().sum() > 0:
                assert "Unknown" in clean_df[c].values, \
                    f"Column '{c}' should have 'Unknown' fill"

    def test_numeric_dtype_preserved(self, clean_df):
        for c in NUM_COLS:
            if c in clean_df.columns:
                assert pd.api.types.is_numeric_dtype(clean_df[c]), \
                    f"Column '{c}' lost numeric dtype after cleaning"


# ════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    def test_engineered_columns_exist(self, engineered_df):
        expected = ["is_m2m", "is_fiber", "has_security", "addon_count",
                    "high_m2m", "new_fiber", "zero_ref_m2m", "m2m_fiber",
                    "m2m_no_sec", "low_addon_m2m", "rev_tenure"]
        for col in expected:
            assert col in engineered_df.columns, f"Engineered column '{col}' missing"

    def test_is_m2m_binary(self, engineered_df):
        vals = engineered_df["is_m2m"].unique()
        assert set(vals).issubset({0.0, 1.0}), f"is_m2m has non-binary values: {vals}"

    def test_is_m2m_matches_contract(self, engineered_df):
        m2m_flag = engineered_df["is_m2m"] == 1.0
        m2m_contract = engineered_df["contract"] == "Month-to-Month"
        assert (m2m_flag == m2m_contract).all(), "is_m2m doesn't match contract column"

    def test_addon_count_range(self, engineered_df):
        assert engineered_df["addon_count"].between(0, 6).all(), \
            "addon_count outside expected range 0–6"

    def test_rev_tenure_no_infinity(self, engineered_df):
        assert not engineered_df["rev_tenure"].isin([np.inf, -np.inf]).any(), \
            "rev_tenure contains infinity values"

    def test_rev_tenure_clipped(self, engineered_df):
        assert engineered_df["rev_tenure"].max() <= 500, \
            "rev_tenure not clipped at 500"

    def test_high_m2m_is_subset_of_m2m(self, engineered_df):
        high_m2m = engineered_df["high_m2m"] == 1.0
        is_m2m   = engineered_df["is_m2m"] == 1.0
        assert (high_m2m & ~is_m2m).sum() == 0, \
            "high_m2m=1 found on non-M2M customers"

    def test_m2m_fiber_is_intersection(self, engineered_df):
        expected = (engineered_df["is_m2m"] == 1) & (engineered_df["is_fiber"] == 1)
        actual   = engineered_df["m2m_fiber"] == 1.0
        assert (expected == actual).all(), "m2m_fiber is not the intersection of is_m2m and is_fiber"

    def test_m2m_churn_rate_higher_than_annual(self, engineered_df):
        m2m_churn = engineered_df[engineered_df["is_m2m"] == 1]["churn"].mean()
        ann_churn = engineered_df[engineered_df["is_m2m"] == 0]["churn"].mean()
        assert m2m_churn > ann_churn, \
            f"M2M churn ({m2m_churn:.1%}) should exceed annual ({ann_churn:.1%})"

    def test_no_new_nulls_after_engineering(self, engineered_df):
        eng_cols = ["is_m2m","is_fiber","has_security","addon_count","high_m2m",
                    "new_fiber","zero_ref_m2m","m2m_fiber","m2m_no_sec","low_addon_m2m","rev_tenure"]
        for c in eng_cols:
            nulls = engineered_df[c].isna().sum()
            assert nulls == 0, f"Engineered column '{c}' has {nulls} null values"
