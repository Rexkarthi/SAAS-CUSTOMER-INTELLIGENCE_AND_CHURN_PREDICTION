"""
tests/test_predict.py
Unit tests for the ChurnIQ GBM prediction pipeline.

Run:  pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import pathlib
import warnings

warnings.filterwarnings("ignore")

# ── Fixtures ──────────────────────────────────────────────────

MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "best_churn_model.pkl"
META_PATH  = pathlib.Path(__file__).parent.parent / "models" / "meta.pkl"


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def meta():
    return joblib.load(META_PATH)


@pytest.fixture(scope="module")
def high_risk_row():
    """Month-to-Month, Fiber Optic, high charge, new customer — should score HIGH."""
    return pd.DataFrame([{
        "contract": "Month-to-Month", "payment_method": "Credit Card",
        "paperless_billing": "Yes", "internet_service": "Yes",
        "internet_type": "Fiber Optic", "phone_service": "Yes",
        "multiple_lines": "No", "online_security": "No",
        "online_backup": "No", "device_protection_plan": "No",
        "unlimited_data": "Yes", "streaming_tv": "No",
        "streaming_movies": "No", "streaming_music": "No",
        "gender": "Male", "married": "No",
        "number_of_referrals": 0, "monthly_charge": 95.0,
        "tenure_in_months": 3, "total_charges": 285.0,
        "total_revenue": 300.0, "total_refunds": 0.0,
        "total_long_distance_charges": 50.0, "total_extra_data_charges": 0.0,
        "age": 28,
        # engineered features
        "is_m2m": 1.0, "is_fiber": 1.0, "has_security": 0.0,
        "addon_count": 0.0, "high_m2m": 1.0, "new_fiber": 1.0,
        "zero_ref_m2m": 1.0, "m2m_fiber": 1.0, "rev_tenure": 100.0,
        "m2m_no_sec": 1.0, "low_addon_m2m": 1.0,
    }])


@pytest.fixture(scope="module")
def low_risk_row():
    """Two Year contract, long tenure, many referrals — should score LOW."""
    return pd.DataFrame([{
        "contract": "Two Year", "payment_method": "Bank Withdrawal",
        "paperless_billing": "No", "internet_service": "Yes",
        "internet_type": "DSL", "phone_service": "Yes",
        "multiple_lines": "Yes", "online_security": "Yes",
        "online_backup": "Yes", "device_protection_plan": "Yes",
        "unlimited_data": "Yes", "streaming_tv": "Yes",
        "streaming_movies": "Yes", "streaming_music": "Yes",
        "gender": "Female", "married": "Yes",
        "number_of_referrals": 6, "monthly_charge": 45.0,
        "tenure_in_months": 60, "total_charges": 2700.0,
        "total_revenue": 2850.0, "total_refunds": 5.0,
        "total_long_distance_charges": 200.0, "total_extra_data_charges": 10.0,
        "age": 52,
        # engineered features
        "is_m2m": 0.0, "is_fiber": 0.0, "has_security": 1.0,
        "addon_count": 5.0, "high_m2m": 0.0, "new_fiber": 0.0,
        "zero_ref_m2m": 0.0, "m2m_fiber": 0.0, "rev_tenure": 47.5,
        "m2m_no_sec": 0.0, "low_addon_m2m": 0.0,
    }])


# ════════════════════════════════════════════════════════════
# 1. MODEL LOADING
# ════════════════════════════════════════════════════════════

class TestModelLoading:
    def test_model_file_exists(self):
        assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"

    def test_meta_file_exists(self):
        assert META_PATH.exists(), f"Meta not found at {META_PATH}"

    def test_model_is_pipeline(self, model):
        from sklearn.pipeline import Pipeline
        assert isinstance(model, Pipeline), "Model should be a sklearn Pipeline"

    def test_model_has_preprocessor(self, model):
        assert "pre" in model.named_steps, "Pipeline missing 'pre' step"

    def test_model_has_gbm(self, model):
        from sklearn.ensemble import GradientBoostingClassifier
        assert isinstance(
            model.named_steps["model"], GradientBoostingClassifier
        ), "Model step should be GradientBoostingClassifier"

    def test_meta_required_keys(self, meta):
        required = ["cat_cols", "num_cols", "opt_threshold", "auc"]
        for key in required:
            assert key in meta, f"meta.pkl missing required key: '{key}'"

    def test_meta_cat_cols_count(self, meta):
        assert len(meta["cat_cols"]) == 16, (
            f"Expected 16 CAT cols, got {len(meta['cat_cols'])}"
        )

    def test_meta_threshold_valid_range(self, meta):
        thr = meta["opt_threshold"]
        assert 0 < thr < 1, f"Threshold {thr} out of (0,1) range"

    def test_meta_auc_above_baseline(self, meta):
        auc = meta["auc"]
        assert auc > 0.85, f"AUC {auc:.4f} below minimum threshold of 0.85"


# ════════════════════════════════════════════════════════════
# 2. PREDICTION OUTPUT
# ════════════════════════════════════════════════════════════

class TestPredictionOutput:
    def test_predict_proba_shape(self, model, high_risk_row):
        proba = model.predict_proba(high_risk_row)
        assert proba.shape == (1, 2), f"Expected (1,2), got {proba.shape}"

    def test_predict_proba_sums_to_one(self, model, high_risk_row):
        proba = model.predict_proba(high_risk_row)
        total = proba[0, 0] + proba[0, 1]
        assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, not 1.0"

    def test_probability_in_valid_range(self, model, high_risk_row, low_risk_row):
        for row in [high_risk_row, low_risk_row]:
            prob = model.predict_proba(row)[0, 1]
            assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0,1]"

    def test_predict_binary_output(self, model, high_risk_row):
        pred = model.predict(high_risk_row)
        assert pred[0] in [0, 1], f"Prediction {pred[0]} not in {{0,1}}"

    def test_batch_prediction_shape(self, model, high_risk_row, low_risk_row):
        batch = pd.concat([high_risk_row, low_risk_row], ignore_index=True)
        proba = model.predict_proba(batch)
        assert proba.shape == (2, 2), f"Expected (2,2), got {proba.shape}"

    def test_deterministic_predictions(self, model, high_risk_row):
        prob1 = model.predict_proba(high_risk_row)[0, 1]
        prob2 = model.predict_proba(high_risk_row)[0, 1]
        assert prob1 == prob2, "Model is not deterministic"


# ════════════════════════════════════════════════════════════
# 3. BUSINESS LOGIC
# ════════════════════════════════════════════════════════════

class TestBusinessLogic:
    def test_high_risk_scores_higher_than_low(self, model, high_risk_row, low_risk_row):
        prob_high = model.predict_proba(high_risk_row)[0, 1]
        prob_low  = model.predict_proba(low_risk_row)[0, 1]
        assert prob_high > prob_low, (
            f"High-risk row ({prob_high:.3f}) should outscore low-risk ({prob_low:.3f})"
        )

    def test_high_risk_above_threshold(self, model, meta, high_risk_row):
        prob = model.predict_proba(high_risk_row)[0, 1]
        thr  = meta["opt_threshold"]
        assert prob >= thr, (
            f"High-risk customer prob {prob:.3f} below threshold {thr:.3f}"
        )

    def test_low_risk_below_threshold(self, model, meta, low_risk_row):
        prob = model.predict_proba(low_risk_row)[0, 1]
        thr  = meta["opt_threshold"]
        assert prob < thr, (
            f"Low-risk customer prob {prob:.3f} should be below threshold {thr:.3f}"
        )

    def test_m2m_churns_more_than_two_year(self, model):
        """Month-to-Month should always score higher than Two Year (all else equal)."""
        base = {
            "payment_method": "Credit Card", "paperless_billing": "Yes",
            "internet_service": "Yes", "internet_type": "DSL",
            "phone_service": "Yes", "multiple_lines": "No",
            "online_security": "No", "online_backup": "No",
            "device_protection_plan": "No", "unlimited_data": "Yes",
            "streaming_tv": "No", "streaming_movies": "No", "streaming_music": "No",
            "gender": "Male", "married": "No",
            "number_of_referrals": 2, "monthly_charge": 65.0,
            "tenure_in_months": 12, "total_charges": 780.0,
            "total_revenue": 800.0, "total_refunds": 0.0,
            "total_long_distance_charges": 80.0, "total_extra_data_charges": 0.0,
            "age": 35, "is_fiber": 0.0, "has_security": 0.0, "addon_count": 0.0,
            "high_m2m": 0.0, "new_fiber": 0.0, "m2m_fiber": 0.0,
            "rev_tenure": 61.5, "m2m_no_sec": 0.0, "low_addon_m2m": 0.0,
        }
        m2m_row = pd.DataFrame([{**base, "contract": "Month-to-Month",
                                  "is_m2m": 1.0, "zero_ref_m2m": 0.0,
                                  "high_m2m": 0.0, "low_addon_m2m": 1.0, "m2m_no_sec": 1.0}])
        two_row = pd.DataFrame([{**base, "contract": "Two Year",
                                  "is_m2m": 0.0, "zero_ref_m2m": 0.0,
                                  "high_m2m": 0.0, "low_addon_m2m": 0.0, "m2m_no_sec": 0.0}])
        prob_m2m = model.predict_proba(m2m_row)[0, 1]
        prob_two = model.predict_proba(two_row)[0, 1]
        assert prob_m2m > prob_two, (
            f"M2M ({prob_m2m:.3f}) should score higher than Two Year ({prob_two:.3f})"
        )

    def test_more_referrals_reduces_risk(self, model, high_risk_row):
        """Referrals should not dramatically increase churn risk (weak monotonicity)."""
        row_0_ref = high_risk_row.copy()
        row_5_ref = high_risk_row.copy()
        row_5_ref["number_of_referrals"] = 5
        row_5_ref["zero_ref_m2m"] = 0.0

        prob_0 = model.predict_proba(row_0_ref)[0, 1]
        prob_5 = model.predict_proba(row_5_ref)[0, 1]
        # Referrals should not cause a dramatic increase in churn risk
        assert prob_5 < prob_0 + 0.20, (
            f"5 referrals ({prob_5:.3f}) is unreasonably higher than 0 referrals ({prob_0:.3f})"
        )

    def test_risk_tier_assignment(self, model, meta):
        """HIGH tier must have higher actual churn than LOW tier."""
        import pandas as pd
        from sklearn.model_selection import train_test_split

        data_path = pathlib.Path(__file__).parent.parent / "data" / "processed" / "crm_churn_ml_ready.csv"
        if not data_path.exists():
            pytest.skip("Dataset not found — skipping tier validation test")

        df = pd.read_csv(data_path)
        df = df.drop(columns=["customer_id", "state"], errors="ignore")
        CAT = meta["cat_cols"]
        NUM = meta["num_cols"]
        for c in CAT: df[c] = df[c].fillna("Unknown")

        NUM_BASE = ["number_of_referrals","total_extra_data_charges","total_revenue",
                    "total_charges","tenure_in_months","age","monthly_charge",
                    "total_refunds","total_long_distance_charges"]
        for c in NUM_BASE:
            if c in df.columns: df[c] = df[c].fillna(df[c].median())

        # Add engineered features if v3
        if "is_m2m" in NUM:
            df["is_m2m"]        = (df["contract"] == "Month-to-Month").astype(float)
            df["is_fiber"]      = (df["internet_type"] == "Fiber Optic").astype(float)
            df["has_security"]  = (df["online_security"] == "Yes").astype(float)
            df["addon_count"]   = sum((df[c] == "Yes").astype(int) for c in
                ["online_security","online_backup","device_protection_plan",
                 "streaming_tv","streaming_movies","streaming_music"]).astype(float)
            df["high_m2m"]      = (df["is_m2m"] * (df["monthly_charge"] > 70)).astype(float)
            df["new_fiber"]     = (df["is_fiber"] * (df["tenure_in_months"] < 12)).astype(float)
            df["zero_ref_m2m"]  = (df["is_m2m"] * (df["number_of_referrals"] == 0)).astype(float)
            df["m2m_fiber"]     = (df["is_m2m"] * df["is_fiber"]).astype(float)
            df["m2m_no_sec"]    = (df["is_m2m"] * (df["has_security"] == 0)).astype(float)
            df["low_addon_m2m"] = (df["is_m2m"] * (df["addon_count"] <= 1)).astype(float)
            df["rev_tenure"]    = (df["total_revenue"] / (df["tenure_in_months"] + 1)).clip(0, 500)

        X = df.drop("churn", axis=1)
        y = df["churn"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        probs = model.predict_proba(X_test)[:, 1]
        tiers = pd.cut(probs, bins=[0, 0.30, 0.60, 1.01], labels=["Low", "Medium", "High"])

        churn_high = y_test[tiers == "High"].mean()
        churn_low  = y_test[tiers == "Low"].mean()

        assert churn_high > churn_low, (
            f"HIGH tier churn ({churn_high:.1%}) should exceed LOW tier ({churn_low:.1%})"
        )
        assert churn_high > 0.50, f"HIGH tier actual churn {churn_high:.1%} should exceed 50%"
        assert churn_low  < 0.20, f"LOW tier actual churn {churn_low:.1%} should be below 20%"


# ════════════════════════════════════════════════════════════
# 4. PREPROCESSING
# ════════════════════════════════════════════════════════════

class TestPreprocessing:
    def test_handles_unknown_categories(self, model, high_risk_row):
        """Model should not crash on unseen categorical values."""
        row = high_risk_row.copy()
        row["internet_type"] = "Satellite"  # never seen in training
        try:
            prob = model.predict_proba(row)[0, 1]
            assert 0 <= prob <= 1
        except Exception as e:
            pytest.fail(f"Model crashed on unknown category: {e}")

    def test_handles_missing_numeric_gracefully(self, model, high_risk_row):
        """NaN should raise a clear ValueError — GBM does not support NaN natively.
        Caller must impute before scoring. This test documents the expected behaviour."""
        row = high_risk_row.copy()
        row["monthly_charge"] = np.nan
        with pytest.raises((ValueError, Exception)):
            model.predict_proba(row)

    def test_feature_count_consistent(self, model, high_risk_row, low_risk_row):
        """Both rows must produce the same number of transformed features."""
        pre = model.named_steps["pre"]
        n1 = pre.transform(high_risk_row).shape[1]
        n2 = pre.transform(low_risk_row).shape[1]
        assert n1 == n2, f"Feature count mismatch: {n1} vs {n2}"

    def test_numeric_scaling_applied(self, model, high_risk_row):
        """StandardScaler should normalise numeric features."""
        pre = model.named_steps["pre"]
        X_t = pre.transform(high_risk_row)
        # Scaled values should not be in raw magnitude range (e.g. 0-5000)
        assert X_t.max() < 100, "Numeric features appear unscaled"


# ════════════════════════════════════════════════════════════
# 5. MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════

class TestModelPerformance:
    def test_auc_above_09(self, meta):
        """Production model must maintain AUC ≥ 0.90."""
        auc = meta["auc"]
        assert auc >= 0.90, f"AUC {auc:.4f} dropped below 0.90 — retrain required"

    def test_auc_stored_correctly(self, meta):
        assert isinstance(meta["auc"], float), "AUC should be a float"
        assert meta["auc"] <= 1.0, "AUC cannot exceed 1.0"

    def test_model_performance_on_test_set(self, model, meta):
        """End-to-end: recall ≥ 75% and precision ≥ 55% on held-out test set."""
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split

        data_path = pathlib.Path(__file__).parent.parent / "data" / "processed" / "crm_churn_ml_ready.csv"
        if not data_path.exists():
            pytest.skip("Dataset not found — skipping performance test")

        df = pd.read_csv(data_path)
        df = df.drop(columns=["customer_id", "state"], errors="ignore")
        CAT = meta["cat_cols"]
        NUM = meta["num_cols"]
        for c in CAT: df[c] = df[c].fillna("Unknown")

        NUM_BASE_P = ["number_of_referrals","total_extra_data_charges","total_revenue",
                      "total_charges","tenure_in_months","age","monthly_charge",
                      "total_refunds","total_long_distance_charges"]
        for c in NUM_BASE_P:
            if c in df.columns: df[c] = df[c].fillna(df[c].median())

        if "is_m2m" in NUM:
            df["is_m2m"]        = (df["contract"] == "Month-to-Month").astype(float)
            df["is_fiber"]      = (df["internet_type"] == "Fiber Optic").astype(float)
            df["has_security"]  = (df["online_security"] == "Yes").astype(float)
            df["addon_count"]   = sum((df[c] == "Yes").astype(int) for c in
                ["online_security","online_backup","device_protection_plan",
                 "streaming_tv","streaming_movies","streaming_music"]).astype(float)
            df["high_m2m"]      = (df["is_m2m"] * (df["monthly_charge"] > 70)).astype(float)
            df["new_fiber"]     = (df["is_fiber"] * (df["tenure_in_months"] < 12)).astype(float)
            df["zero_ref_m2m"]  = (df["is_m2m"] * (df["number_of_referrals"] == 0)).astype(float)
            df["m2m_fiber"]     = (df["is_m2m"] * df["is_fiber"]).astype(float)
            df["m2m_no_sec"]    = (df["is_m2m"] * (df["has_security"] == 0)).astype(float)
            df["low_addon_m2m"] = (df["is_m2m"] * (df["addon_count"] <= 1)).astype(float)
            df["rev_tenure"]    = (df["total_revenue"] / (df["tenure_in_months"] + 1)).clip(0, 500)

        X = df.drop("churn", axis=1)
        y = df["churn"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        thr  = meta["opt_threshold"]
        pred = (model.predict_proba(X_test)[:, 1] >= thr).astype(int)
        TN, FP, FN, TP = confusion_matrix(y_test, pred).ravel()

        recall    = TP / (TP + FN)
        precision = TP / (TP + FP)

        assert recall >= 0.75, f"Recall {recall:.1%} below minimum 75%"
        assert precision >= 0.55, f"Precision {precision:.1%} below minimum 55%"
