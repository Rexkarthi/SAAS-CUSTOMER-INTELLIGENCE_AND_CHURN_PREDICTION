"""
tests/test_api.py
Unit tests for the ChurnIQ Flask API endpoints.

Run:  pytest tests/ -v
"""

import pytest
import json
import io
import sys
import pathlib

# Add webapp to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "webapp"))
import app as webapp


# ── Fixture ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    webapp.app.config["TESTING"] = True
    with webapp.app.test_client() as c:
        yield c


@pytest.fixture
def valid_payload():
    return {
        "contract": "Month-to-Month", "payment_method": "Credit Card",
        "paperless_billing": "Yes", "internet_service": "Yes",
        "internet_type": "Fiber Optic", "phone_service": "Yes",
        "multiple_lines": "No", "online_security": "No",
        "online_backup": "No", "device_protection_plan": "No",
        "unlimited_data": "Yes", "streaming_tv": "No",
        "streaming_movies": "No", "streaming_music": "No",
        "gender": "Male", "married": "No",
        "number_of_referrals": "0", "monthly_charge": "95",
        "tenure_in_months": "3", "total_charges": "285",
        "total_revenue": "300", "total_refunds": "0",
        "total_long_distance_charges": "50", "total_extra_data_charges": "0",
        "age": "28",
    }


@pytest.fixture
def sample_csv_bytes():
    rows = [
        "contract,payment_method,paperless_billing,internet_service,internet_type,"
        "phone_service,multiple_lines,online_security,online_backup,device_protection_plan,"
        "unlimited_data,streaming_tv,streaming_movies,streaming_music,gender,married,"
        "number_of_referrals,monthly_charge,tenure_in_months,total_charges,total_revenue,"
        "total_refunds,total_long_distance_charges,total_extra_data_charges,age",
        "Month-to-Month,Credit Card,Yes,Yes,Fiber Optic,Yes,No,No,No,No,Yes,No,No,No,Male,No,0,95,3,285,300,0,50,0,28",
        "Two Year,Bank Withdrawal,No,Yes,DSL,Yes,Yes,Yes,Yes,Yes,Yes,Yes,Yes,Yes,Female,Yes,5,45,60,2700,2850,5,200,10,52",
        "One Year,Credit Card,Yes,Yes,Cable,No,No,No,No,No,No,No,No,No,Male,No,2,65,24,1560,1650,0,100,5,38",
    ]
    return "\n".join(rows).encode("utf-8")


# ════════════════════════════════════════════════════════════
# 1. HEALTH ENDPOINT
# ════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_returns_ok_status(self, client):
        r = client.get("/health")
        d = r.get_json()
        assert d["status"] == "ok"

    def test_health_returns_auc(self, client):
        r = client.get("/health")
        d = r.get_json()
        assert "auc" in d
        assert d["auc"] > 0.85

    def test_health_content_type_json(self, client):
        r = client.get("/health")
        assert "application/json" in r.content_type


# ════════════════════════════════════════════════════════════
# 2. SINGLE PREDICT ENDPOINT
# ════════════════════════════════════════════════════════════

class TestPredictEndpoint:
    def test_predict_returns_200(self, client, valid_payload):
        r = client.post("/predict", json=valid_payload)
        assert r.status_code == 200

    def test_predict_returns_probability(self, client, valid_payload):
        r = client.get_json() if hasattr(client, 'get_json') else client.post("/predict", json=valid_payload).get_json()
        r = client.post("/predict", json=valid_payload).get_json()
        assert "probability" in r
        assert 0 <= r["probability"] <= 100

    def test_predict_returns_tier(self, client, valid_payload):
        r = client.post("/predict", json=valid_payload).get_json()
        assert "tier" in r
        assert r["tier"] in ["HIGH RISK", "MEDIUM RISK", "LOW RISK"]

    def test_predict_returns_action(self, client, valid_payload):
        r = client.post("/predict", json=valid_payload).get_json()
        assert "action" in r
        assert len(r["action"]) > 10

    def test_predict_returns_signals(self, client, valid_payload):
        r = client.post("/predict", json=valid_payload).get_json()
        assert "signals" in r
        assert isinstance(r["signals"], list)

    def test_predict_returns_churn_flag(self, client, valid_payload):
        r = client.post("/predict", json=valid_payload).get_json()
        assert "churn" in r
        assert r["churn"] in [0, 1]

    def test_high_risk_customer_flagged(self, client, valid_payload):
        """M2M + Fiber + new customer should be flagged as HIGH or MEDIUM risk."""
        r = client.post("/predict", json=valid_payload).get_json()
        assert r["tier"] in ["HIGH RISK", "MEDIUM RISK"], (
            f"High-risk customer scored as: {r['tier']}"
        )

    def test_low_risk_customer_not_high(self, client):
        low_payload = {
            "contract": "Two Year", "payment_method": "Bank Withdrawal",
            "paperless_billing": "No", "internet_service": "Yes",
            "internet_type": "DSL", "phone_service": "Yes",
            "multiple_lines": "Yes", "online_security": "Yes",
            "online_backup": "Yes", "device_protection_plan": "Yes",
            "unlimited_data": "Yes", "streaming_tv": "Yes",
            "streaming_movies": "Yes", "streaming_music": "Yes",
            "gender": "Female", "married": "Yes",
            "number_of_referrals": "6", "monthly_charge": "45",
            "tenure_in_months": "60", "total_charges": "2700",
            "total_revenue": "2850", "total_refunds": "5",
            "total_long_distance_charges": "200", "total_extra_data_charges": "10",
            "age": "52",
        }
        r = client.post("/predict", json=low_payload).get_json()
        assert r["tier"] == "LOW RISK", (
            f"Low-risk customer flagged as: {r['tier']} ({r['probability']}%)"
        )

    def test_predict_empty_payload_returns_prediction(self, client):
        """Empty payload should use defaults and return a valid prediction, not crash."""
        r = client.post("/predict", json={})
        assert r.status_code == 200
        d = r.get_json()
        assert "probability" in d

    def test_predict_content_type_json(self, client, valid_payload):
        r = client.post("/predict", json=valid_payload)
        assert "application/json" in r.content_type


# ════════════════════════════════════════════════════════════
# 3. BATCH ENDPOINT
# ════════════════════════════════════════════════════════════

class TestBatchEndpoint:
    def test_batch_returns_200(self, client, sample_csv_bytes):
        r = client.post(
            "/batch",
            data={"file": (io.BytesIO(sample_csv_bytes), "customers.csv")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200

    def test_batch_returns_correct_count(self, client, sample_csv_bytes):
        r = client.post(
            "/batch",
            data={"file": (io.BytesIO(sample_csv_bytes), "customers.csv")},
            content_type="multipart/form-data",
        ).get_json()
        assert r["total"] == 3

    def test_batch_returns_tier_counts(self, client, sample_csv_bytes):
        r = client.post(
            "/batch",
            data={"file": (io.BytesIO(sample_csv_bytes), "customers.csv")},
            content_type="multipart/form-data",
        ).get_json()
        assert r["high"] + r["medium"] + r["low"] == r["total"]

    def test_batch_returns_results_list(self, client, sample_csv_bytes):
        r = client.post(
            "/batch",
            data={"file": (io.BytesIO(sample_csv_bytes), "customers.csv")},
            content_type="multipart/form-data",
        ).get_json()
        assert "results" in r
        assert len(r["results"]) == 3

    def test_batch_result_fields(self, client, sample_csv_bytes):
        r = client.post(
            "/batch",
            data={"file": (io.BytesIO(sample_csv_bytes), "customers.csv")},
            content_type="multipart/form-data",
        ).get_json()
        required = ["churn_probability", "risk_tier", "predicted_churn",
                    "net_value", "revenue_at_risk", "top_signal", "action"]
        for field in required:
            assert field in r["results"][0], f"Missing field in result: {field}"

    def test_batch_returns_csv(self, client, sample_csv_bytes):
        r = client.post(
            "/batch",
            data={"file": (io.BytesIO(sample_csv_bytes), "customers.csv")},
            content_type="multipart/form-data",
        ).get_json()
        assert "csv" in r
        assert len(r["csv"]) > 100

    def test_batch_no_file_returns_400(self, client):
        r = client.post("/batch", content_type="multipart/form-data")
        assert r.status_code == 400

    def test_batch_wrong_extension_returns_400(self, client):
        r = client.post(
            "/batch",
            data={"file": (io.BytesIO(b"data"), "customers.xlsx")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400

    def test_batch_avg_probability_in_range(self, client, sample_csv_bytes):
        r = client.post(
            "/batch",
            data={"file": (io.BytesIO(sample_csv_bytes), "customers.csv")},
            content_type="multipart/form-data",
        ).get_json()
        assert 0 <= r["avg_probability"] <= 100

    def test_batch_revenue_at_risk_positive(self, client, sample_csv_bytes):
        r = client.post(
            "/batch",
            data={"file": (io.BytesIO(sample_csv_bytes), "customers.csv")},
            content_type="multipart/form-data",
        ).get_json()
        assert r["total_rev_at_risk"] >= 0
