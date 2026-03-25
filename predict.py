#!/usr/bin/env python3
"""
predict.py — ChurnIQ Production Inference Script
saas-churn-intelligence

Usage:
    # Single customer (JSON string)
    python predict.py --customer '{"contract": "Month-to-Month", "monthly_charge": 89.5, ...}'

    # Batch from CSV
    python predict.py --batch data/processed/crm_churn_ml_ready.csv --output outputs/scored.csv

    # Interactive demo mode
    python predict.py --demo
"""

import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
MODEL_PATHS = [
    "models/best_churn_model.pkl",
    "../models/best_churn_model.pkl",
    "best_churn_model.pkl",
]
DATA_PATHS = [
    "data/processed/crm_churn_ml_ready.csv",
    "../data/processed/crm_churn_ml_ready.csv",
    "crm_churn_ml_ready.csv",
]

OPT_THRESHOLD      = 0.2728   # Youden's J — optimized on test set
DROP_COLS          = ["customer_id", "state"]

RISK_TIERS = {
    "HIGH":   (0.60, 1.01, "🔴 HIGH RISK",   "Immediate outreach — retention specialist call + competitive offer"),
    "MEDIUM": (0.30, 0.60, "🟡 MEDIUM RISK", "Proactive campaign — loyalty discount + service upgrade offer"),
    "LOW":    (0.00, 0.30, "🟢 LOW RISK",    "Nurture — referral program invitation + upsell opportunity"),
}

BUSINESS_PARAMS = {
    "avg_monthly_charge":  70.20,
    "retention_horizon":   18,       # months
    "intervention_cost":   50,       # $ per outreach
}


# ── UTILITIES ──────────────────────────────────────────────────────────────────

def find_path(candidates: list) -> str | None:
    """Return first existing path from candidate list."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def get_risk_tier(prob: float) -> tuple:
    for tier, (lo, hi, label, action) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier, label, action
    return "HIGH", RISK_TIERS["HIGH"][2], RISK_TIERS["HIGH"][3]


def compute_business_value(prob: float) -> dict:
    """Estimate expected net value of intervening on this customer."""
    p  = BUSINESS_PARAMS
    revenue_if_saved  = p["avg_monthly_charge"] * p["retention_horizon"]
    net_value_if_churn = prob * (revenue_if_saved - p["intervention_cost"])
    net_cost_if_stay   = (1 - prob) * p["intervention_cost"]
    expected_net       = net_value_if_churn - net_cost_if_stay
    return {
        "churn_probability":     round(prob, 4),
        "revenue_if_saved":      round(revenue_if_saved, 2),
        "expected_net_value":    round(expected_net, 2),
        "recommend_intervene":   expected_net > 0,
    }


# ── MODEL LOADING / TRAINING ───────────────────────────────────────────────────

def load_or_train_model(data_path: str | None = None) -> Pipeline:
    """Load saved model. If not found, retrain from data."""
    model_path = find_path(MODEL_PATHS)

    if model_path:
        print(f"✅ Model loaded from: {model_path}")
        return joblib.load(model_path)

    # Retrain
    print("⚠️  Saved model not found — retraining (takes ~60s)...")
    dp = data_path or find_path(DATA_PATHS)
    if dp is None:
        raise FileNotFoundError(
            "Cannot find model or training data.\n"
            "Place 'crm_churn_ml_ready.csv' in data/processed/ or the current directory."
        )

    df = pd.read_csv(dp)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X  = df.drop("churn", axis=1)
    y  = df["churn"]

    for col in X.select_dtypes(include=["object", "str"]).columns:
        X[col] = X[col].fillna("Unknown")
    for col in X.select_dtypes(include="number").columns:
        X[col] = X[col].fillna(X[col].median())

    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    model = Pipeline([
        ("pre",   pre),
        ("model", GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )),
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"   Retrained AUC: {auc:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_churn_model.pkl")
    print("   Saved to models/best_churn_model.pkl")
    return model


def prepare_single_customer(customer_dict: dict) -> pd.DataFrame:
    """
    Convert a raw customer dict into a DataFrame the model can score.
    Missing features are filled with sensible defaults.
    """
    DEFAULTS = {
        "streaming_movies":          "No",
        "number_of_referrals":        0,
        "total_extra_data_charges":   0,
        "multiple_lines":            "No",
        "total_revenue":              500.0,
        "paperless_billing":         "Yes",
        "phone_service":             "Yes",
        "total_charges":              500.0,
        "internet_service":          "Yes",
        "tenure_in_months":           12,
        "age":                        35,
        "streaming_tv":              "No",
        "gender":                    "Male",
        "online_backup":             "No",
        "online_security":           "No",
        "internet_type":             "Cable",
        "device_protection_plan":    "No",
        "contract":                  "Month-to-Month",
        "monthly_charge":             70.0,
        "unlimited_data":            "Yes",
        "married":                   "No",
        "streaming_music":           "No",
        "payment_method":            "Bank Withdrawal",
        "total_refunds":              0.0,
        "total_long_distance_charges": 100.0,
    }
    row = {**DEFAULTS, **customer_dict}
    # Remove ID columns if passed
    for col in DROP_COLS:
        row.pop(col, None)
    row.pop("churn", None)
    return pd.DataFrame([row])


# ── PREDICTION FUNCTIONS ───────────────────────────────────────────────────────

def predict_single(model: Pipeline, customer_dict: dict, verbose: bool = True) -> dict:
    """Score a single customer and return full result dict."""
    df_cust = prepare_single_customer(customer_dict)
    prob    = model.predict_proba(df_cust)[0, 1]
    tier_key, tier_label, action = get_risk_tier(prob)
    biz     = compute_business_value(prob)

    result = {
        "churn_probability":   round(prob, 4),
        "risk_tier":           tier_key,
        "risk_label":          tier_label,
        "recommendation":      action,
        "intervene":           biz["recommend_intervene"],
        "expected_net_value":  biz["expected_net_value"],
    }

    if verbose:
        _print_single_result(result, customer_dict)

    return result


def predict_batch(model: Pipeline, input_path: str, output_path: str) -> pd.DataFrame:
    """Score an entire CSV file and save results."""
    print(f"\n📂 Loading batch: {input_path}")
    df = pd.read_csv(input_path)

    actual_churn = df.pop("churn") if "churn" in df.columns else None
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    probs = model.predict_proba(df)[:, 1]

    results = df.copy()
    results["churn_probability"] = probs.round(4)
    results["risk_tier"]         = [get_risk_tier(p)[0] for p in probs]
    results["risk_label"]        = [get_risk_tier(p)[1] for p in probs]
    results["intervene"]         = (probs >= OPT_THRESHOLD).astype(int)
    results["expected_net_value"]= [compute_business_value(p)["expected_net_value"] for p in probs]

    if actual_churn is not None:
        results["actual_churn"] = actual_churn.values
        auc = roc_auc_score(actual_churn, probs)
        print(f"   Validation AUC: {auc:.4f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results.to_csv(output_path, index=False)

    # Summary
    tier_counts = results["risk_tier"].value_counts()
    total       = len(results)
    print(f"\n📊 Batch Scoring Complete — {total:,} customers")
    print(f"   🔴 High Risk  : {tier_counts.get('HIGH',0):>5,}  ({tier_counts.get('HIGH',0)/total*100:.1f}%)")
    print(f"   🟡 Med Risk   : {tier_counts.get('MEDIUM',0):>5,}  ({tier_counts.get('MEDIUM',0)/total*100:.1f}%)")
    print(f"   🟢 Low Risk   : {tier_counts.get('LOW',0):>5,}  ({tier_counts.get('LOW',0)/total*100:.1f}%)")
    print(f"\n   Flagged for intervention: {results['intervene'].sum():,}")
    print(f"   Expected net value (total): ${results['expected_net_value'].sum():,.0f}")
    print(f"\n💾 Saved to: {output_path}")

    return results


def run_demo(model: Pipeline) -> None:
    """Run predictions on 3 illustrative customers."""
    demo_customers = [
        {
            "name": "High-Risk Customer",
            "contract": "Month-to-Month",
            "monthly_charge": 95.0,
            "tenure_in_months": 6,
            "internet_type": "Fiber Optic",
            "number_of_referrals": 0,
            "total_charges": 570.0,
            "total_revenue": 570.0,
            "age": 28,
            "online_security": "No",
            "total_long_distance_charges": 45.0,
        },
        {
            "name": "Medium-Risk Customer",
            "contract": "Month-to-Month",
            "monthly_charge": 65.0,
            "tenure_in_months": 18,
            "internet_type": "Cable",
            "number_of_referrals": 1,
            "total_charges": 1170.0,
            "total_revenue": 1250.0,
            "age": 45,
            "online_security": "Yes",
            "total_long_distance_charges": 120.0,
        },
        {
            "name": "Low-Risk Customer",
            "contract": "Two Year",
            "monthly_charge": 55.0,
            "tenure_in_months": 48,
            "internet_type": "DSL",
            "number_of_referrals": 5,
            "total_charges": 2640.0,
            "total_revenue": 2750.0,
            "age": 52,
            "online_security": "Yes",
            "total_long_distance_charges": 280.0,
        },
    ]

    print("\n" + "═"*55)
    print("  ChurnIQ — Demo Mode  (3 Sample Customers)")
    print("═"*55)

    for cust in demo_customers:
        name = cust.pop("name")
        print(f"\n  ── {name} ──")
        predict_single(model, cust, verbose=True)


# ── PRINT HELPERS ──────────────────────────────────────────────────────────────

def _print_single_result(result: dict, customer_dict: dict) -> None:
    prob  = result["churn_probability"]
    bar_len = int(prob * 30)
    bar  = "█" * bar_len + "░" * (30 - bar_len)

    print("\n" + "─"*50)
    print("  ChurnIQ Prediction Result")
    print("─"*50)
    print(f"  Risk Score   : [{bar}] {prob:.1%}")
    print(f"  Risk Tier    : {result['risk_label']}")
    print(f"  Recommend    : {result['recommendation']}")
    print(f"  Intervene?   : {'✅ YES' if result['intervene'] else '⬜ No'}")
    print(f"  Expected ROI : ${result['expected_net_value']:,.0f}  per intervention")
    print("─"*50)

    # Highlight key risk factors from input
    risk_signals = []
    if customer_dict.get("contract") == "Month-to-Month":
        risk_signals.append("Month-to-Month contract (+29% churn signal)")
    if float(customer_dict.get("monthly_charge", 0)) > 80:
        risk_signals.append(f"High monthly charge: ${customer_dict['monthly_charge']}")
    if int(customer_dict.get("tenure_in_months", 99)) < 12:
        risk_signals.append(f"Short tenure: {customer_dict['tenure_in_months']} months")
    if customer_dict.get("internet_type") == "Fiber Optic":
        risk_signals.append("Fiber Optic customer (40.1% segment churn rate)")
    if int(customer_dict.get("number_of_referrals", 99)) == 0:
        risk_signals.append("Zero referrals (low engagement signal)")
    if customer_dict.get("online_security") == "No":
        risk_signals.append("No online security add-on")

    if risk_signals:
        print("  Risk Signals:")
        for sig in risk_signals:
            print(f"    ⚠️  {sig}")
    print("─"*50)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ChurnIQ — Telecom Customer Churn Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --demo
  python predict.py --customer '{"contract":"Month-to-Month","monthly_charge":89.5}'
  python predict.py --batch data/processed/crm_churn_ml_ready.csv --output outputs/scored.csv
        """
    )
    parser.add_argument("--customer", type=str,
                        help="JSON string of customer features")
    parser.add_argument("--batch",    type=str,
                        help="Path to CSV file for batch scoring")
    parser.add_argument("--output",   type=str, default="outputs/scored_customers.csv",
                        help="Output path for batch results (default: outputs/scored_customers.csv)")
    parser.add_argument("--demo",     action="store_true",
                        help="Run demo with 3 sample customers")
    parser.add_argument("--threshold",type=float, default=0.2728,
                        help="Decision threshold (default: 0.2728)")

    args = parser.parse_args()

    global OPT_THRESHOLD

    if not any([args.customer, args.batch, args.demo]):
        parser.print_help()
        print("\n💡 Try:  python predict.py --demo")
        sys.exit(0)

    # Override threshold if specified
    OPT_THRESHOLD = args.threshold

    # Load model
    model = load_or_train_model()

    if args.demo:
        run_demo(model)

    elif args.customer:
        try:
            customer_dict = json.loads(args.customer)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            sys.exit(1)
        predict_single(model, customer_dict, verbose=True)

    elif args.batch:
        if not os.path.exists(args.batch):
            print(f"❌ File not found: {args.batch}")
            sys.exit(1)
        predict_batch(model, args.batch, args.output)


if __name__ == "__main__":
    main()
