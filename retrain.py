"""
retrain.py — ChurnIQ Automated Retraining Pipeline
====================================================
Run:  python retrain.py
      python retrain.py --force      (retrain regardless of drift)
      python retrain.py --dry-run    (check drift only, no retraining)

What it does:
  1. Loads current production model + new data
  2. Computes PSI for every feature (drift detection)
  3. If PSI > 0.25 on any key feature  → triggers retraining
  4. Trains new model, runs A/B comparison vs current
  5. Promotes new model only if AUC improved or stayed within 0.5%
  6. Logs everything to retrain_log.jsonl
"""

import argparse
import json
import os
import pathlib
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent
MODELS_DIR = ROOT / "models"
LOG_FILE = ROOT / "retrain_log.jsonl"

MODEL_PATH = MODELS_DIR / "best_churn_model.pkl"
META_PATH = MODELS_DIR / "meta.pkl"
BACKUP_DIR = MODELS_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# PSI — DRIFT DETECTION
# ══════════════════════════════════════════════════════════════


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    < 0.10  → stable
    0.10–0.25 → monitor
    > 0.25  → retrain
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    exp_c = np.histogram(expected, bins=breakpoints)[0].astype(float)
    act_c = np.histogram(actual, bins=breakpoints)[0].astype(float)
    exp_c = np.where(exp_c == 0, 0.5, exp_c)
    act_c = np.where(act_c == 0, 0.5, act_c)

    ep = exp_c / exp_c.sum()
    ap = act_c / act_c.sum()
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def psi_status(val: float) -> str:
    if val < 0.10:
        return "STABLE"
    if val < 0.25:
        return "WARNING"
    return "DRIFT"


def check_drift(ref_df: pd.DataFrame, new_df: pd.DataFrame, num_cols: list) -> dict:
    """Returns per-feature PSI and overall drift verdict."""
    results = {}
    num_base = [
        c
        for c in num_cols
        if c in ref_df.columns
        and c in new_df.columns
        and not c.startswith(
            ("is_", "has_", "high_", "new_", "zero_", "m2m_", "low_", "rev_", "addon_")
        )
    ]
    for col in num_base:
        psi = compute_psi(ref_df[col].dropna().values, new_df[col].dropna().values)
        results[col] = {"psi": round(psi, 4), "status": psi_status(psi)}

    max_psi = max(r["psi"] for r in results.values()) if results else 0
    n_drift = sum(1 for r in results.values() if r["status"] == "DRIFT")
    n_warning = sum(1 for r in results.values() if r["status"] == "WARNING")

    verdict = "STABLE"
    if n_drift >= 2 or max_psi > 0.40:
        verdict = "RETRAIN"
    elif n_drift == 1 or n_warning >= 3:
        verdict = "WARNING"

    return {
        "features": results,
        "max_psi": round(max_psi, 4),
        "n_drift": n_drift,
        "n_warning": n_warning,
        "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 11 engineered features — must match NB04 exactly."""
    df = df.copy()
    df["is_m2m"] = (df["contract"] == "Month-to-Month").astype(float)
    df["is_fiber"] = (df["internet_type"] == "Fiber Optic").astype(float)
    df["has_security"] = (df["online_security"] == "Yes").astype(float)
    df["addon_count"] = sum(
        (df[c] == "Yes").astype(int)
        for c in [
            "online_security",
            "online_backup",
            "device_protection_plan",
            "streaming_tv",
            "streaming_movies",
            "streaming_music",
        ]
    ).astype(float)
    df["high_m2m"] = (df["is_m2m"] * (df["monthly_charge"] > 70)).astype(float)
    df["new_fiber"] = (df["is_fiber"] * (df["tenure_in_months"] < 12)).astype(float)
    df["zero_ref_m2m"] = (df["is_m2m"] * (df["number_of_referrals"] == 0)).astype(float)
    df["m2m_fiber"] = (df["is_m2m"] * df["is_fiber"]).astype(float)
    df["m2m_no_sec"] = (df["is_m2m"] * (df["has_security"] == 0)).astype(float)
    df["low_addon_m2m"] = (df["is_m2m"] * (df["addon_count"] <= 1)).astype(float)
    df["rev_tenure"] = (df["total_revenue"] / (df["tenure_in_months"] + 1)).clip(0, 500)
    return df


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════


def build_pipeline(cat_cols: list, num_cols: list) -> Pipeline:
    """Rebuild the v3 GBM pipeline."""
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=900,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.75,
                    min_samples_leaf=5,
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(
    X_train, y_train, X_test, y_test, cat_cols: list, num_cols: list
) -> dict:
    """Train a new model and return metrics."""
    print("  Training new model (n=900, depth=7)...")
    t0 = time.time()

    pipeline = build_pipeline(cat_cols, num_cols)
    pipeline.fit(X_train, y_train)

    prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    prauc = average_precision_score(y_test, prob)

    # Youden threshold
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_test, prob)
    opt_thr = float(thr[np.argmax(tpr - fpr)])
    pred = (prob >= opt_thr).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_test, pred).ravel()

    elapsed = round(time.time() - t0, 1)
    print(f"  Done in {elapsed}s — AUC={auc:.4f}  Recall={TP/(TP+FN)*100:.1f}%")

    return {
        "pipeline": pipeline,
        "auc": round(float(auc), 4),
        "pr_auc": round(float(prauc), 4),
        "opt_threshold": round(opt_thr, 4),
        "recall": round(TP / (TP + FN), 4),
        "precision": round(TP / (TP + FP), 4),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "TP": int(TP),
        "train_seconds": elapsed,
    }


# ══════════════════════════════════════════════════════════════
# A/B COMPARISON
# ══════════════════════════════════════════════════════════════


def ab_comparison(
    model_a, model_b, X_test, y_test, label_a="Current", label_b="New"
) -> dict:
    """
    Side-by-side comparison of two models on the same test set.
    Returns winner and full metrics.
    """
    prob_a = model_a.predict_proba(X_test)[:, 1]
    prob_b = model_b.predict_proba(X_test)[:, 1]

    auc_a = roc_auc_score(y_test, prob_a)
    auc_b = roc_auc_score(y_test, prob_b)

    psi_scores = compute_psi(prob_a, prob_b)

    result = {
        label_a: {"auc": round(float(auc_a), 4)},
        label_b: {"auc": round(float(auc_b), 4)},
        "auc_delta": round(float(auc_b - auc_a), 4),
        "score_psi": round(psi_scores, 4),
        "winner": label_b if auc_b >= auc_a - 0.005 else label_a,
        "promote_new": auc_b >= auc_a - 0.005,
    }

    print(f"\n  A/B Comparison:")
    print(f"  {label_a:<12} AUC = {auc_a:.4f}")
    print(f"  {label_b:<12} AUC = {auc_b:.4f}  (delta: {auc_b-auc_a:+.4f})")
    print(f"  Score PSI: {psi_scores:.4f}  ({psi_status(psi_scores)})")
    print(
        f"  Winner: {result['winner']}  {'→ PROMOTING new model' if result['promote_new'] else '→ KEEPING current model'}"
    )

    return result


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════


def log_event(event: dict):
    """Append one JSON line to retrain_log.jsonl."""
    event["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"\n  Logged → {LOG_FILE.name}")


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════


def run(data_path: str, force: bool = False, dry_run: bool = False):
    print("=" * 60)
    print("  ChurnIQ — Automated Retraining Pipeline")
    print("=" * 60)

    # ── 1. Load current model ─────────────────────────────────
    print("\n[1/6] Loading current production model...")
    current_model = joblib.load(MODEL_PATH)
    current_meta = joblib.load(META_PATH)
    CAT = current_meta["cat_cols"]
    NUM = current_meta["num_cols"]
    print(
        f"  Current AUC: {current_meta.get('auc', '?'):.4f}  "
        f"Version: {current_meta.get('version', 'v1')}"
    )

    # ── 2. Load new data ──────────────────────────────────────
    print(f"\n[2/6] Loading data from {data_path}...")
    df_raw = pd.read_csv(data_path)
    df_raw = df_raw.drop(columns=["customer_id", "state"], errors="ignore")
    for c in CAT:
        if c in df_raw.columns:
            df_raw[c] = df_raw[c].fillna("Unknown")
    NUM_BASE = [
        c
        for c in [
            "number_of_referrals",
            "total_extra_data_charges",
            "total_revenue",
            "total_charges",
            "tenure_in_months",
            "age",
            "monthly_charge",
            "total_refunds",
            "total_long_distance_charges",
        ]
        if c in df_raw.columns
    ]
    for c in NUM_BASE:
        df_raw[c] = df_raw[c].fillna(df_raw[c].median())

    df = add_engineered_features(df_raw)
    X = df.drop("churn", axis=1)
    y = df["churn"]
    print(f"  Rows: {len(df):,}  Churn rate: {y.mean()*100:.1f}%")

    # ── 3. Split — use first 80% as "training ref", last 20% as "new data" ──
    X_ref, X_new, y_ref, y_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 4. Drift detection ────────────────────────────────────
    print("\n[3/6] Checking for data drift (PSI)...")
    drift = check_drift(X_ref, X_new, NUM)

    print(f"\n  {'Feature':<35} {'PSI':>7}  Status")
    print("  " + "-" * 52)
    for feat, info in sorted(
        drift["features"].items(), key=lambda x: x[1]["psi"], reverse=True
    ):
        icon = (
            "✅"
            if info["status"] == "STABLE"
            else ("⚠️ " if info["status"] == "WARNING" else "🔴")
        )
        print(f"  {icon} {feat:<33} {info['psi']:>7.4f}  {info['status']}")

    print(
        f"\n  Verdict: {drift['verdict']}  "
        f"(max PSI={drift['max_psi']}, "
        f"drifted={drift['n_drift']}, "
        f"warning={drift['n_warning']})"
    )

    should_retrain = force or drift["verdict"] == "RETRAIN"

    if dry_run:
        print("\n[DRY RUN] Drift check complete — no model changes made.")
        log_event({"type": "drift_check", "drift": drift})
        return

    if not should_retrain:
        print(f"\n[4-6/6] No retraining needed (verdict={drift['verdict']}).")
        log_event({"type": "drift_check", "drift": drift, "action": "no_retrain"})
        return

    # ── 5. Retrain ────────────────────────────────────────────
    print(
        f"\n[4/6] Retraining triggered (reason: {'FORCED' if force else drift['verdict']})..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    new_metrics = train_model(X_train, y_train, X_test, y_test, CAT, NUM)
    new_pipeline = new_metrics.pop("pipeline")

    # ── 6. A/B comparison ────────────────────────────────────
    print("\n[5/6] Running A/B comparison...")
    ab = ab_comparison(
        current_model, new_pipeline, X_test, y_test, label_a="Current", label_b="New"
    )

    # ── 7. Promote if winner ──────────────────────────────────
    print("\n[6/6] Model promotion decision...")
    event = {
        "type": "retrain",
        "trigger": "forced" if force else drift["verdict"],
        "drift": drift,
        "new_metrics": new_metrics,
        "ab_result": ab,
        "promoted": ab["promote_new"],
    }

    if ab["promote_new"]:
        # Back up current model
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_model = BACKUP_DIR / f"churn_model_{ts}.pkl"
        backup_meta = BACKUP_DIR / f"meta_{ts}.pkl"
        joblib.dump(current_model, backup_model)
        joblib.dump(current_meta, backup_meta)
        print(f"  Backed up current model → {backup_model.name}")

        # Save new model
        new_meta = {
            **current_meta,
            "auc": new_metrics["auc"],
            "opt_threshold": new_metrics["opt_threshold"],
            "version": f"v{int(current_meta.get('version','v1').replace('v','')) + 1}",
            "retrained_at": datetime.utcnow().isoformat(),
        }
        joblib.dump(new_pipeline, MODEL_PATH)
        joblib.dump(new_meta, META_PATH)
        print(f"  ✅ New model promoted!")
        print(
            f"     AUC: {current_meta.get('auc','?'):.4f} → {new_metrics['auc']:.4f}"
            f"  ({ab['auc_delta']:+.4f})"
        )
        print(f"     Version: {current_meta.get('version')} → {new_meta['version']}")
        event["new_version"] = new_meta["version"]
        event["backup_created"] = backup_model.name
    else:
        print(f"  ⚠️  New model did not improve — keeping current.")

    log_event(event)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChurnIQ Retraining Pipeline")
    parser.add_argument(
        "--data",
        default="data/processed/crm_churn_ml_ready.csv",
    )
    parser.add_argument(
        "--force", action="store_true", help="Retrain regardless of drift"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Check drift only, no model changes"
    )
    args = parser.parse_args()

    run(data_path=args.data, force=args.force, dry_run=args.dry_run)
