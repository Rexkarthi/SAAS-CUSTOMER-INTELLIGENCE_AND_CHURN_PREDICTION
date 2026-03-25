<img width="1919" height="861" alt="Screenshot 2026-03-17 230934" src="https://github.com/user-attachments/assets/467bf2b4-7c7e-46e9-88e1-b2ebc12077fd" />
<img width="1915" height="649" alt="Screenshot 2026-03-17 230913" src="https://github.com/user-attachments/assets/7eae1dc8-e0bd-4bcb-992e-f101fe762825" />
<img width="1913" height="837" alt="Screenshot 2026-03-17 230734" src="https://github.com/user-attachments/assets/964035a7-ed21-47e8-97e1-dc916714c622" />
<img width="1915" height="763" alt="Screenshot 2026-03-17 230653" src="https://github.com/user-attachments/assets/c6ba6523-d52a-46ea-aa7b-1d1a1816697c" />
<img width="1778" height="847" alt="Screenshot 2026-03-17 230637" src="https://github.com/user-attachments/assets/ac22e381-daa7-4b2a-97a9-99facec2a9c6" />
<img width="1919" height="840" alt="Screenshot 2026-03-17 230619" src="https://github.com/user-attachments/assets/13149935-4261-4980-a1aa-bf76494b66b3" />
<img width="1917" height="748" alt="Screenshot 2026-03-17 230549" src="https://github.com/user-attachments/assets/361bd825-2eb4-494b-9637-c403b45b226b" />
<div align="center">

# 🔮 saas-churn-intelligence

### *Predict who leaves. Retain who matters. Protect your revenue.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-0.923_AUC-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-00c9a7?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production_Ready-00c9a7?style=flat-square)]()

**End-to-end ML pipeline that identifies at-risk telecom customers with 91.3% ROC-AUC and translates predictions into $3.6M+ estimated annual revenue retention.**

[📊 View Dashboard](#-live-dashboard) · [🚀 Quick Start](#-quick-start) · [📓 Notebooks](#-notebooks) · [📈 Results](#-results)

---

</div>

## 🎯 The Problem

Every month, **26.8% of telecom customers churn** — silently, before anyone notices.

Traditional approaches react *after* they leave. This project predicts churn **30+ days in advance** using CRM behavioral signals, enabling proactive retention at a fraction of the cost of acquiring a new customer.

| Metric | Without Model | With This Model |
|--------|:---:|:---:|
| Churners caught | 0% | **85%** |
| Monthly revenue protected | $0 | **$201K+** |
| Strategy | React after churn | Predict & retain |
| Cost per intervention | $0 saved | **$1,214 net saved** per churner |

---

## 📊 Live Dashboard

> Open `churn_intelligence_dashboard.html` in any browser — no Python, no server, no setup.

The standalone dashboard shows real-time model metrics, feature importance, risk tiers, and business ROI — built for stakeholder presentations.

---

## 🏆 Results

| Model | ROC-AUC | Accuracy | F1 (Churn) | PR-AUC |
|-------|:-------:|:--------:|:----------:|:------:|
| Logistic Regression | 0.851 | 74% | 0.63 | — |
| Random Forest | 0.860 | 81% | 0.62 | — |
| GBM (tuned) | 0.913 | 83% | 0.72 | 0.806 |
| **XGBoost (best)** | **0.923** | **85%** | **0.74** | **0.824** |

**Optimal threshold: 0.273** (Youden's J statistic — tuned for churn recall, not accuracy)

### Business Impact
```
Churners caught (TP)        :   611 / 720   (85% recall)
Est. revenue per save       :   $1,264      (18mo × $70.20/mo)
Intervention cost           :   $50/customer
────────────────────────────────────────────────────────
Net benefit (test set)      :   $722,000+
Scaled to full dataset      :   $3,616,723
vs. Rule-based approach     :   +$1.3M improvement
vs. Baseline (do nothing)   :   +$3.6M improvement
```

---

## 🗂 Project Structure

```
saas-churn-intelligence/
│
├── 📓 notebooks/
│   ├── 01_business_understanding.ipynb   # Problem framing, dataset overview
│   ├── 02_data_exploration.ipynb         # EDA, distributions, correlations
│   ├── 03_benchmark_comparison.ipynb     # LR vs RF baseline benchmarks
│   ├── 04_model_training.ipynb           # XGBoost + GBM tuning, SMOTE
│   ├── 05_model_interpretation.ipynb     # Feature importance, explainability ✅
│   └── 06_business_impact.ipynb          # ROI, CLTV segmentation, playbook ✅
│
├── 📊 data/
│   ├── processed/
│   │   └── crm_churn_ml_ready.csv        # 13,461 rows, 28 features, ML-ready
│   └── raw/
│       ├── train.csv                     # 4,225 rows, 52 features (CLTV, reasons)
│       ├── test.csv                      # 1,409 rows
│       └── validation.csv               # 1,409 rows
│
├── 🤖 models/
│   └── best_churn_model.pkl              # Saved GBM pipeline (sklearn)
│
├── 📦 outputs/
│   ├── test_risk_scored.csv              # Every test customer with risk tier
│   └── feature_importance.csv           # Full ranked feature list
│
├── 🌐 churn_intelligence_dashboard.html  # Standalone stakeholder dashboard
├── 🔮 predict.py                         # Production inference script
├── 📋 MODEL_CARD.md                      # Model documentation
└── 📄 README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/saas-churn-intelligence.git
cd saas-churn-intelligence
pip install -r requirements.txt
```

### 2. Run a Prediction (Single Customer)
```bash
python predict.py --customer '{
  "contract": "Month-to-Month",
  "monthly_charge": 89.50,
  "tenure_in_months": 8,
  "internet_type": "Fiber Optic",
  "number_of_referrals": 0,
  "total_charges": 716.0,
  "age": 42
}'
```

**Output:**
```
──────────────────────────────────────────
  ChurnIQ Prediction
──────────────────────────────────────────
  Customer Risk Score  :  0.847
  Risk Tier            :  🔴 HIGH RISK
  Recommendation       :  Immediate outreach — retention specialist
                          call + competitive contract upgrade offer
──────────────────────────────────────────
```

### 3. Score a Batch (CSV)
```bash
python predict.py --batch data/processed/crm_churn_ml_ready.csv \
                  --output outputs/scored_customers.csv
```

### 4. Open the Dashboard
```bash
open churn_intelligence_dashboard.html   # macOS
start churn_intelligence_dashboard.html  # Windows
```

---

## 📓 Notebooks

| # | Notebook | What's inside |
|---|----------|---------------|
| 01 | **Business Understanding** | Dataset profiling, churn cost framing, success metrics |
| 02 | **Data Exploration** | Missing values, distributions, correlation heatmap, churn segmentation |
| 03 | **Benchmark Comparison** | Logistic Regression + Random Forest baselines on IBM Telco dataset |
| 04 | **Model Training** | SMOTE balancing, XGBoost + GBM tuning, threshold optimization |
| 05 | **Model Interpretation** ✨ | Feature importance, manual SHAP-style waterfall plots, risk segmentation |
| 06 | **Business Impact** ✨ | Revenue at risk, CLTV segmentation, 3-strategy ROI comparison, executive summary |

---

## 🔍 Key Findings

### Top Churn Drivers (Feature Importance)

| Rank | Feature | Importance | Business Insight |
|------|---------|:----------:|-----------------|
| 1 | `contract_Month-to-Month` | 29.1% | M2M churn rate: **45.5%** vs 2.4% Two Year |
| 2 | `monthly_charge` | 11.9% | Churners pay **$12/mo more** on average |
| 3 | `total_revenue` | 11.0% | Short-tenure customers = lower total spend |
| 4 | `age` | 10.1% | Younger customers churn more |
| 5 | `total_charges` | 8.0% | Proxy for tenure × spend |
| 6 | `total_long_distance_charges` | 5.6% | Usage signal |
| 7 | `internet_type_Fiber Optic` | 3.5% | 40.1% churn rate — highest risk segment |
| 8 | `number_of_referrals` | 3.1% | Referrers almost never churn |

### Critical Business Insights

**1. Satisfaction Score is a leading indicator**
- Score 1–2 → **100% churn rate** in training data
- Deploy quarterly satisfaction surveys as early warning system
- Intervene *before* scores drop to 1–2

**2. Month-to-Month is the retention battleground**
- 2,193 M2M customers (52% of base) churning at 45.5%
- Migrating M2M → One Year with a 2-month incentive (~$140 cost) saves ~$1,264 per customer
- **ROI: 9× return on migration incentive**

**3. Competitor threat dominates churn reasons**
- 43% of churners left because a competitor made a better offer
- Proactive competitive pricing review required — reactive pricing loses customers

**4. Fiber Optic paradox**
- Highest-paying segment (avg $89/mo), highest churn rate (40.1%)
- These customers have the most options — must be retained with premium service SLAs

---

## ⚙️ ML Pipeline

```
Raw CRM Data (13,461 rows)
    │
    ▼
Preprocessing
    ├── Null imputation (median for numeric, 'Unknown' for categorical)
    ├── Drop ID columns (customer_id, state)
    └── StandardScaler (numeric) + OneHotEncoder (categorical)
    │
    ▼
Class Balancing → SMOTE (26.8% minority → balanced)
    │
    ▼
Model Training
    ├── Logistic Regression  (baseline)     AUC: 0.851
    ├── Random Forest        (benchmark)    AUC: 0.860
    ├── GBM (sklearn)        (production)   AUC: 0.913
    └── XGBoost              (best)         AUC: 0.923
    │
    ▼
Threshold Optimization → Youden's J Statistic → 0.273
    │
    ▼
Risk Scoring → High / Medium / Low tiers
    │
    ▼
CRM Integration → Automated retention workflow trigger
```

---

## 📦 Requirements

```txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
shap>=0.44          # optional — for SHAP waterfall plots
```

Install:
```bash
pip install -r requirements.txt
```

---

## 🗃 Dataset

**Source:** Indian Telecom CRM dataset (synthetic, privacy-safe)

| Property | Value |
|----------|-------|
| Total rows | 13,461 |
| Features | 28 |
| Target | `churn` (binary: 0 = stayed, 1 = churned) |
| Churn rate | 26.8% (class imbalanced → handled with SMOTE) |
| Missing values | 6 features (streaming/internet services) — imputed |
| States covered | Delhi, Maharashtra, Karnataka, Tamil Nadu, + others |

**Rich CRM dataset** (`train.csv`) additionally includes: CLTV, Satisfaction Score, Churn Reason, Churn Category, City, Latitude/Longitude, Offer type, Senior Citizen flag.

---

## 🔮 Future Work

- [ ] **SHAP integration** — per-customer waterfall explanations in production
- [ ] **CLTV-weighted scoring** — prioritize high-value churners automatically
- [ ] **Real-time API** — FastAPI endpoint for CRM system integration
- [ ] **Quarterly retraining pipeline** — automated with Airflow/Prefect
- [ ] **Survival analysis** — predict *when* a customer will churn, not just *if*
- [ ] **A/B test framework** — measure actual retention lift from model-driven campaigns

---

## 👤 Author

Built as a production-grade ML portfolio project demonstrating end-to-end data science:
business framing → EDA → feature engineering → model training → explainability → business impact.

---

<div align="center">

**If this project helped you, ⭐ star the repo!**

*"The best model is the one that saves the most revenue — not the one with the highest AUC."*

</div>
