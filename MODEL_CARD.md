# 📋 Model Card — ChurnIQ Churn Prediction Model

> *Following the Model Card framework (Mitchell et al., 2019)*

---

## Model Details

| Property | Value |
|----------|-------|
| **Model name** | ChurnIQ Churn Predictor v1.0 |
| **Model type** | Gradient Boosted Decision Trees (GBM) |
| **Framework** | scikit-learn `GradientBoostingClassifier` + XGBoost |
| **Task** | Binary classification — churn (1) vs. retained (0) |
| **Output** | Churn probability [0–1] + Risk tier (High / Medium / Low) |
| **Version** | 1.0.0 |
| **Date** | March 2026 |
| **Project** | [saas-churn-intelligence](https://github.com/yourusername/saas-churn-intelligence) |

---

## Intended Use

### Primary Use Cases ✅
- **Proactive retention campaigns** — identify at-risk customers 30+ days before expected churn
- **CRM prioritization** — rank customer success team's outreach queue by churn probability
- **Revenue forecasting** — estimate expected churn revenue loss in upcoming quarter
- **Offer personalization** — route high-risk customers to appropriate retention offers

### Secondary Use Cases ✅
- Internal analytics and churn driver reporting
- Satisfaction score early warning system validation
- Contract-type migration campaign targeting

### Out-of-Scope Uses ❌
- **Credit scoring or financial eligibility decisions** — model is not designed for this
- **Individual-level legal or regulatory decisions** — predictions are probabilistic, not deterministic
- **Non-telecom sectors** without retraining and validation on domain-specific data
- **Real-time fraud detection** — model is not calibrated or optimized for this task

---

## Training Data

### Dataset
| Property | Value |
|----------|-------|
| **Name** | Indian Telecom CRM Dataset |
| **Total rows** | 13,461 customers |
| **Features used** | 26 (after dropping `customer_id`, `state`) |
| **Target** | `churn` (binary: 0 = stayed, 1 = churned) |
| **Churn rate** | 26.8% (class imbalanced) |
| **Train split** | 80% (10,769 rows) |
| **Test split** | 20% (2,692 rows) — stratified |

### Features
| Category | Features |
|----------|----------|
| **Contractual** | `contract`, `payment_method`, `paperless_billing` |
| **Financial** | `monthly_charge`, `total_charges`, `total_revenue`, `total_refunds`, `total_extra_data_charges` |
| **Service** | `internet_service`, `internet_type`, `phone_service`, `multiple_lines` |
| **Add-ons** | `online_security`, `online_backup`, `device_protection_plan`, `streaming_*`, `unlimited_data` |
| **Behavioral** | `number_of_referrals`, `total_long_distance_charges`, `tenure_in_months` |
| **Demographic** | `age`, `gender`, `married` |

### Preprocessing
- **Missing values:** Categorical → `"Unknown"`, Numeric → median imputation
- **Encoding:** OneHotEncoder (categorical), StandardScaler (numeric)
- **Class balancing:** SMOTE applied to training set only (never to test/validation)
- **Dropped columns:** `customer_id` (identifier), `state` (high-cardinality geography, leakage risk)

---

## Model Architecture

### Algorithm
Gradient Boosting Machine (scikit-learn `GradientBoostingClassifier`)

### Hyperparameters
```python
GradientBoostingClassifier(
    n_estimators  = 300,
    max_depth     = 5,
    learning_rate = 0.05,
    subsample     = 0.8,
    random_state  = 42
)
```

### Pipeline
```
ColumnTransformer (StandardScaler + OneHotEncoder)
        │
        ▼
GradientBoostingClassifier
        │
        ▼
Threshold: 0.2728 (Youden's J statistic)
```

### Threshold Selection
The default probability threshold (0.5) was overridden using **Youden's J statistic** (maximizes TPR − FPR), resulting in threshold = **0.2728**.

**Rationale:** In churn prediction, the cost of a missed churner ($1,264 revenue lost) far outweighs the cost of a false positive ($50 intervention cost). Lower threshold → higher recall → better business outcome.

---

## Evaluation Results

### Performance Metrics (Test Set, n=2,692)

| Metric | Value |
|--------|:-----:|
| **ROC-AUC** | **0.913** |
| **PR-AUC** | **0.806** |
| **Overall Accuracy** | 83% |
| **Churn Precision** | 63% |
| **Churn Recall** | 85% |
| **Churn F1** | 0.72 |
| **No-Churn Precision** | 94% |
| **No-Churn Recall** | 82% |

### Confusion Matrix (threshold = 0.2728)
```
                 Predicted: Stay   Predicted: Churn
Actual: Stay          1,614              359        ← False Positives (FP)
Actual: Churn           109              611        ← True Positives (TP)  ★
```

### Model Comparison
| Model | ROC-AUC | Notes |
|-------|:-------:|-------|
| Logistic Regression | 0.851 | Baseline |
| Random Forest | 0.860 | Good baseline, lower recall |
| **GBM (production)** | **0.913** | **Selected for deployment** |
| XGBoost | 0.923 | Best AUC, same pipeline |

### Business Impact
| Scenario | Net Value |
|----------|----------:|
| Baseline (no model) | $0 |
| Rule-based (all M2M customers) | ~$2.3M |
| **ML Model (this model)** | **$3.6M+** |

---

## Limitations & Known Issues

### Data Limitations
- **Geographic bias:** Dataset covers Indian telecom customers only. Churn patterns may differ significantly in other markets (pricing structures, regulations, competitive landscape)
- **Temporal snapshot:** Dataset is cross-sectional — does not capture customer trajectory over time. Survival analysis would improve predictions
- **Missing values:** 6 features have 10–22% missing data (streaming and internet add-on services). Imputed with `"Unknown"` — may introduce noise
- **Synthetic data:** Dataset is privacy-safe synthetic data. Real-world performance may vary

### Model Limitations
- **No causal inference:** High feature importance ≠ causation. `monthly_charge` is predictive but cutting prices randomly is not guaranteed to reduce churn
- **CLTV not used:** Customer Lifetime Value is available in the raw CRM data but not included in this model version. Incorporating CLTV would improve business prioritization
- **Static model:** Model does not update with real-time customer behavior. Recommend quarterly retraining
- **Satisfaction score not available at scoring time:** Satisfaction score (strongest signal in EDA) requires a survey — cannot be used in real-time batch scoring without survey integration

### Potential Failure Modes
- **Distribution shift:** If marketing launches a new offer type or the company expands to a new region, model performance may degrade — monitor AUC monthly
- **New internet types:** If a new `internet_type` category is introduced, it will be handled as `"Unknown"` by the OHE — retrain promptly
- **Adversarial input:** Model has no input validation layer — malformed or malicious inputs will produce silently incorrect predictions. Wrap with input schema validation in production

---

## Fairness & Bias Considerations

### Demographic Features in Model
The model includes `age`, `gender`, and `married` as features. These were included because they show predictive value in EDA, but their use warrants careful consideration:

| Feature | Predictive Signal | Fairness Risk | Mitigation |
|---------|:-----------------:|:-------------|:-----------|
| `age` | High (10.1%) | Age-based discrimination in offer eligibility | Do not use to deny service — only to tailor retention approach |
| `gender` | Low (<1%) | Gender-based pricing risk | Monitor retention offer conversion rates by gender |
| `married` | Low (~0.8%) | Proxy for household income | Audit retention outcomes by marital status quarterly |

### Recommended Fairness Audits
- [ ] Audit retention offer acceptance rates by `gender`, `age_group`, `married`
- [ ] Ensure High-risk tier distribution does not disproportionately exclude protected groups from beneficial retention offers
- [ ] Track whether false negative rate (missed churners) differs significantly across demographic groups

---

## Ethical Considerations

### Appropriate Use
- ✅ Use predictions to **prioritize** outreach and **personalize** retention offers
- ✅ Use feature importance to guide **product and pricing strategy**
- ✅ Use business impact analysis to justify **retention budget allocation**

### Inappropriate Use
- ❌ Do not use predictions to **restrict service access** or impose **punitive conditions** on high-risk customers
- ❌ Do not use demographic features as the **primary** basis for offer eligibility
- ❌ Do not treat model output as a **definitive** determination of whether a customer will churn — it is a probability

### Human Oversight
Model predictions should augment, not replace, human judgment. Recommended oversight:
- Customer success team reviews all "High Risk" outreach before execution
- Quarterly business review of model-driven vs. baseline campaign performance
- Annual full model audit including fairness metrics

---

## Deployment & Monitoring

### Deployment
```bash
# Single prediction
python predict.py --customer '{"contract":"Month-to-Month","monthly_charge":89.5}'

# Batch scoring
python predict.py --batch data/customers.csv --output outputs/scored.csv

# Demo
python predict.py --demo
```

### Monitoring Recommendations
| Metric | Frequency | Alert Threshold |
|--------|:---------:|:---------------:|
| ROC-AUC (production) | Monthly | < 0.88 |
| Churn recall | Monthly | < 0.78 |
| Feature distribution drift | Monthly | PSI > 0.2 |
| Prediction distribution shift | Weekly | Mean prob shift > 5% |
| Retention campaign lift | Quarterly | < 15% vs. control |

### Retraining Trigger
Retrain model when any of:
- AUC drops below 0.88 on recent data
- New product/pricing changes are launched
- Customer base composition changes significantly (e.g., new market entry)
- More than 6 months have elapsed since last training

---

## Citation

If you use this model or methodology in your work:

```bibtex
@misc{churniq2026,
  title   = {ChurnIQ: End-to-End Customer Churn Prediction for Telecom},
  author  = {[Your Name]},
  year    = {2026},
  url     = {https://github.com/yourusername/saas-churn-intelligence}
}
```

---

## References

- Mitchell, M., et al. (2019). *Model Cards for Model Reporting.* FAccT 2019.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- Chawla, N. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
- Youden, W.J. (1950). *Index for rating diagnostic tests.* Cancer.

---

*Model Card version 1.0 | Last updated: March 2026 | saas-churn-intelligence*
