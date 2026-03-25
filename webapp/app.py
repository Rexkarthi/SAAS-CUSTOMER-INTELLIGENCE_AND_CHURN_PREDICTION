"""
ChurnIQ — Real-Time Churn Prediction Web App
GBM Model | AUC 0.905 | Indian Telecom Dataset
"""
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd, numpy as np, joblib, json, os, io, traceback

app = Flask(__name__)

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request', 'details': str(e)}), 400

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error', 'details': str(e)}), 500

# ── Load artefacts ────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, 'models/churn_model.pkl'))
meta  = joblib.load(os.path.join(BASE, 'models/meta.pkl'))
with open(os.path.join(BASE, 'models/defaults.json')) as f:
    cfg = json.load(f)

CAT_COLS  = meta['cat_cols']
NUM_COLS  = meta['num_cols']
THRESHOLD = meta['opt_threshold']
MODEL_AUC = meta['auc']
DEFAULTS  = cfg['defaults']
CAT_OPTS  = cfg['cat_options']

CLEAN_OPTS = {}
for col, opts in CAT_OPTS.items():
    clean = [o for o in opts if o not in ('0', '1', 'Unknown')]
    CLEAN_OPTS[col] = clean if clean else ['Yes', 'No']


# ── Shared scoring logic ──────────────────────────────────────
def score_row(row_dict):
    """Score a single customer dict. Returns (prob, tier, tcls, action, signals, net_value)."""
    row = {}
    for col in NUM_COLS:
        try:    row[col] = float(row_dict.get(col, DEFAULTS.get(col, 0)))
        except: row[col] = float(DEFAULTS.get(col, 0))
    for col in CAT_COLS:
        row[col] = str(row_dict.get(col, DEFAULTS.get(col, 'No')))

    prob = float(model.predict_proba(pd.DataFrame([row]))[0, 1])

    if prob >= 0.60:
        tier, tcls, action = "HIGH RISK",   "high",   "Immediate action — retention specialist + contract upgrade offer within 48h."
    elif prob >= 0.30:
        tier, tcls, action = "MEDIUM RISK", "medium", "Proactive outreach — loyalty discount (10-15%) within 1 week."
    else:
        tier, tcls, action = "LOW RISK",    "low",    "Nurture — referral programme + upsell next billing cycle."

    signals = []
    if row.get('contract') == 'Month-to-Month':
        signals.append({"label": "Contract Type",    "desc": "Month-to-Month → 45.5% churn rate", "level": "high"})
    if float(row.get('monthly_charge', 0)) > 80:
        signals.append({"label": "High Charge",       "desc": f"${row['monthly_charge']:.0f}/mo above $80 bracket", "level": "high"})
    if float(row.get('tenure_in_months', 99)) < 12:
        signals.append({"label": "New Customer",      "desc": f"{row['tenure_in_months']:.0f} months — critical window", "level": "high"})
    if row.get('internet_type') == 'Fiber Optic':
        signals.append({"label": "Fiber Optic",       "desc": "40.1% churn — highest of all types", "level": "medium"})
    try:
        if int(float(row.get('number_of_referrals', 1))) == 0:
            signals.append({"label": "Zero Referrals","desc": "No referrals — low engagement signal", "level": "medium"})
    except: pass
    if row.get('online_security') in ('No', '0'):
        signals.append({"label": "No Security",       "desc": "Missing add-on = lower stickiness", "level": "low"})

    revenue_save = float(row.get('monthly_charge', 70)) * 18
    net_value    = prob * (revenue_save - 50) - (1 - prob) * 50

    return prob, tier, tcls, action, signals, round(net_value, 0), round(revenue_save, 0)


# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
        cat_cols=CAT_COLS, num_cols=NUM_COLS,
        cat_opts=CLEAN_OPTS, defaults=DEFAULTS,
        model_auc=round(MODEL_AUC, 4),
        threshold=round(THRESHOLD * 100, 1))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True) or {}
        prob, tier, tcls, action, signals, net_value, revenue_save = score_row(data)
        return jsonify({
            'probability':  round(prob * 100, 1),
            'churn':        int(prob >= THRESHOLD),
            'tier':         tier,
            'tier_class':   tcls,
            'action':       action,
            'signals':      signals,
            'net_value':    net_value,
            'threshold':    round(THRESHOLD * 100, 1),
            'revenue_save': revenue_save,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()[-400:]}), 500


@app.route('/batch', methods=['POST'])
def batch():
    """
    Score a CSV of customers.
    Accepts: multipart/form-data with field 'file' (CSV)
    Returns: JSON summary + scored rows
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded. Send a CSV in field named "file".'}), 400

        f  = request.files['file']
        if not f.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a .csv'}), 400

        df = pd.read_csv(f)
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

        if len(df) == 0:
            return jsonify({'error': 'CSV is empty'}), 400
        if len(df) > 50000:
            return jsonify({'error': 'Max 50,000 rows per batch'}), 400

        results = []
        for _, row_raw in df.iterrows():
            prob, tier, tcls, action, signals, net_value, rev_save = score_row(row_raw.to_dict())
            results.append({
                'churn_probability': round(prob * 100, 1),
                'risk_tier':         tier,
                'tier_class':        tcls,
                'predicted_churn':   int(prob >= THRESHOLD),
                'net_value':         net_value,
                'revenue_at_risk':   round(float(row_raw.get('monthly_charge', 70)) * 18 * prob, 2),
                'top_signal':        signals[0]['label'] if signals else 'None',
                'action':            action,
            })

        # Summary stats
        high   = sum(1 for r in results if r['tier_class'] == 'high')
        medium = sum(1 for r in results if r['tier_class'] == 'medium')
        low    = sum(1 for r in results if r['tier_class'] == 'low')
        total_rev_at_risk = sum(r['revenue_at_risk'] for r in results)
        total_net_value   = sum(r['net_value']        for r in results)
        avg_prob          = np.mean([r['churn_probability'] for r in results])

        # Build downloadable scored CSV
        out_df = df.copy()
        for key in ['churn_probability','risk_tier','predicted_churn',
                    'net_value','revenue_at_risk','top_signal','action']:
            out_df[key] = [r[key] for r in results]

        csv_buffer = io.StringIO()
        out_df.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()

        return jsonify({
            'total':             len(results),
            'high':              high,
            'medium':            medium,
            'low':               low,
            'avg_probability':   round(float(avg_prob), 1),
            'total_rev_at_risk': round(total_rev_at_risk, 0),
            'total_net_value':   round(total_net_value, 0),
            'results':           results[:5000],   # preview cap
            'csv':               csv_str,
            'columns':           df.columns.tolist(),
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()[-500:]}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'auc': round(MODEL_AUC, 4)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
