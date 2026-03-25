# ChurnIQ — Real-Time Churn Prediction Web App

**GBM Model | AUC 0.905 | Indian Telecom Dataset**

## Quick Start (Local)

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Deploy to Render.com (Free)

1. Push to GitHub: `git init && git add . && git commit -m "init" && git push`
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`
6. Click Deploy — your app is live!

## Deploy to Railway.app (Free)

1. Install CLI: `npm install -g @railway/cli`
2. `railway login && railway init && railway up`

## Deploy to Heroku

```bash
heroku create your-churniq-app
git push heroku main
heroku open
```

## Project Structure

```
webapp/
├── app.py              ← Flask backend + predict API
├── templates/
│   └── index.html      ← Full frontend (single file)
├── models/
│   ├── churn_model.pkl ← Trained GBM pipeline
│   ├── meta.pkl        ← AUC, threshold, column names
│   └── defaults.json   ← Form defaults + cat options
├── requirements.txt
├── Procfile
└── render.yaml
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/predict` | POST | JSON prediction |
| `/health` | GET | Model health check |

### /predict — Request Body
```json
{
  "contract": "Month-to-Month",
  "monthly_charge": 85,
  "tenure_in_months": 6,
  "internet_type": "Fiber Optic",
  "number_of_referrals": 0,
  ...
}
```

### /predict — Response
```json
{
  "probability": 78.4,
  "churn": 1,
  "tier": "HIGH RISK",
  "action": "Immediate action required...",
  "signals": [...],
  "net_value": 986.0
}
```
