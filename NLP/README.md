# Automated Short Answer Grading (ASAG) — Minimal Prototype

This repository contains a minimal Python prototype for Automated Short Answer Grading (ASAG) and a Flask server to serve predictions.

Files created:
- `asag/` — package with data, features, train, predict modules.
- `app.py` — Flask app exposing `/health` and `/predict` endpoints.
- `data/train.csv` — small synthetic dataset so you can run an end-to-end demo immediately.
- `requirements.txt` — Python dependencies.

Quick start (PowerShell)
1. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Train models (writes to `models/`):

```powershell
python -m asag.train --train-baseline
python -m asag.train --train-sbert
```

3. Run the Flask server:

```powershell
# run directly
python app.py
# or using Flask CLI
set FLASK_APP=app.py; flask run --port 5000
```

4. Predict (PowerShell example):

```powershell
$json = '{"student_answer":"Plants make food using light and water","model_answer":"Photosynthesis is the process by which plants convert light energy into chemical energy using CO2 and water."}'
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -Body $json -ContentType 'application/json'
```

If you want, I can now:
- run a quick training on the synthetic data and show the printed metrics and a sample API response, or
- switch to using the ASAP dataset (you can upload it), or
- implement QWK threshold optimization and per-question models.
