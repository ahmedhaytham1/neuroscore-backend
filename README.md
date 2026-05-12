# NeuroScore Parkinson Video AI Backend

FastAPI backend for three video tests:

- `POST /analyze/finger`
- `POST /analyze/romberg`
- `POST /analyze/tandem`
- `GET /health`

## Local run

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Render deploy

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Python runtime is pinned in `runtime.txt`.
