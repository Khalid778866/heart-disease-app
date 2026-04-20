# ── Base image ────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application ──────────────────────────────────────────
COPY . .

# ── Expose port ───────────────────────────────────────────────
EXPOSE 5000

# ── Run with Gunicorn (production-grade WSGI server) ──────────
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
