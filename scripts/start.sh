#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python scripts/setup.py
.venv/bin/pip install -q -r backend/requirements.lock.txt
npm --prefix frontend ci --no-audit --no-fund
npm --prefix frontend run build
.venv/bin/alembic upgrade head
export PYTHONPATH="$PWD/backend"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8018 --no-access-log
