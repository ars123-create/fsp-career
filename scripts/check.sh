#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHONPATH=backend .venv/bin/pytest backend/tests -q
npm --prefix frontend run build
