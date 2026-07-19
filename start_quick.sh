#!/usr/bin/env bash
set -euo pipefail
python -m streamlit run quick_app.py --server.address 0.0.0.0
