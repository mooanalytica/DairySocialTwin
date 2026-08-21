#!/usr/bin/env bash
set -euo pipefail

cd /home/hyw/DXW/Dashboard
source /home/hyw/.venvs/dcsna/bin/activate
export MPLCONFIGDIR=/home/hyw/DXW/Dashboard/.cache/matplotlib
exec python3 server.py --host 0.0.0.0 --port 2299
