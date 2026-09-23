#!/bin/bash
# ── YOLO Detection Suite Launcher ─────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

THEME=(
  --theme.base dark
  --theme.primaryColor "#58a6ff"
  --theme.backgroundColor "#0d1117"
  --theme.secondaryBackgroundColor "#161b22"
  --theme.textColor "#c9d1d9"
  --server.headless false
  --browser.gatherUsageStats false
)

echo "📦 Checking dependencies..."
pip install -r requirements.txt -q

echo ""
echo "┌──────────────────────────────────────────────────┐"
echo "│         YOLO Detection Suite — Jetson Orin       │"
echo "├──────────────────────────────────────────────────┤"
echo "│  1) Live Detection UI       (app.py)             │"
echo "│  2) Scientific Benchmark UI (benchmark.py)       │"
echo "│  3) RPi Camera Sender       (rpi_sender.py)      │"
echo "│  4) Terminal Benchmark CLI  (cli_benchmark.py)   │"
echo "└──────────────────────────────────────────────────┘"
echo ""
read -rp "Choose [1/2/3/4] (default: 4): " choice

case "$choice" in
  1)
    echo "🚀 Starting Live Detection UI..."
    streamlit run app.py "${THEME[@]}"
    ;;
  2)
    echo "🔬 Starting Scientific Benchmark UI..."
    streamlit run benchmark.py "${THEME[@]}"
    ;;
  3)
    echo "📷 Starting RPi Camera Sender..."
    streamlit run rpi_sender.py "${THEME[@]}"
    ;;
  *)
    echo "⚡ Starting Terminal Benchmark CLI..."
    python3 cli_benchmark.py --frames 150 --runs 3 --warmup 10 --interval 1000
    ;;
esac
