#!/usr/bin/env bash
# Installs the vendored tinker-cookbook (which contains the paint_rl recipe)
# with the recipe's dependencies, plus Playwright's Chromium for rendering.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
pip install -e "$HERE/tinker-cookbook[paint-rl]"
python -m playwright install --with-deps chromium
if [ ! -f "$HERE/.env" ]; then
  echo "No $HERE/.env yet — copy .env.example and fill in TINKER_API_KEY, GEMINI_API_KEY."
fi
echo "Installed. Run: python -m tinker_cookbook.recipes.paint_rl.train log_path=/tmp/paint_rl/verifier"
