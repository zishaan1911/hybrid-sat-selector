#!/usr/bin/env bash
# One-click training for Linux and macOS: ./train.sh   (options: ./train.sh --help)
# Sets up .venv with the right PyTorch build, then downloads the data, builds graphs,
# trains the encoder and draws the figures. Safe to re-run: every step resumes.
set -euo pipefail
cd "$(dirname "$0")"
for candidate in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    exec "$candidate" scripts/setup.py "$@"
  fi
done
echo "Python 3.11 or newer is needed: https://www.python.org/downloads/" >&2
exit 1
