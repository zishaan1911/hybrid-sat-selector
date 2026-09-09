#!/usr/bin/env bash
# Fetch the ASlib scenarios used by this project into data/.
# Blob-filtered clone: the full history is large, the three scenarios are ~20 MB.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --filter=blob:none --depth 1 https://github.com/coseal/aslib_data.git "$TMP/aslib"
mkdir -p data
for scenario in SAT18-EXP SAT20-MAIN SAT03-16_INDU; do
  cp -r "$TMP/aslib/$scenario" data/
  echo "fetched $scenario"
done
