#!/usr/bin/env bash
# Download Metrica Sports open sample data (3 matches) into data/metrica.
# Source: https://github.com/metrica-sports/sample-data (please credit Metrica Sports)
set -euo pipefail
DEST="${1:-data/metrica}"
if [ -d "$DEST/data" ]; then echo "already downloaded: $DEST"; exit 0; fi
git clone --depth 1 https://github.com/metrica-sports/sample-data.git "$DEST"
echo "done: $DEST/data"
