#!/bin/bash
set -euo pipefail
SPICE_ROOT=${SPICE_ROOT:-$HOME/spice}
OUT="$HOME/spice_results.tar.gz"
cd "$SPICE_ROOT/data/data_efficiency"
tar czf "$OUT" --ignore-failed-read \
  n10/runs n100/runs n1000/runs n45000/runs results.csv 2>/dev/null || true
echo "→ $OUT  ($(du -h "$OUT" | cut -f1))"
echo "Web Console -> File Manager -> Download. After unpacking locally, each scale's coverage.csv resides under n{N}/runs/posttrain/"
