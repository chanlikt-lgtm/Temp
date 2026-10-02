#!/usr/bin/env bash
# Regenerate SHA256SUMS.txt over the Python sources, relative to this directory.
# Run from the bundle's code/ directory:  bash gen_checksums.sh
# Do this ONLY after all code changes for a release are final.
set -euo pipefail
cd "$(dirname "$0")"
find . -type f -name '*.py' -not -path '*/__pycache__/*' -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS.txt
echo "Wrote SHA256SUMS.txt ($(grep -c . SHA256SUMS.txt) files)."
# Verify with:  sha256sum -c SHA256SUMS.txt
