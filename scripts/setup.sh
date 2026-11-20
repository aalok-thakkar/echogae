#!/usr/bin/env bash
# Clone the EchoGAE release at the pinned commit used by the validation.
# See experiments/validation/upstream/manifest.json for the verified file hashes.
set -euo pipefail

cd "$(dirname "$0")/.."

COMMIT=062633017c9ea886b15d099f8370bbef447ecb5c
DEST=experiments/vendor/echo-chamber-score

mkdir -p experiments/vendor
if [ ! -d "$DEST" ]; then
  git clone https://github.com/faalatawi/echo-chamber-score.git "$DEST"
fi
git -C "$DEST" checkout "$COMMIT"

# Verify the twelve raw graph, tweet, and label files against the pinned remote
# Git blob hashes. Fails loudly if any file drifts.
if [ -f scripts/fetch_echo_source.py ]; then
  python3 scripts/fetch_echo_source.py
fi

echo "EchoGAE release pinned at $COMMIT under $DEST."
