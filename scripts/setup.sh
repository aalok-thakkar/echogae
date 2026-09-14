#!/usr/bin/env bash
# Clone the three audited releases into experiments/vendor/.
# Only paths needed by the audit are checked out; the full repos are large.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments/vendor
cd experiments/vendor

# EchoGAE release (Alatawi, Sheth, Liu, KDD 2023)
if [ ! -d echo-chamber-score ]; then
  git clone --depth 1 https://github.com/echo-chamber-score/echo-chamber-score.git
fi

# InfoVGAE release (Li et al., SIGIR 2022)
if [ ! -d InfoVGAE ]; then
  git clone --depth 1 https://github.com/HKUST-KnowComp/InfoVGAE.git
fi

# TIMME release (Xiao et al., KDD 2020). Sparse-checkout to avoid the
# ~250 MB history and LFS blobs.
if [ ! -d TIMME ]; then
  git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/PatriciaXiao/TIMME.git
  cd TIMME
  git sparse-checkout init --cone
  git sparse-checkout set code data/PureP data/P50 data/additional_labels
  git checkout
  cd ..
fi

# SNAP Reddit hyperlinks (Kumar, Hamilton, Leskovec, Jurafsky, 2018)
mkdir -p reddit
if [ ! -f reddit/hyperlinks-body.tsv ]; then
  curl -fsSL -o reddit/hyperlinks-body.tsv.bz2 \
    https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv.bz2
  bunzip2 reddit/hyperlinks-body.tsv.bz2
fi

# Waller-Anderson community scores (bundled small file, expected at reddit/wa/data/scores.csv)
# Download instructions in reddit/README.md.

# Adamic-Glance 2005 political blogs
mkdir -p polblogs
if [ ! -f polblogs/polblogs.gml ]; then
  curl -fsSL -o polblogs/polblogs.zip \
    https://websites.umich.edu/~mejn/netdata/polblogs.zip
  (cd polblogs && unzip -q polblogs.zip)
fi

echo "Vendor setup complete."
