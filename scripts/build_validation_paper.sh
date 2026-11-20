#!/usr/bin/env bash
# Rebuild only the corrected study; works in the anonymous artifact.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/plot_echo_design.py
python3 scripts/summarize_echo_validation.py
python3 scripts/plot_echo_validation.py
python3 scripts/write_echo_validation_text.py
cd paper
mkdir -p build
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=build main.tex > build/pass1.log
(cd build && BIBINPUTS=..: BSTINPUTS=..: bibtex main > bib.log)
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=build main.tex > build/pass2.log
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=build main.tex > build/pass3.log
if rg -n 'Overfull|undefined|multiply defined' build/main.log; then
  echo 'Resolve LaTeX overflow or reference warnings before delivery.' >&2
  exit 1
fi
cp build/main.pdf build/icwsm_rev.pdf
printf 'Built paper/build/icwsm_rev.pdf\n'
