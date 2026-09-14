# Graph Neural Networks Do Not Measure Echo Chambers

ICWSM 2026 submission: an audit of three GNN-based polarisation estimators (EchoGAE, InfoVGAE, TIMME) against reproducibility, response, and stability, on ten datasets.

Latest build: `paper/build/icwsm_rev.pdf`.

## Layout

- `paper/` — LaTeX source (`main.tex`, `appendix.tex`, `macros.tex`, `references.bib`), AAAI-25 style, build PDF in `paper/build/`.
- `experiments/src/` — reproduction, rewire, bootstrap, direct-estimator, forensics scripts.
- `experiments/results/` — JSON outputs and run logs. This is the reproduction proof.
- `experiments/vendor/` — audited upstream releases (gitignored). Cloned by `scripts/setup.sh`.
- `theory/` — proof drafts and witness constructions that feed the appendix.
- `references/`, `data/` — bibliography working notes, dataset landing pages.
- `tasks/todo.md`, `tasks/lessons.md` — plan and correction log.

## Reproducing

```bash
./scripts/setup.sh              # clones EchoGAE / InfoVGAE / TIMME / SNAP Reddit / polblogs
cd experiments
python3 -m pip install -r requirements.txt   # or use the pinned venv
python3 src/multi_seed_reproduction.py --n_seeds 10
python3 src/timme_multi_seed.py --n_seeds 10
python3 src/infovgae_native.py --config InfoVGAE_election_3D --n_seeds 10
python3 src/direct_estimators.py
python3 src/assortativity_bootstrap.py
```

Each script writes to `experiments/results/<script_name>/`.

## Rebuilding the paper

```bash
cd paper
pdflatex -interaction=nonstopmode -output-directory=build main.tex
(cd build && bibtex main)
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex
```
