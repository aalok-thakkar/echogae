# rescience-echogae

Reproduction of Alatawi, Sheth, and Liu (2023), [*Quantifying the Echo Chamber Effect: An Embedding Distance-based Approach*](https://arxiv.org/abs/2307.04668), submitted to [ReScience C](https://rescience.github.io/).

The reproduction classification is **Rp** (same code, same raw inputs, with only a minimal fitting-seed patch applied to the released source). On SXSW ten fits give a user-average Echo Chamber Score of $0.467$, within one standard deviation of the paper's $0.465$. The reproduction also documents that the paper's Equation 4 and the released notebook compute different graph-level quantities under the same name, and reports the size of the discrepancy on the audited topics.

The paper draft is at [paper/rescience/article.pdf](paper/rescience/article.pdf).

## Repository layout

```
paper/rescience/            ReScience C manuscript (LaTeX + metadata)
experiments/validation/     reproduction protocol and results
experiments/src/            Python scripts used by the validation
scripts/                    setup, run, and summarisation helpers
```

Two directories are gitignored and populated on first run:
- `experiments/vendor/` holds the pinned clone of the EchoGAE release (`scripts/setup.sh`).
- `experiments/build/venv_clean/` holds the isolated Python environment used for validation.

## Reproducing the study

The reproduction runs end to end on a single machine with Python 3.13, `make`, `xelatex`, `biber`, `git`, and `curl`.

1. **Clone this repository and the pinned EchoGAE release.**
   ```bash
   git clone https://github.com/aalok-thakkar/rescience-echogae
   cd rescience-echogae
   bash scripts/setup.sh
   ```
   The setup script clones [`faalatawi/echo-chamber-score`](https://github.com/faalatawi/echo-chamber-score) into `experiments/vendor/echo-chamber-score` and checks it out at commit `0626330`. All twelve raw graph, tweet, and label files are verified against the release's Git blob hashes.

2. **Create the pinned Python environment.**
   ```bash
   python3 -m venv experiments/build/venv_clean
   experiments/build/venv_clean/bin/python -m pip install -r experiments/validation/requirements.lock
   experiments/build/venv_clean/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"
   experiments/build/venv_clean/bin/python -m nltk.downloader stopwords
   ```

3. **Regenerate text features and run the crossed-seed validation.**
   ```bash
   experiments/build/venv_clean/bin/python scripts/run_echo_validation.py
   ```
   Outputs land in `experiments/validation/results/` as one JSON per topic, plus the design-level summary and the independent-environment checks.

4. **Build the paper.**
   ```bash
   cd paper/rescience
   make article
   ```
   Produces `article.pdf`.

## Study design

The validation protocol is specified in [experiments/validation/protocol.md](experiments/validation/protocol.md). The seed-parameterisation patch applied to the release is at [experiments/validation/seed_parameterization.patch](experiments/validation/seed_parameterization.patch). Source and raw-input hashes are at [experiments/validation/upstream/manifest.json](experiments/validation/upstream/manifest.json).

Each primary run is a crossing of ten Louvain seeds (42–51, resolution 0.05) and ten fitting seeds (42–51). Both graph-level averages ($S_{\mathrm{user}}$, $S_{\mathrm{community}}$) are computed on every seed pair. The notebook's own configuration corresponds to the cell at Louvain seed 42, fitting seed 42. One-community partitions are recorded as undefined rather than assigned a value.

The structural experiment applies two rewiring operations to the seed-42 partition: a closure-targeted swap that preserves every user's number of neighbours in every original community and reduces triangle count, and a degree-only swap that permits community connections to change. Each condition uses five graph realisations and three fits per realisation. The encoder is retrained on every graph.

## Classification

Rp (reproduction): same code, same raw inputs, minimal seed patch. The SXSW reproduction matches the paper within one standard deviation across ten fitting seeds under the released code's averaging rule.

## Communication with the original authors

The draft was shared with the original paper's corresponding author in advance of submission. Correspondence received before the editorial deadline will be noted in the final version of the article.

## Licence

The paper (everything under `paper/`) is released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0), matching ReScience C policy.

The reproduction code and scripts in `experiments/` and `scripts/` are released under the MIT licence. See `LICENSE-CODE`.

The upstream EchoGAE source retained under `experiments/vendor/echo-chamber-score/` is the authors' own code under its original licence; see that directory's `LICENSE` for terms.

## Citing

```bibtex
@article{GautamThakkar2026ReScienceECS,
  author  = {Vedant Gautam and Aalok Thakkar},
  title   = {{[Rp]} Reproducing EchoGAE's Echo Chamber Score: an averaging ambiguity between paper and code},
  journal = {ReScience C},
  year    = {2026},
  note    = {Under review}
}
```
