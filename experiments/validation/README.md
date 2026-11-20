# Controlled EchoGAE validation

This study corrects three differences between the earlier exploratory audit and the released experiment: largest-component filtering, encoder width, and graph-score averaging. It preserves all historical outputs. Read `protocol.md` before interpreting results.

## Verified inputs and source

- Public source: `faalatawi/echo-chamber-score`, commit `062633017c9ea886b15d099f8370bbef447ecb5c`.
- `upstream/manifest.json` checks nine source/document files and twelve raw data files against the public snapshot. `controlled/` includes only the model/metric files and the upstream MIT licence. `seed_parameterization.patch` records the two seed-related source changes.
- Features are regenerated locally from released tweets with the upstream preprocessor and Sentence-BERT snapshot `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Original caches are never overwritten.
- `results/upstream_loader_check.json` compares the prepared input against the unchanged upstream loader. `results/clean_environment_check.json` documents the independent environment and unchanged-upstream reference execution.

## Environment and commands

From the repository root, create an isolated Python 3.13 environment and install the exact lock:

```bash
python3 -m venv experiments/build/venv_clean
experiments/build/venv_clean/bin/python -m pip install -r experiments/validation/requirements.lock
```

The source and original data must be available under `experiments/vendor/echo-chamber-score/` at the pinned commit. The full source snapshot can be obtained from the public repository. The seed-controlled model code used for validation is included separately; do not substitute upstream HEAD.

For a new checkout, obtain the upstream material with `git clone https://github.com/faalatawi/echo-chamber-score.git experiments/vendor/echo-chamber-score` and `git -C experiments/vendor/echo-chamber-score checkout 062633017c9ea886b15d099f8370bbef447ecb5c`. Do not overwrite an existing local checkout. Run `python3 scripts/fetch_echo_source.py` to verify its raw files against the pinned remote Git blobs.

Download the immutable model and the NLTK corpus required at import by the upstream preprocessor:

```bash
experiments/build/venv_clean/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"
experiments/build/venv_clean/bin/python -m nltk.downloader stopwords
```

The BERT preprocessing path does not use stopwords. `ECHO_MODEL_SNAPSHOT` can override the default local snapshot path. Legacy feature-cache comparisons are optional; reproduction does not require those caches.

```bash
experiments/build/venv_clean/bin/python scripts/verify_echo_features.py
experiments/build/venv_clean/bin/python scripts/run_echo_validation.py
experiments/build/venv_clean/bin/python scripts/check_echo_loader.py
experiments/build/venv_clean/bin/python scripts/check_echo_clean_environment.py
experiments/build/venv_clean/bin/python scripts/check_echo_training_graphs.py
python3 scripts/summarize_echo_validation.py
python3 scripts/plot_echo_validation.py
```

Feature extraction uses MPS when available, otherwise CPU. Model fitting is CPU with two threads and deterministic Torch algorithms. The recorded environment file distinguishes this contemporary execution from the historical author environment. The lock includes the full dependencies of the independently recreated environment. Publication figures additionally require Matplotlib.

Model fits are resumable; feature extraction restarts a topic if interrupted. Fitted representations are reused only when the ordered graph edge list, ordered feature matrix, model source, seed, width, and epochs match their recorded key. Final summaries explicitly count completed cells and graph realisations. Undefined one-community scores remain missing.

## Interpretation

The two graph averages answer different questions: equal weight per user versus equal weight per community. They are computed on identical fitted representations. Fitting seeds include edge splitting, initialization, and negative sampling; partition seeds are independently crossed.

Structural deltas are averaged over three fits within each graph, then summarised over five graph realisations. Their ranges are not confidence intervals. Closure swaps preserve each user's full community-neighbour counts; mixing swaps preserve degree but can change those counts. Both retain text features and use the original partition to score the result. Triangles are not treated as ground-truth echo-chamber intensity.

Regenerated user-level feature files and fitted arrays stay in ignored local directories. Release of any derived user-level artifact requires review against source terms. This repository provides aggregate results, source hashes, pinned code, and reconstruction commands; it does not claim historical authorship provenance for newly regenerated features.

## Optional original-version partition check

Create a separate environment sharing the installed analysis packages, install `networkx==3.1` there, and execute `scripts/check_echo_partition_version.py`. The script asserts version 3.1, verifies ordered inputs, and compares all 40 partitions on the primary fitted arrays. `results/partition_version_check.json` contains the completed check. This is a secondary version control, not the primary dependency lock.

The original completed run used canonical edge hashes in fit-cache keys. The packaged runner strengthens this to ordered edge hashes because edge order affects the released split. Legacy local caches may therefore be recomputed; this cache-safety change does not alter model fitting or the reported saved results.
