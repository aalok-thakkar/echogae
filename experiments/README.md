# experiments/

Reproduction artifact for the ReScience C submission.

## Layout

```
validation/                 the artifact: protocol, results, environment
  protocol.md               the full study design, written before model runs
  requirements.in           direct Python dependencies
  requirements.lock         exact pinned lock used in the reported runs
  seed_parameterization.patch   source changes applied to the release
  upstream/
    manifest.json           SHA-256 and Git blob checks for 12 raw input files
    README.md               licence and pointer to upstream release
  controlled/               model source actually used (upstream + patch)
  results/
    sxsw.json, abortion.json, super_bowl.json, gun.json    per-topic outputs
    summary.json                                             design-level aggregate
    upstream_loader_check.json                               unmodified-loader check
    clean_environment_check.json                             isolated-env reference fit
    partition_version_check.json                             NetworkX 3.1 vs 3.6 check
    training_graph_diagnostics.json                          per-fit edge-split stats
    environment.json                                         host and library versions
    validation_checks.json                                   assertion log
    feature_manifest.json                                    regenerated feature hashes

src/                        scripts referenced by scripts/run_echo_validation.py
vendor/                     gitignored; populated by scripts/setup.sh
build/                      gitignored; holds the isolated venv
```

## Classification

Rp (reproduction) in the ReScience sense: same code, same raw inputs, with only the fitting-seed patch applied to the released source.

## Reproducing

From the repository root:

```bash
bash scripts/setup.sh
python3 -m venv experiments/build/venv_clean
experiments/build/venv_clean/bin/python -m pip install -r experiments/validation/requirements.lock
experiments/build/venv_clean/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"
experiments/build/venv_clean/bin/python -m nltk.downloader stopwords
experiments/build/venv_clean/bin/python scripts/run_echo_validation.py
```

Running time on an Apple M3 Max is roughly 90 minutes for the primary ten-fit SXSW table, four hours for all four topics and all ten fitting seeds, and four to six hours for the structural experiment.

## What this artifact does not promise

The current executions document what we ran; they do not reconstruct the original authors' historical environment or any unpublished feature caches. The paper's Section~5 discusses the residuals.
