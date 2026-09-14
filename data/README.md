# Data

Dataset candidates for the Reddit demonstration. Pick one before experiments start.

## Shortlist

1. **Waller & Anderson (2021) community embeddings dataset.** 10k subreddits, user-comment interaction structure. Widely cited in the polarization-measurement literature; matches the target audience's expectations. Public release via the authors' repo.
2. **r/politics + r/conservative co-commenter graphs.** A narrower two-community setup. Users are nodes; edges are co-comments on the same submission. Small enough to run many rewirings; large enough to matter. Constructed from Pushshift dumps.
3. **PushShift political-subreddit sample, 2016–2020.** A wider community set with time-slicing. Heavier lift; better if we want a temporal ablation.

## Selection criteria

- Availability under a licence that permits redistribution of derived measurements.
- Prior use in polarization-measurement papers so reviewers accept it without discussion.
- Small enough that degree- and block-preserving rewirings are tractable to run many times.

## Contents

Raw datasets are not committed. Loader in `experiments/src/load.py` fetches or points to a local path via config. `data/` holds only pointers, checksums, and a `.gitignore` entry for the raw files.
