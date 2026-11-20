# Validation protocol, specified before the new model comparisons

Date: 2026-09-16. The earlier result files are preserved as exploratory history, not modified.

1. Pin upstream source at commit 062633017c9ea886b15d099f8370bbef447ecb5c; compare local raw files with Git blob hashes. Record source modifications explicitly.
2. Regenerate all text features using upstream preprocessing and local model snapshot 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Compare old and regenerated features. Do not overwrite historical caches.
3. Follow the notebook's graph and architecture: feature intersection, largest connected component, 20 hidden/10 output dimensions, 300 epochs, learning rate .01. Distinguish paper Eq.4 (equal community weights) from notebook (equal user weights), reporting both on identical embeddings. Record undefined single-community scores as missing, not zero.
4. Cross ten partition seeds (42--51) with ten fitting seeds (42--51) on each topic. Use fixed original graph/text per topic; fit once per seed and reuse embeddings over partitions. Control Python, NumPy, and Torch RNGs. Fitting variation includes edge splitting, initialization, and negative sampling. Report finite-design variation separately for partition, fitting, and their interaction; no population variance inference.
5. Architecture control: five fitting seeds (42--46) at 100/50 dimensions, same corrected inputs, partition seed42, both averaging rules. Domain control: seed42 on all feature-covered users with the earlier outside-LCC singleton convention, both widths and averaging rules. Label these controls rather than faithful reproductions.
6. Structural tests: five graph seeds (100--104), three fitting seeds (42--44), fixed original partition seed42. (a) Target triangle removal under per-user full-community degree preservation with exactly 100,000 proposals, accepting only triangle-decreasing swaps. (b) Degree-only rewiring with 10 accepted swaps per original non-loop edge, maximum 100 times that proposal count. Measure actual cross-community edge fraction, triangle loss, degree preservation, community-neighbour preservation, and connectivity. The first tests closure response while holding mixing fixed; the second changes mixing. Neither treats triangles as an echo-chamber ground truth.
7. Summarise structural deltas within each rewired graph over fitting seeds, then report the five graph means and their range. Shared original fits are retained explicitly; do not count the 15 fits as 15 independent graph realisations.
8. Verify efficient ECS computation against upstream's implementation. Run an independently recreated environment on a seed42 reference case with the exact saved input to check executability and repeatability. Publish hashes, environment versions, commands, and full results.

The source notebook already stores results differing from Table II for several topics. These are documentary evidence, not new executions. Original notebook randomness is not fully parameterized. The controlled repetitions explicitly extend its seed handling while retaining its model computation.

## Descriptive comparison extension

After the SXSW, Super Bowl, and abortion primary results were available, but before the gun primary fits, add a transparent descriptive summary of the crossed four-topic comparison. For each common pair of fitting/partition seed labels, calculate the two-topic-minus-two-topic mean contrast for both averages. Keep undefined combinations missing and count opposite signs. Seed labels merely index configurations; they do not pair social observations. Report finite-grid ranges/counts without p-values, population probabilities, or confidence intervals. The primary seed-42-partition ten-fit table remains unchanged.

A subsequent secondary check changes only NetworkX to 3.1 (the version specified by the release), verifies the same ordered inputs, and compares all 40 partitions and their scores on the primary fitted arrays. No model refitting or full historical environment reconstruction is implied.

## Encountered undefined-distance case

The gun full-feature-graph control encountered a zero ECS denominator and initially stopped as an explicit error. To finish the declared diagnostic without inventing a score, the runner now records user/community averages as undefined whenever an included user has a zero denominator. It records the count of affected users. The non-singleton-only diagnostic is defined only if all users it includes have defined scores. Valid-score arithmetic and all previously completed primary/structural outputs are unchanged. No failed seed is dropped or replaced.
