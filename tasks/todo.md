# TIMME full audit

## Plan

- [ ] Clone TIMME (PatriciaXiao/TIMME) into `experiments/vendor/TIMME` with a shallow, no-blob clone
- [ ] Inspect what data ships with the repo (P, P_all, P50 datasets, split files, features)
- [ ] Get a baseline TIMME run working end-to-end on one dataset (P), producing their reported metrics
- [ ] Wrap their pipeline so we can set the random seed and re-run
- [ ] Multi-seed reproduction: 10 seeds × their headline metric (link-prediction AUC or ideology-classification accuracy) on P and one other dataset; compare mean±std to the published number
- [ ] Rewire audit: apply our WL-preserving rewire to the input graph, re-run TIMME, report Δ metric under rewiring
- [ ] Graph-blind baseline: strip the message-passing (linear model on the same input features / node embeddings), report the R² / accuracy the graph-blind route already achieves
- [ ] Write results into `paper/main.tex`: replace the "Limitations: TIMME deferred" paragraph with a new subsection under Section "Auditing" that reports the three response properties for TIMME
- [ ] Rebuild PDF, verify 0 warnings, verify numbers match

## Notes

- Original TIMME repo is large because it ships pre-trained embeddings and a Twitter dump. Only need the code + at least one dataset; skip pretrained weights.
- Their code targets old TensorFlow / PyTorch versions. Expect environment friction.
- If a full re-training of their supervised head is infeasible on CPU, fall back to a smaller subset (e.g. P without P50 expansion) rather than skipping the audit entirely.

# Revision pass (2026-09-02)

## Plan

- [x] Verify every TIMME number in main.tex against the results JSON and logs
- [x] Measure what the TIMME shuffle does to the binarised graph (unique edges, triangles, neighbourhood label composition)
- [x] Extract TIMME's own Table 2 values and compare to the multi-seed reproduction
- [x] Rewrite the TIMME subsection, abstract, intro, recommendations, discussion, limitations, checklist accordingly
- [x] Sweep for banned words and dashes
- [x] Rebuild PDF, 0 warnings (13 pages)

## Review

- The TIMME rewiring was described in the paper as "removes higher-order structure" and "WL-preserving". Measurement showed the opposite on the graph TIMME consumes: TIMME binarises relation counts, the shuffle disperses parallel records to new pairs, unique edges rise 70-110% and triangles rise 2-6x on the four weighted relations. What the shuffle preserves is per-account neighbourhood label composition (exactly on friend, within 0.006-0.057 elsewhere). The paper now says this, with a table (tab:timme-closure) and a Limitations sentence.
- "Within 0.1 pp of the paper's reported value" was wrong. TIMME Table 2 reports 1.000 (PureP) and 0.973 (P50) for the single-task classifier; our means are 2.4 and 4.6 pp lower, within 1 sigma.
- PureP per-relation swap counts and the P50 "at least 120,000" claim were wrong; corrected from the logs.
- Checklist answers on seeds, run counts, and variance were stale from an earlier draft.
- Small fixes: depth-2 TV range 0.69-0.79 (was 0.64), rewire caption 5% (was 8%), SXSW zero-variance wording in abstract and intro, section heading "Audit".

# Reviewer-concern pass (2026-09-02)

## Plan

- [x] Position the 1-WL bound as background and the audit as the contribution (intro paragraph "Why this happens")
- [x] Louvain: EchoGAE's release uses networkx louvain_communities(resolution=0.05, seed=42), pinned to networkx 3.1; rerun the 10-seed reproduction under networkx 3.1 in a venv and report the gap against the 3.6 run
- [x] InfoVGAE in its authors' setting: run the released pipeline on Election and Eurovision at 10 seeds (orig, binarised, label-block-preserving bipartite rewire); compare to the paper's Table 1
- [x] TIMME closure-removing rewire on the binarised relations (degree- and label-composition-preserving, triangle-destroying), PureP then P50
- [ ] Rewrite the affected paper sections; rebuild
- [ ] Author contact: draft an email for the user to send (cannot be done by me)

# 80% push (2026-09-05)

## Plan

- [x] Bootstrap CIs on assortativity across 7 datasets
- [ ] Add CIs to Table 8 (direct estimators)
- [ ] Add positive-template compact paragraph
- [ ] Foreground new theoretical contributions in intro (rewire-delta bound, RWC-non-1WL witness, ensemble MSE bound)
- [ ] Trim body to 8 pages
- [ ] Draft author-contact emails (user to send)
