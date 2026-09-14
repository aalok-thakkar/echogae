"""Statistical firming of the graph-blind reproduction.

For the two non-degenerate v2 topics (super_bowl, sxsw) this script:

1. Recomputes per-community ECS* and per-community features under the
   paper's exact pipeline (Louvain resolution 0.05, seed 42, SBERT
   features, 300 epochs).
2. Reports the Pearson r^2 of community size against ECS* per topic.
3. Runs an exact permutation test on that r^2 (n! permutations
   exhausted for n in {3, 5}).
4. Reports a pooled analysis: rank-transform ECS* within each topic,
   pool the (size, rank-ECS) pairs across the two topics, and give
   the pooled r^2 with a stratified bootstrap 95% CI (10000 draws,
   resample within topic to preserve topic size).

The pooled analysis addresses the reviewer's request for a
hierarchical bootstrap across topics; rank-transformation within
topic removes the raw-scale confound the reviewer flags.
"""

from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import pearsonr, rankdata

ECHOGAE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
sys.path.insert(0, str(ECHOGAE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.EchoGAE import EchoGAE_algorithm  # noqa: E402
from echogae_reproduction import (  # noqa: E402
    load_topic, compute_sbert_features, louvain_communities,
)
from feature_ablation import v2_features, per_community_ecs  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "graph_blind_stats"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def collect_topic(topic: str, epochs: int = 300, seed: int = 42) -> dict:
    """Return per-community ECS and size for one topic."""
    print(f"\n=== {topic} ===", flush=True)
    G = load_topic(topic)
    sbert = compute_sbert_features(topic)
    keep = [n for n in G.nodes if str(G.nodes[n]["user_id"]) in sbert]
    if len(keep) < G.number_of_nodes():
        Gs = G.subgraph(keep).copy()
        for n in Gs.nodes:
            Gs.nodes[n]["twitter_user_id"] = str(Gs.nodes[n]["user_id"])
        G = nx.convert_node_labels_to_integers(Gs, label_attribute="_orig")
    else:
        for n in G.nodes:
            G.nodes[n]["twitter_user_id"] = str(G.nodes[n]["user_id"])
    user_embeddings = {n: sbert[G.nodes[n]["twitter_user_id"]] for n in G.nodes}

    comm_ids = louvain_communities(G, resolution=0.05, seed=seed)
    n_ns = int((np.unique(comm_ids, return_counts=True)[1] > 1).sum())
    print(f"  Louvain: {n_ns} non-singleton communities", flush=True)

    print(f"  Training EchoGAE for {epochs} epochs...", flush=True)
    emb = EchoGAE_algorithm(G, user_embeddings=user_embeddings, epochs=epochs,
                            show_progress=False)
    per = per_community_ecs(emb, comm_ids)
    feats = v2_features(G, comm_ids)
    y = feats["community"].map(per).values.astype(float)
    sizes = feats["size"].values.astype(float)
    print(f"  Per-community (size, ECS): "
          + ", ".join(f"({int(s)},{v:.3f})" for s, v in zip(sizes, y)),
          flush=True)
    return {"topic": topic, "n": int(len(y)),
            "sizes": sizes.tolist(), "ecs": y.tolist()}


def exact_permutation_r2(x: np.ndarray, y: np.ndarray) -> dict:
    """Exact permutation test on Pearson r^2 for small n."""
    n = len(y)
    if n < 3:
        return {"n": int(n), "r2_obs": None, "p_two_sided": None,
                "note": "n<3"}
    r_obs, _ = pearsonr(x, y)
    r2_obs = r_obs * r_obs
    tot = 0
    hit = 0
    for perm in permutations(range(n)):
        r_perm, _ = pearsonr(x, y[list(perm)])
        tot += 1
        if r_perm * r_perm >= r2_obs - 1e-12:
            hit += 1
    p_two_sided = hit / tot
    return {"n": int(n), "r2_obs": float(r2_obs),
            "permutations_total": int(tot),
            "permutations_ge_obs": int(hit),
            "p_two_sided": float(p_two_sided)}


def pooled_bootstrap(records: list, n_boot: int = 10000, seed: int = 42) -> dict:
    """Rank-transform ECS within topic, pool across topics, bootstrap r^2."""
    rng = np.random.default_rng(seed)
    strata = []
    for rec in records:
        y = np.asarray(rec["ecs"])
        s = np.asarray(rec["sizes"])
        y_r = rankdata(y)
        s_r = rankdata(s)
        strata.append((s_r, y_r))

    def r2_of(strata_pairs):
        xs = np.concatenate([p[0] for p in strata_pairs])
        ys = np.concatenate([p[1] for p in strata_pairs])
        r, _ = pearsonr(xs, ys)
        return r * r

    r2_obs = r2_of(strata)

    boots = np.empty(n_boot)
    for b in range(n_boot):
        resampled = []
        for s_r, y_r in strata:
            idx = rng.integers(0, len(y_r), size=len(y_r))
            resampled.append((s_r[idx], y_r[idx]))
        try:
            boots[b] = r2_of(resampled)
        except Exception:
            boots[b] = np.nan

    boots = boots[~np.isnan(boots)]
    ci_low, ci_high = np.quantile(boots, [0.025, 0.975])
    return {"pooled_r2_obs": float(r2_obs),
            "bootstrap_n": int(len(boots)),
            "ci_95": [float(ci_low), float(ci_high)],
            "bootstrap_mean": float(boots.mean()),
            "bootstrap_median": float(np.median(boots))}


if __name__ == "__main__":
    records = []
    for t in ["super_bowl", "sxsw"]:
        records.append(collect_topic(t))

    perms = {}
    for rec in records:
        x = np.asarray(rec["sizes"])
        y = np.asarray(rec["ecs"])
        perms[rec["topic"]] = exact_permutation_r2(x, y)
        print(f"\n{rec['topic']}: r^2 = {perms[rec['topic']]['r2_obs']:.4f}, "
              f"exact permutation p = {perms[rec['topic']]['p_two_sided']:.4f} "
              f"(n = {perms[rec['topic']]['n']}, "
              f"{perms[rec['topic']]['permutations_total']} perms)")

    pooled = pooled_bootstrap(records)
    print(f"\nPooled (rank-transformed): r^2 = {pooled['pooled_r2_obs']:.4f}, "
          f"95% CI [{pooled['ci_95'][0]:.4f}, {pooled['ci_95'][1]:.4f}] "
          f"({pooled['bootstrap_n']} bootstraps)")

    out = {"records": records, "permutation": perms, "pooled": pooled}
    with (RESULTS_DIR / "stats.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {RESULTS_DIR / 'stats.json'}")
