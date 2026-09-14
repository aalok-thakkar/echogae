"""Permutation null for EchoGAE's cross-topic separation claim.

The paper's Table 2 gives graph-level ECS values 0.714 (gun), 0.626
(abortion), 0.485 (super_bowl), 0.465 (sxsw) and reads the 0.195
polarising-vs-non-polarising mean gap as evidence for topic
polarisation. Our multi-seed reproduction gives 0.42, 0.64, 0.60,
0.64: the four means do not separate. This script gives a
per-topic permutation null against which cross-topic differences
would have to be argued.

For each topic, hold the trained EchoGAE embedding fixed at the
released seed, then compute graph-level ECS under 1000 uniform
random partitions of the vertex set into blocks of matched sizes.
Report the paper's ECS value against the null percentile.

Also computes the four across-topic mean-gap null: draw one ECS
from each topic's null 1000 times, compute the polarising-minus-
non-polarising mean gap, and report the paper's 0.195 against
this distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ECHOGAE = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
sys.path.insert(0, str(ECHOGAE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.EchoGAE import EchoGAE_algorithm  # noqa: E402
from src.echo_chamber_measure import EchoChamberMeasure  # noqa: E402
from echogae_reproduction import (  # noqa: E402
    load_topic, compute_sbert_features, louvain_communities,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "null_permutation"
RESULTS.mkdir(parents=True, exist_ok=True)

PUBLISHED = {"gun": 0.714, "abortion": 0.626,
             "super_bowl": 0.485, "sxsw": 0.465}
POLAR = {"gun", "abortion"}


def prepare(topic: str):
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
    emb = {n: sbert[G.nodes[n]["twitter_user_id"]] for n in G.nodes}
    return G, emb


def one_topic(topic: str, n_perm: int, seed: int, epochs: int) -> dict:
    G, emb = prepare(topic)
    ci = louvain_communities(G, resolution=0.05, seed=seed)
    E = EchoGAE_algorithm(G, user_embeddings=emb, epochs=epochs,
                          show_progress=False, seed=seed)
    unique, counts = np.unique(ci, return_counts=True)
    ns = counts > 1
    block_sizes = counts[ns].tolist()

    from sklearn.metrics.pairwise import euclidean_distances
    D = euclidean_distances(E).astype(np.float32)
    n = D.shape[0]

    def graph_ecs(labels: np.ndarray) -> float:
        """Vectorised community-mean of EchoChamberMeasure.metric."""
        u, counts = np.unique(labels, return_counts=True)
        keep = counts > 1
        if keep.sum() == 0:
            return float("nan")
        k = len(u)
        M = np.zeros((n, k), dtype=np.float32)
        M[np.arange(n), np.searchsorted(u, labels)] = 1.0
        sums = D @ M
        counts_f = counts.astype(np.float32)
        mean_dist = sums / counts_f[None, :]
        li = np.searchsorted(u, labels)
        a = mean_dist[np.arange(n), li]
        mean_other = mean_dist.copy()
        mean_other[np.arange(n), li] = np.inf
        b = mean_other.min(axis=1)
        m = np.maximum(a, b)
        metric = (-a + b + m) / (2 * m)
        pc = []
        for j, cid in enumerate(u):
            if not keep[j]:
                continue
            pc.append(float(metric[labels == cid].mean()))
        return float(np.mean(pc))

    observed = graph_ecs(ci)
    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    n = G.number_of_nodes()
    for k in range(n_perm):
        perm = rng.permutation(n)
        labels = np.full(n, -1, dtype=np.int64)
        offset = 0
        for cid, sz in enumerate(block_sizes):
            labels[perm[offset:offset + sz]] = cid
            offset += sz
        # singletons take unique ids so they are filtered by size > 1
        for j in range(offset, n):
            labels[perm[j]] = len(block_sizes) + j
        null[k] = graph_ecs(labels)
    print(f"  {topic}: observed = {observed:.4f}, null mean = "
          f"{null.mean():.4f} +/- {null.std(ddof=1):.4f}, "
          f"paper = {PUBLISHED[topic]}", flush=True)
    return {"topic": topic, "observed": float(observed),
            "paper": PUBLISHED[topic],
            "null_mean": float(null.mean()),
            "null_std": float(null.std(ddof=1)),
            "null_min": float(null.min()),
            "null_max": float(null.max()),
            "n_perm": int(n_perm),
            "null_samples": null.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+",
                     default=["gun", "abortion", "super_bowl", "sxsw"])
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    out = {}
    for t in args.topics:
        print(f"\n=== {t} ===", flush=True)
        out[t] = one_topic(t, args.n_perm, args.seed, args.epochs)

    # Cross-topic gap null
    rng = np.random.default_rng(args.seed)
    n_min = min(out[t]["n_perm"] for t in args.topics)
    K = 1000
    samples = np.stack([np.array(out[t]["null_samples"])[:n_min]
                        for t in args.topics])
    idx = rng.integers(0, n_min, size=(K, len(args.topics)))
    picked = np.stack([samples[i][idx[:, i]] for i in range(len(args.topics))],
                      axis=1)
    pol_cols = [i for i, t in enumerate(args.topics) if t in POLAR]
    non_cols = [i for i, t in enumerate(args.topics) if t not in POLAR]
    gaps = picked[:, pol_cols].mean(axis=1) - picked[:, non_cols].mean(axis=1)
    paper_gap = (np.mean([PUBLISHED[t] for t in args.topics if t in POLAR]) -
                 np.mean([PUBLISHED[t] for t in args.topics if t not in POLAR]))
    p = float(((np.abs(gaps) >= abs(paper_gap)).sum() + 1) / (K + 1))
    print(f"\nCross-topic gap null: paper gap = {paper_gap:.4f}, "
          f"null mean = {gaps.mean():.4f} +/- {gaps.std(ddof=1):.4f}; "
          f"two-sided p = {p:.4f}", flush=True)
    summary = {"per_topic": {t: {k: v for k, v in out[t].items()
                                  if k != "null_samples"}
                              for t in args.topics},
                "gap_null": {"paper_gap": float(paper_gap),
                              "null_mean": float(gaps.mean()),
                              "null_std": float(gaps.std(ddof=1)),
                              "two_sided_p": p, "K": K}}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {RESULTS / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
