"""Rewire-instance ensemble for the Reddit audit.

The paper's Reddit rewire delta (-0.041 +/- 0.040 across 10 EchoGAE
training seeds) is computed on a single rewired graph. That
conflates rewiring-instance variance with training variance. This
script draws R rewired graphs (independent rewire seeds), trains
EchoGAE at S seeds on each, and decomposes Var(delta) into a
rewire component and a training-seed component.

Report:
  Var_rewire  variance across rewire seeds of the delta means (over
              training seeds)
  Var_seed    within-rewire variance across training seeds
  Total       Var of all R * S deltas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" /
                        "echo-chamber-score"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.EchoGAE import EchoGAE_algorithm  # noqa: E402
from src.echo_chamber_measure import EchoChamberMeasure  # noqa: E402
from echogae_reproduction import louvain_communities  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "reddit_ensemble"
RESULTS.mkdir(parents=True, exist_ok=True)

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "reddit"


def load_reddit():
    import pandas as pd
    df = pd.read_csv(VENDOR / "hyperlinks-body.tsv", sep="\t")
    edges = df[df["SOURCE_SUBREDDIT"] != df["TARGET_SUBREDDIT"]]
    from collections import Counter
    c = Counter(tuple(sorted((a, b))) for a, b in
                zip(edges["SOURCE_SUBREDDIT"], edges["TARGET_SUBREDDIT"]))
    G = nx.Graph()
    for (a, b), n in c.items():
        if n >= 3:
            G.add_edge(a.lower(), b.lower())
    sc = pd.read_csv(VENDOR / "wa" / "data" / "scores.csv")
    m = {str(n).lower(): float(v) for n, v in
         zip(sc["community"], sc["partisan"])}
    keep = [v for v in G.nodes if v in m]
    G = G.subgraph(keep).copy()
    cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(cc).copy()
    G = nx.convert_node_labels_to_integers(G, label_attribute="_orig")
    return G, m


def rewire(G: nx.Graph, block: dict, rewire_seed: int,
            n_swaps: int = 10000) -> nx.Graph:
    """Block-preserving double edge swap."""
    rng = np.random.default_rng(rewire_seed)
    G = G.copy()
    edges = list(G.edges)
    m = len(edges)
    n_target = min(n_swaps, m // 2)
    accepted = 0
    tries = 0
    while accepted < n_target and tries < 20 * n_target:
        tries += 1
        i, j = int(rng.integers(m)), int(rng.integers(m))
        if i == j: continue
        a, b = edges[i]; c, d = edges[j]
        if len({a, b, c, d}) < 4: continue
        if (block.get(a) != block.get(c) or
             block.get(b) != block.get(d)):
            continue
        if G.has_edge(a, d) or G.has_edge(c, b):
            continue
        G.remove_edge(a, b); G.remove_edge(c, d)
        G.add_edge(a, d); G.add_edge(c, b)
        edges[i] = (a, d); edges[j] = (c, b)
        accepted += 1
    return G


def graph_ecs(G: nx.Graph, comm_ids: np.ndarray, features: np.ndarray,
               seed: int, epochs: int) -> float:
    E = EchoGAE_algorithm(G, user_embeddings={v: features[v] for v in G.nodes},
                           epochs=epochs, show_progress=False, seed=seed)
    ec = EchoChamberMeasure(E, comm_ids, metric="euclidean")
    u, c = np.unique(comm_ids, return_counts=True)
    pc = []
    for cid, sz in zip(u, c):
        if sz <= 1: continue
        idx = np.where(comm_ids == cid)[0]
        pc.append(float(np.mean([ec.metric(int(i)) for i in idx])))
    return float(np.mean(pc)) if pc else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rewires", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    G, score_map = load_reddit()
    n = G.number_of_nodes()
    block = {v: (0 if score_map[G.nodes[v]["_orig"]] < 0 else 1)
             for v in G.nodes}
    # 2-D features: partisan score plus a bias.
    feats = np.zeros((n, 2), dtype=np.float32)
    for v in G.nodes:
        feats[v, 0] = float(score_map[G.nodes[v]["_orig"]])
        feats[v, 1] = 1.0
    print(f"Reddit LCC: n={n}, m={G.number_of_edges()}", flush=True)
    comm_ids = louvain_communities(G, resolution=1.0, seed=42)

    per_seed_orig = []
    for s in range(42, 42 + args.n_seeds):
        v = graph_ecs(G, comm_ids, feats, seed=s, epochs=args.epochs)
        per_seed_orig.append(v)
        print(f"  orig s={s}: ECS={v:.4f}", flush=True)

    deltas_by_rewire = []
    per_rewire_details = []
    for r in range(args.n_rewires):
        print(f"\n=== rewire {r} ===", flush=True)
        Gr = rewire(G, block, rewire_seed=100 + r)
        seed_ecs = []
        for s in range(42, 42 + args.n_seeds):
            v = graph_ecs(Gr, comm_ids, feats, seed=s, epochs=args.epochs)
            seed_ecs.append(v)
            print(f"  rw s={s}: ECS={v:.4f}", flush=True)
        deltas = [rw - o for rw, o in zip(seed_ecs, per_seed_orig)]
        deltas_by_rewire.append(deltas)
        per_rewire_details.append({"rewire_seed": 100 + r,
                                    "orig_ecs": per_seed_orig,
                                    "rewired_ecs": seed_ecs,
                                    "deltas": deltas})

    A = np.array(deltas_by_rewire)  # (R, S)
    rewire_means = A.mean(axis=1)
    var_between_rewires = float(rewire_means.var(ddof=1))
    var_within_rewire = float(np.mean(A.var(axis=1, ddof=1)))
    var_total = float(A.flatten().var(ddof=1))
    summary = {
        "n_rewires": args.n_rewires, "n_seeds": args.n_seeds,
        "delta_mean": float(A.mean()),
        "delta_std_total": float(A.std(ddof=1)),
        "var_total": var_total,
        "var_between_rewires": var_between_rewires,
        "var_within_rewire_mean": var_within_rewire,
        "sign_by_rewire": [int(np.sign(m)) for m in rewire_means],
        "per_rewire_mean_delta": [float(x) for x in rewire_means],
    }
    print(f"\n=== summary ===\n{json.dumps(summary, indent=2)}", flush=True)
    out = {"summary": summary, "per_rewire": per_rewire_details}
    (RESULTS / "ensemble.json").write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {RESULTS / 'ensemble.json'}", flush=True)


if __name__ == "__main__":
    main()
