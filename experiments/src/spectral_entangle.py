"""Numerical check for the spectral-entanglement route to Theorem 2.

Setup.
  Ensemble A_n: uniform random d-regular graph on n vertices (one connected component).
  Ensemble B_n: two disjoint uniform random d-regular graphs, each on n/2 vertices.
  Both ensembles are d-regular on n vertices. Both converge (Benjamini-Schramm) to the
  d-regular tree locally.

Assign opinion vector o = (+1 on first n/2, -1 on last n/2), aligned with the natural
bipartition of B (so o is +1 on component 1, -1 on component 2 of B; on A, this vector is
arbitrary structural signal on a connected graph).

Checks.
  1. Same degree everywhere (obvious).
  2. Depth-T 1-WL type distributions coincide up to noise, for small T.
  3. Local clustering ~ 0 in both (locally tree-like).
  4. Spectral quantity separates: laplacian_spectrum has a second zero eigenvalue in B
     (because B is disconnected), giving qualitatively different generalized effective
     resistance behaviour. Report:
       (i) algebraic connectivity mu_2 (second-smallest Laplacian eigenvalue),
       (ii) GED under o (per-component convention),
       (iii) cut ratio |E(V_+, V_-)| / (|V_+| * |V_-|).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "ppm_check"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sample_A(n: int, d: int, rng: np.random.Generator) -> nx.Graph:
    seed = int(rng.integers(0, 2**31 - 1))
    return nx.random_regular_graph(d, n, seed=seed)


def sample_B(n: int, d: int, rng: np.random.Generator) -> nx.Graph:
    seed1 = int(rng.integers(0, 2**31 - 1))
    seed2 = int(rng.integers(0, 2**31 - 1))
    G1 = nx.random_regular_graph(d, n // 2, seed=seed1)
    G2 = nx.random_regular_graph(d, n // 2, seed=seed2)
    G = nx.disjoint_union(G1, G2)
    return G


def opinion_vector(n: int) -> np.ndarray:
    o = np.ones(n)
    o[n // 2 :] = -1
    return o


def wl_type_at_depth(G: nx.Graph, feat: np.ndarray, depth: int, v: int) -> tuple:
    if depth == 0:
        return (int(feat[v]),)
    return (
        int(feat[v]),
        tuple(sorted(wl_type_at_depth(G, feat, depth - 1, u) for u in G.neighbors(v))),
    )


def ensemble_wl_dist(graphs: list[tuple[nx.Graph, np.ndarray]], depth: int) -> dict:
    counts: Counter = Counter()
    total = 0
    for G, f in graphs:
        for v in G.nodes:
            counts[wl_type_at_depth(G, f, depth, v)] += 1
            total += 1
    return {k: c / total for k, c in counts.items()}


def tv(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def algebraic_connectivity(G: nx.Graph) -> float:
    L = nx.laplacian_matrix(G).astype(float).toarray()
    eig = np.linalg.eigvalsh(L)
    return float(eig[1])  # smallest is 0 (or two zeros if disconnected)


def ged_per_component(G: nx.Graph, o: np.ndarray) -> float:
    """GED = sqrt((o+ - o-)^T L^dagger (o+ - o-)), applied per connected component and summed.
    Only well-defined on each component; the paper's convention is per-component.
    """
    total = 0.0
    for comp in nx.connected_components(G):
        idx = sorted(comp)
        sub = G.subgraph(idx).copy()
        o_sub = o[idx]
        # Split into positive / negative parts.
        o_pos = np.clip(o_sub, 0, None)
        o_neg = np.clip(-o_sub, 0, None)
        diff = o_pos - o_neg
        L = nx.laplacian_matrix(sub, nodelist=idx).astype(float).toarray()
        L_pinv = np.linalg.pinv(L)
        val = float(diff @ L_pinv @ diff)
        total += max(val, 0.0)
    return float(np.sqrt(total))


def cut_ratio(G: nx.Graph, o: np.ndarray) -> float:
    pos = [v for v in G.nodes if o[v] > 0]
    neg = [v for v in G.nodes if o[v] < 0]
    if not pos or not neg:
        return 0.0
    cross = sum(1 for u, v in G.edges if (o[u] * o[v]) < 0)
    return cross / (len(pos) * len(neg))


def local_clustering(G: nx.Graph) -> float:
    return float(nx.average_clustering(G))


def run(n: int = 120, d: int = 4, n_samples: int = 60, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    o = opinion_vector(n)

    ens_A, ens_B = [], []
    metrics_A, metrics_B = [], []

    for _ in range(n_samples):
        GA = sample_A(n, d, rng)
        GB = sample_B(n, d, rng)
        ens_A.append((GA, o))
        ens_B.append((GB, o))
        metrics_A.append(
            {
                "clus": local_clustering(GA),
                "mu2": algebraic_connectivity(GA),
                "ged": ged_per_component(GA, o),
                "cut": cut_ratio(GA, o),
                "conn": int(nx.is_connected(GA)),
            }
        )
        metrics_B.append(
            {
                "clus": local_clustering(GB),
                "mu2": algebraic_connectivity(GB),
                "ged": ged_per_component(GB, o),
                "cut": cut_ratio(GB, o),
                "conn": int(nx.is_connected(GB)),
            }
        )

    def summarise(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    sA = summarise(metrics_A)
    sB = summarise(metrics_B)

    tv_d0 = tv(ensemble_wl_dist(ens_A, 0), ensemble_wl_dist(ens_B, 0))
    tv_d1 = tv(ensemble_wl_dist(ens_A, 1), ensemble_wl_dist(ens_B, 1))
    tv_d2 = tv(ensemble_wl_dist(ens_A, 2), ensemble_wl_dist(ens_B, 2))
    tv_d3 = tv(ensemble_wl_dist(ens_A, 3), ensemble_wl_dist(ens_B, 3))

    print(f"n={n}, d={d}, n_samples={n_samples}")
    print(f"\nEnsemble A (connected d-regular):")
    for k, v in sA.items():
        print(f"  {k:5s} = {v:.4f}")
    print(f"\nEnsemble B (two disjoint d-regular halves):")
    for k, v in sB.items():
        print(f"  {k:5s} = {v:.4f}")
    print("\nCross-graph WL type TV distance A vs B:")
    print(f"  depth 0: {tv_d0:.4f}")
    print(f"  depth 1: {tv_d1:.4f}")
    print(f"  depth 2: {tv_d2:.4f}")
    print(f"  depth 3: {tv_d3:.4f}")

    out = {
        "n": n, "d": d, "n_samples": n_samples, "seed": seed,
        "A_mean": sA, "B_mean": sB,
        "wl_tv": {"depth_0": tv_d0, "depth_1": tv_d1, "depth_2": tv_d2, "depth_3": tv_d3},
    }
    with (RESULTS_DIR / "spectral_entangle.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    run()
