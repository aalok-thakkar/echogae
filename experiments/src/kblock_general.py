"""General K-block SBM entanglement check with K blocks per feature side.

Blocks: 2K total, K with feature +1 and K with feature -1.
Each block has size m; total n = 2Km.

Edge probability matrix. Simplification: 3 params for same-feature-side structure and 1
param for cross-feature. That gives (p_in, p_sfd, p_df).
  p_in  = within-block
  p_sfd = between different blocks with same feature (all pairs equal)
  p_df  = between different-feature blocks

For a vertex v in one of the K same-feature blocks:
  +count = Binom(m-1, p_in) + (K-1) * Binom(m, p_sfd)
  -count = K * Binom(m, p_df)

Under this simple parameterisation, swapping (p_in, p_sfd) between two ensembles keeps
sum-of-Binomials mean equal but not variance. Specifically:
  Var_diff = ((m-1) - (K-1)*m) * (p_in(1-p_in) - p_sfd(1-p_sfd))
           = -(Km - K + 1 - m) * ...
  Never zero for generic parameters unless (m-1) = (K-1)m i.e. K = 2 - 1/m (impossible for
  K >= 2, m >= 1).

To match variance exactly, use different K on each side (via subblock sizing) OR use a
solver that jointly picks (K, m, p_in, p_sfd) to match mean and variance while creating a
closure gap.

We take the pragmatic approach: given a target K, solve the constrained system numerically
using scipy.optimize.

Ensemble 1: K1 same-feature blocks per side, block size m1, params (p_in^1, p_sfd^1).
Ensemble 2: K2 same-feature blocks per side, block size m2, params (p_in^2, p_sfd^2).
Both have same total same-feature vertices: K1*m1 = K2*m2 = M (say).
Match:
  (i)  mean +count:  (m1 - 1) p_in^1 + (K1 - 1) m1 p_sfd^1  =  (m2 - 1) p_in^2 + (K2 - 1) m2 p_sfd^2
  (ii) var +count:   (m1 - 1) p_in^1 (1-p_in^1) + (K1 - 1) m1 p_sfd^1 (1-p_sfd^1)  =  (m2 - 1) p_in^2 (1-p_in^2) + (K2 - 1) m2 p_sfd^2 (1-p_sfd^2)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import fsolve


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "ppm_check"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def mean_plus_count(K: int, m: int, p_in: float, p_sfd: float) -> float:
    return (m - 1) * p_in + (K - 1) * m * p_sfd


def var_plus_count(K: int, m: int, p_in: float, p_sfd: float) -> float:
    return (m - 1) * p_in * (1 - p_in) + (K - 1) * m * p_sfd * (1 - p_sfd)


def analytic_clustering(K: int, m: int, p_in: float, p_sfd: float, p_df: float) -> float:
    """Per-vertex triangles from three-block types, divided by (deg choose 2)."""
    deg = (m - 1) * p_in + (K - 1) * m * p_sfd + K * m * p_df
    # Triangle types (v in block X, neighbours u, w in same-feature-blocks or different).
    # Enumerate ordered types (u_block_type, w_block_type) with 4 categories:
    #   S = same block as v            (contributes m-1 candidates)
    #   F = same feature different block (contributes (K-1) * m candidates)
    #   D = different feature          (contributes K * m candidates)
    counts = {"S": m - 1, "F": (K - 1) * m, "D": K * m}
    edge_prob = {"S": p_in, "F": p_sfd, "D": p_df}
    # Pairs (u_type, w_type) with u_type <= w_type (unordered).
    types = ["S", "F", "D"]
    triangles = 0.0
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types):
            if j < i:
                continue
            if t1 == t2:
                n_pairs = counts[t1] * (counts[t1] - 1) / 2
            else:
                n_pairs = counts[t1] * counts[t2]
            # Edge (v, u): prob edge_prob[t1]. Edge (v, w): prob edge_prob[t2].
            # Edge (u, w): depends on both blocks. Ugh, this depends on more than just types.
            # For simplicity, use the max-entropy assumption: edge (u, w) has edge prob
            # = same-feature-prob if both u, w have same feature as v (either S or F),
            # = p_df if one has different feature (D).
            # This ignores the "same-block-as-v vs same-feature-as-v" fine distinction for
            # u,w edges, but is a decent approximation.
            if t1 in {"S", "F"} and t2 in {"S", "F"}:
                # both u, w have same feature as v; edge (u, w) same-feature.
                # If both in S: (u, w) in same block as v: prob p_in.
                # If S x F: prob p_sfd. F x F: prob p_sfd if in different blocks, p_in if same block.
                if t1 == "S" and t2 == "S":
                    p_uw = p_in
                elif t1 == "F" and t2 == "F":
                    # F x F: two vertices in K-1 blocks (other than v's block).
                    # With prob 1/(K-1), same block: p_in. Otherwise: p_sfd.
                    p_uw = p_in / (K - 1) + p_sfd * (K - 2) / (K - 1) if K > 1 else 0.0
                else:  # S x F or F x S
                    p_uw = p_sfd
            elif (t1 == "D" and t2 in {"S", "F"}) or (t2 == "D" and t1 in {"S", "F"}):
                p_uw = p_df
            else:  # both D
                # D x D: two vertices in different-feature blocks (K of them).
                # With prob 1/K same block: p_in. (K-1)/K different block: p_sfd.
                p_uw = p_in / K + p_sfd * (K - 1) / K
            triangles += n_pairs * edge_prob[t1] * edge_prob[t2] * p_uw
    if deg < 2:
        return 0.0
    return triangles / (deg * (deg - 1) / 2)


def match_moments(K1: int, m1: int, p_in1: float, p_sfd1: float, p_df: float,
                  K2: int, m2: int, p_in2_init: float = 0.05, p_sfd2_init: float = 0.1) -> tuple[float, float] | None:
    """Given ensemble-1 params, find ensemble-2's (p_in^2, p_sfd^2) matching mean and variance
    of +count distribution. Also compare closure."""
    target_mean = mean_plus_count(K1, m1, p_in1, p_sfd1)
    target_var = var_plus_count(K1, m1, p_in1, p_sfd1)

    def equations(x):
        p_in2, p_sfd2 = x
        return [
            mean_plus_count(K2, m2, p_in2, p_sfd2) - target_mean,
            var_plus_count(K2, m2, p_in2, p_sfd2) - target_var,
        ]

    best = None
    best_res = float("inf")
    # Try several initial conditions to escape local minima or unfortunate seeds.
    for init in [(0.05, 0.10), (0.10, 0.05), (0.20, 0.02), (0.02, 0.20),
                 (0.30, 0.01), (0.01, 0.30), (0.15, 0.15), (0.08, 0.12)]:
        try:
            sol, _, ier, _ = fsolve(equations, init, full_output=True)
            p_in2, p_sfd2 = sol
            if not (0 <= p_in2 <= 1 and 0 <= p_sfd2 <= 1):
                continue
            res = equations(sol)
            resnorm = abs(res[0]) + abs(res[1])
            if resnorm < best_res:
                best_res = resnorm
                best = (p_in2, p_sfd2)
        except Exception:
            continue
    if best is None or best_res > 1e-4:
        print(f"    solver could not match: best residual {best_res:.4e}, best sol {best}")
        return None
    return best


def sample_kblock(K: int, m: int, p_in: float, p_sfd: float, p_df: float,
                   rng: np.random.Generator) -> tuple[nx.Graph, np.ndarray]:
    n = 2 * K * m
    block = np.repeat(np.arange(2 * K), m)  # blocks 0..K-1 are +feature; K..2K-1 are -feature
    feat = np.where(block < K, 1, -1).astype(np.int8)
    P = np.full((n, n), p_df)
    for i in range(2 * K):
        for j in range(2 * K):
            same_feat = (i < K) == (j < K)
            if i == j:
                P_ij = p_in
            elif same_feat:
                P_ij = p_sfd
            else:
                P_ij = p_df
            P[i * m : (i + 1) * m, j * m : (j + 1) * m] = P_ij
    upper = np.triu(rng.random((n, n)) < P, k=1)
    A = upper | upper.T
    G = nx.from_numpy_array(A.astype(np.int8))
    return G, feat


def wl_type(G: nx.Graph, feat: np.ndarray, depth: int, v: int) -> tuple:
    if depth == 0:
        return (int(feat[v]),)
    return (int(feat[v]),
            tuple(sorted(wl_type(G, feat, depth - 1, u) for u in G.neighbors(v))))


def ensemble_wl_dist(graphs, depth):
    from collections import Counter
    counts: Counter = Counter()
    total = 0
    for G, f in graphs:
        for v in G.nodes:
            counts[wl_type(G, f, depth, v)] += 1
            total += 1
    return {k: c / total for k, c in counts.items()}


def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def empirical_stats(G, feat):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if m == 0:
        return {"degree": 0.0, "clustering": 0.0, "homophily": np.nan}
    return {
        "degree": 2 * m / n,
        "clustering": nx.average_clustering(G),
        "homophily": sum(1 for u, v in G.edges if feat[u] == feat[v]) / m,
    }


def run_scenario(K1: int, m1: int, p_in1: float, p_sfd1: float, p_df: float,
                 K2: int, m2: int, n_samples: int = 60, seed: int = 0):
    """Match moments between two K-block SBMs with different (K, m)."""
    sol = match_moments(K1, m1, p_in1, p_sfd1, p_df, K2, m2)
    if sol is None:
        print(f"  Match failed for K1={K1}, m1={m1}, K2={K2}, m2={m2}.")
        return None
    p_in2, p_sfd2 = sol

    print(f"\n  Ensemble 1: K={K1}, m={m1}, (p_in, p_sfd, p_df) = ({p_in1:.4f}, {p_sfd1:.4f}, {p_df:.4f})")
    print(f"  Ensemble 2: K={K2}, m={m2}, (p_in, p_sfd, p_df) = ({p_in2:.4f}, {p_sfd2:.4f}, {p_df:.4f})")
    print(f"  Analytic mean +count: E1 = {mean_plus_count(K1, m1, p_in1, p_sfd1):.4f}, "
          f"E2 = {mean_plus_count(K2, m2, p_in2, p_sfd2):.4f}")
    print(f"  Analytic var  +count: E1 = {var_plus_count(K1, m1, p_in1, p_sfd1):.4f}, "
          f"E2 = {var_plus_count(K2, m2, p_in2, p_sfd2):.4f}")
    clus1 = analytic_clustering(K1, m1, p_in1, p_sfd1, p_df)
    clus2 = analytic_clustering(K2, m2, p_in2, p_sfd2, p_df)
    print(f"  Analytic clustering:  E1 = {clus1:.4f}, E2 = {clus2:.4f}, gap = {clus1 - clus2:+.4f}")

    rng = np.random.default_rng(seed)
    ens1, ens2 = [], []
    stats1, stats2 = [], []
    for _ in range(n_samples):
        G1, f1 = sample_kblock(K1, m1, p_in1, p_sfd1, p_df, rng)
        G2, f2 = sample_kblock(K2, m2, p_in2, p_sfd2, p_df, rng)
        ens1.append((G1, f1))
        ens2.append((G2, f2))
        stats1.append(empirical_stats(G1, f1))
        stats2.append(empirical_stats(G2, f2))

    def summ(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    s1, s2 = summ(stats1), summ(stats2)
    print(f"  Empirical E1: {s1}")
    print(f"  Empirical E2: {s2}")
    print(f"  Empirical clustering gap: {s1['clustering'] - s2['clustering']:+.4f}")

    tv_d1 = tv(ensemble_wl_dist(ens1, 1), ensemble_wl_dist(ens2, 1))
    tv_d2 = tv(ensemble_wl_dist(ens1, 2), ensemble_wl_dist(ens2, 2))
    print(f"  Depth-1 WL TV: {tv_d1:.4f}")
    print(f"  Depth-2 WL TV: {tv_d2:.4f}")

    return {"K1": K1, "m1": m1, "p_in1": p_in1, "p_sfd1": p_sfd1,
            "K2": K2, "m2": m2, "p_in2": p_in2, "p_sfd2": p_sfd2,
            "clus_gap_analytic": clus1 - clus2, "clus_gap_empirical": s1['clustering'] - s2['clustering'],
            "tv_d1": tv_d1, "tv_d2": tv_d2}


if __name__ == "__main__":
    p_df = 0.05
    print("Matching mean + variance of +count between different-K ensembles.")
    print("Constraint: K*m = M (feature-side vertex count) held constant.")
    print("=" * 70)

    results = []

    # E1: K=2, m=60. M = 120. Try E2 with (K, m) satisfying K*m = 120.
    print("\n### M = 120 ###")
    for K2, m2 in [(3, 40), (4, 30), (5, 24), (6, 20), (8, 15), (10, 12)]:
        assert K2 * m2 == 120
        print(f"\n--- E1: K=2, m=60. E2: K={K2}, m={m2} ---")
        r = run_scenario(K1=2, m1=60, p_in1=0.18, p_sfd1=0.02, p_df=p_df,
                         K2=K2, m2=m2, n_samples=60)
        if r is not None:
            results.append(r)

    # E1: K=2, m=30. M = 60. Try smaller.
    print("\n\n### M = 60 ###")
    for K2, m2 in [(3, 20), (4, 15), (5, 12), (6, 10)]:
        assert K2 * m2 == 60
        print(f"\n--- E1: K=2, m=30. E2: K={K2}, m={m2} ---")
        r = run_scenario(K1=2, m1=30, p_in1=0.18, p_sfd1=0.02, p_df=p_df,
                         K2=K2, m2=m2, n_samples=100)
        if r is not None:
            results.append(r)

    with (RESULTS_DIR / "kblock_general.json").open("w") as fh:
        json.dump(results, fh, indent=2)
