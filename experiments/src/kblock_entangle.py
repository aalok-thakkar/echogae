"""K-block SBM entanglement check.

Setup. Four blocks {A+, B+, A-, B-} of equal size m = n/4. Features are +1 on {A+, B+}
and -1 on {A-, B-}. Symmetric edge probability matrix parameterised by four scalars:
  p_in  = within-block edge probability                        (A+/A+, B+/B+, A-/A-, B-/B-)
  p_sfd = same-feature different-block probability             (A+/B+, A-/B-)
  p_dfs = diff-feature 'symmetric' probability                 (A+/A-, B+/B-)
  p_dfd = diff-feature 'diagonal' probability                  (A+/B-, B+/A-)

Given these, the expected neighbour-feature count of a +1-vertex v in A+ is:
  E[same-feature neighbours] = m * p_in + m * p_sfd
  E[diff-feature neighbours] = m * p_dfs + m * p_dfd
Two parameter settings with matching values of (p_in + p_sfd) and (p_dfs + p_dfd) have
matching first-moment neighbour-feature distributions.

In the Poisson (sparse) regime, matching the two sums also matches the joint distribution
of (+count, -count) at depth 1 up to the Poisson approximation.

Clustering: triangles are counted per vertex, decomposed by the block types of the three
vertices. Different (p_in, p_sfd, p_dfs, p_dfd) at fixed sums can produce different
per-vertex triangle counts.

Search: fix targets alpha := p_in + p_sfd and beta := p_dfs + p_dfd. Vary p_in freely,
also vary the (p_dfs, p_dfd) split. Find two parameter settings with the same (alpha, beta)
and substantially different expected clustering. Verify by simulation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "ppm_check"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Block indices for readability.
A_POS, B_POS, A_NEG, B_NEG = 0, 1, 2, 3
BLOCKS = [A_POS, B_POS, A_NEG, B_NEG]
POS_BLOCKS = {A_POS, B_POS}
NEG_BLOCKS = {A_NEG, B_NEG}
FEATURE = {A_POS: 1, B_POS: 1, A_NEG: -1, B_NEG: -1}


@dataclass
class Params:
    p_in: float   # within-block probability
    p_sfd: float  # same-feature different-block probability
    p_dfs: float  # diff-feature symmetric ({A+, A-} or {B+, B-})
    p_dfd: float  # diff-feature diagonal ({A+, B-} or {B+, A-})

    def matrix(self) -> np.ndarray:
        p = np.zeros((4, 4))
        for i in BLOCKS:
            for j in BLOCKS:
                if i == j:
                    p[i, j] = self.p_in
                elif FEATURE[i] == FEATURE[j]:
                    p[i, j] = self.p_sfd
                elif (i, j) in {(A_POS, A_NEG), (A_NEG, A_POS), (B_POS, B_NEG), (B_NEG, B_POS)}:
                    p[i, j] = self.p_dfs
                else:
                    p[i, j] = self.p_dfd
        return p

    def alpha(self) -> float:
        return self.p_in + self.p_sfd

    def beta(self) -> float:
        return self.p_dfs + self.p_dfd


def analytic_neighbour_counts(params: Params, m: int) -> dict:
    """Expected same- and diff-feature neighbour counts of a vertex (say in A+)."""
    same = m * params.p_in + m * params.p_sfd
    diff = m * params.p_dfs + m * params.p_dfd
    return {"same_feature": same, "diff_feature": diff, "degree": same + diff}


def analytic_triangles_per_vertex(params: Params, m: int) -> float:
    """Expected triangles containing a fixed vertex v in A+. Sums over unordered
    neighbour pairs (u, w) and edge probabilities.
    """
    P = params.matrix()
    total = 0.0
    for u_block in BLOCKS:
        n_u = m - 1 if u_block == A_POS else m
        for w_block in BLOCKS:
            if u_block == w_block:
                n_w = n_u - 1  # exclude u
                pairs = n_u * n_w / 2
            else:
                n_w = m - 1 if w_block == A_POS else m
                pairs = n_u * n_w / 2  # unordered pair (u, w) with u_block != w_block
            total += pairs * P[A_POS, u_block] * P[A_POS, w_block] * P[u_block, w_block]
    return total


def analytic_clustering(params: Params, m: int) -> float:
    """Expected local clustering coefficient of a vertex, using the ratio-of-expectations
    approximation E[T_v] / binom(E[deg_v], 2). For a rigorous version use E[T_v / binom(deg, 2)]
    conditional on deg; the ratio-of-expectations is accurate in the Poisson-like regime."""
    T = analytic_triangles_per_vertex(params, m)
    counts = analytic_neighbour_counts(params, m)
    d = counts["degree"]
    if d < 2:
        return 0.0
    return T / (d * (d - 1) / 2)


def sample_sbm(params: Params, m: int, rng: np.random.Generator) -> tuple[nx.Graph, np.ndarray]:
    n = 4 * m
    block = np.zeros(n, dtype=np.int8)
    for k in BLOCKS:
        block[k * m : (k + 1) * m] = k
    feat = np.array([FEATURE[b] for b in block], dtype=np.int8)
    P = params.matrix()
    prob_matrix = P[block[:, None], block[None, :]]
    upper = np.triu(rng.random((n, n)) < prob_matrix, k=1)
    A = upper | upper.T
    G = nx.from_numpy_array(A.astype(np.int8))
    return G, feat


def empirical_stats(G: nx.Graph, feat: np.ndarray) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if m == 0:
        return {"degree": 0.0, "clustering": 0.0, "homophily": np.nan,
                "mean_same_feat_neigh": 0.0, "mean_diff_feat_neigh": 0.0}
    degree = 2 * m / n
    clustering = nx.average_clustering(G)
    same = sum(1 for u, v in G.edges if feat[u] == feat[v])
    homophily = same / m
    same_counts = []
    diff_counts = []
    for v in G.nodes:
        f = feat[v]
        s = sum(1 for u in G.neighbors(v) if feat[u] == f)
        d = G.degree[v]
        same_counts.append(s)
        diff_counts.append(d - s)
    return {
        "degree": degree,
        "clustering": clustering,
        "homophily": homophily,
        "mean_same_feat_neigh": float(np.mean(same_counts)),
        "mean_diff_feat_neigh": float(np.mean(diff_counts)),
    }


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


def run(m: int = 40, n_samples: int = 100, seed: int = 0,
        params1: Params | None = None, params2: Params | None = None,
        scale_p: bool = False) -> dict:
    """Search for parameter pairs (params_1, params_2) matched on (alpha, beta) with a
    substantial clustering gap. Then simulate and verify.

    If scale_p is True, the base parameters are treated as pilot values at m = 40 and
    rescaled to m by multiplying by 40 / m, so that expected degrees stay constant as m
    grows. This tests the Poisson (sparse-limit) regime.
    """
    if params1 is None:
        params1 = Params(p_in=0.18, p_sfd=0.02, p_dfs=0.05, p_dfd=0.05)
    if params2 is None:
        params2 = Params(p_in=0.05, p_sfd=0.15, p_dfs=0.05, p_dfd=0.05)
    if scale_p:
        scale = 40.0 / m
        params1 = Params(*[getattr(params1, k) * scale for k in ["p_in", "p_sfd", "p_dfs", "p_dfd"]])
        params2 = Params(*[getattr(params2, k) * scale for k in ["p_in", "p_sfd", "p_dfs", "p_dfd"]])

    print(f"m={m}, n={4*m}, n_samples={n_samples}")
    print(f"\nEnsemble 1: {params1}")
    print(f"  alpha (same-feature edge sum) = {params1.alpha():.3f}")
    print(f"  beta  (diff-feature edge sum) = {params1.beta():.3f}")
    print(f"  analytic degree              = {analytic_neighbour_counts(params1, m)['degree']:.3f}")
    print(f"  analytic clustering          = {analytic_clustering(params1, m):.4f}")

    print(f"\nEnsemble 2: {params2}")
    print(f"  alpha (same-feature edge sum) = {params2.alpha():.3f}")
    print(f"  beta  (diff-feature edge sum) = {params2.beta():.3f}")
    print(f"  analytic degree              = {analytic_neighbour_counts(params2, m)['degree']:.3f}")
    print(f"  analytic clustering          = {analytic_clustering(params2, m):.4f}")

    rng = np.random.default_rng(seed)
    ens1, ens2 = [], []
    stats1, stats2 = [], []
    for _ in range(n_samples):
        G1, f1 = sample_sbm(params1, m, rng)
        G2, f2 = sample_sbm(params2, m, rng)
        ens1.append((G1, f1))
        ens2.append((G2, f2))
        stats1.append(empirical_stats(G1, f1))
        stats2.append(empirical_stats(G2, f2))

    def summarise(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    s1 = summarise(stats1)
    s2 = summarise(stats2)
    print("\nEmpirical:")
    print("  Ensemble 1:", s1)
    print("  Ensemble 2:", s2)
    print(f"  clustering gap: {s1['clustering'] - s2['clustering']:+.4f}")
    print(f"  homophily gap:  {s1['homophily'] - s2['homophily']:+.4f}")
    print(f"  degree gap:     {s1['degree'] - s2['degree']:+.4f}")
    print(f"  mean_same_neigh_gap: {s1['mean_same_feat_neigh'] - s2['mean_same_feat_neigh']:+.4f}")
    print(f"  mean_diff_neigh_gap: {s1['mean_diff_feat_neigh'] - s2['mean_diff_feat_neigh']:+.4f}")

    tv_d0 = tv(ensemble_wl_dist(ens1, 0), ensemble_wl_dist(ens2, 0))
    tv_d1 = tv(ensemble_wl_dist(ens1, 1), ensemble_wl_dist(ens2, 1))
    tv_d2 = tv(ensemble_wl_dist(ens1, 2), ensemble_wl_dist(ens2, 2))
    print("\nCross-graph WL type TV:")
    print(f"  depth 0: {tv_d0:.4f}")
    print(f"  depth 1: {tv_d1:.4f}")
    print(f"  depth 2: {tv_d2:.4f}")

    out = {
        "m": m, "n": 4 * m, "n_samples": n_samples,
        "params_1": params1.__dict__, "params_2": params2.__dict__,
        "analytic_1": {"degree": analytic_neighbour_counts(params1, m)['degree'],
                       "clustering": analytic_clustering(params1, m)},
        "analytic_2": {"degree": analytic_neighbour_counts(params2, m)['degree'],
                       "clustering": analytic_clustering(params2, m)},
        "empirical_1": s1, "empirical_2": s2,
        "wl_tv": {"depth_0": tv_d0, "depth_1": tv_d1, "depth_2": tv_d2},
    }
    with (RESULTS_DIR / "kblock_entangle.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    print("=" * 60)
    print("Baseline (non-swap): matched sums, different p_in, p_sfd values.")
    print("=" * 60)
    run(m=40, n_samples=100)

    print("\n" + "=" * 60)
    print("SWAP construction: same multiset {p_in, p_sfd}, roles swapped.")
    print("Ensemble 1 has dense within-block, sparse same-feature cross-block.")
    print("Ensemble 2 has sparse within-block, dense same-feature cross-block.")
    print("=" * 60)
    params_a = Params(p_in=0.18, p_sfd=0.02, p_dfs=0.05, p_dfd=0.05)
    params_b = Params(p_in=0.02, p_sfd=0.18, p_dfs=0.05, p_dfd=0.05)
    for m in [40, 80, 160]:
        print(f"\n--- m = {m} (n = {4*m}) ---")
        run(m=m, n_samples=100 if m <= 80 else 60, params1=params_a, params2=params_b)
