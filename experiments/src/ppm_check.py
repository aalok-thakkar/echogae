"""Numerical sanity check for the entanglement construction in theory/entanglement.tex.

Goal.
  Solve for two parameter triples theta_1 = (p_1, q_1, sigma_1), theta_2 = (p_2, q_2, sigma_2)
  under the feature-perturbed planted-partition model on n vertices with two equal blocks such
  that
    - expected degrees agree,
    - expected edge homophily agrees,
    - expected clustering coefficients differ by a target amount c >= 0.1.
  Then sample N graphs from each and verify that:
    - empirical clustering separates by roughly c,
    - empirical homophily and mean degree agree,
    - depth-1 1-WL colour histograms agree (in distribution),
    - depth-2 1-WL colour histograms agree (this is the harder check; the current
      construction matches only first-order stats, so a mismatch at depth 2 is expected
      and diagnostic for how to strengthen the theorem).

Model recap.
  For (p, q, sigma) on n vertices split evenly:
    d(theta)   = ((n-1)/2) * (p + q)           (expected degree)
    h(theta)   = (p*((1-sigma)^2 + sigma^2) + q*2*sigma*(1-sigma)) / (p + q)
    c(theta)   = closed-form leading term in n from the paper (uses only p, q)
  Reparameterise s = p + q, r = p - q. Then
    d(theta)   = ((n-1)/2) * s
    h(theta)   = 1/2 + (r / (2*s)) * (1 - 2*sigma)^2
    c(theta)   = f(s, r) independent of sigma.
  So fixing s and h leaves r * (1 - 2*sigma)^2 fixed. Choose (r_1, sigma_1), then any
  (r_2, sigma_2) with r_2 * (1 - 2*sigma_2)^2 = r_1 * (1 - 2*sigma_1)^2 preserves h.
  Vary r to obtain a closure gap. sigma is solved from r.

Run.
  python3 ppm_check.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import networkx as nx
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "ppm_check"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Theta:
    p: float
    q: float
    sigma: float

    @property
    def s(self) -> float:
        return self.p + self.q

    @property
    def r(self) -> float:
        return self.p - self.q

    def expected_degree(self, n: int) -> float:
        return (n - 1) / 2 * (self.p + self.q)

    def expected_homophily(self) -> float:
        rho = (1 - self.sigma) ** 2 + self.sigma**2
        mu = 2 * self.sigma * (1 - self.sigma)
        return (self.p * rho + self.q * mu) / (self.p + self.q)


def solve_matched_pair(s: float, r1: float, sigma1: float, r2: float) -> tuple[Theta, Theta]:
    """Given s, choose (r1, sigma1), and a target r2 with |r2| <= s. Solve for sigma2
    such that r2 * (1 - 2*sigma2)^2 == r1 * (1 - 2*sigma1)^2. Both s and h are then
    preserved between the two parameter triples.
    """
    target = r1 * (1 - 2 * sigma1) ** 2
    # r2 * (1 - 2*sigma2)^2 = target  =>  (1 - 2*sigma2)^2 = target / r2.
    val = target / r2
    if val < 0 or val > 1:
        raise ValueError(
            f"Cannot match: target/r2 = {val:.4f} outside [0, 1]. Choose different r2."
        )
    sigma2 = 0.5 * (1 - np.sqrt(val))  # sigma2 in [0, 0.5]
    t1 = Theta(p=(s + r1) / 2, q=(s - r1) / 2, sigma=sigma1)
    t2 = Theta(p=(s + r2) / 2, q=(s - r2) / 2, sigma=sigma2)
    return t1, t2


def sample_ppm(theta: Theta, n: int, rng: np.random.Generator) -> tuple[nx.Graph, np.ndarray]:
    """Sample one graph and feature vector from feature-perturbed PPM."""
    half = n // 2
    block = np.zeros(n, dtype=np.int8)
    block[half:] = 1  # 0 = B_+, 1 = B_-
    # Adjacency via block-conditional Bernoulli.
    prob = np.where(block[:, None] == block[None, :], theta.p, theta.q)
    upper = np.triu(rng.random((n, n)) < prob, k=1)
    A = upper | upper.T
    G = nx.from_numpy_array(A.astype(np.int8))
    # Features: +1 in B_+, -1 in B_-, flipped with prob sigma.
    true_feat = np.where(block == 0, 1, -1).astype(np.int8)
    flips = rng.random(n) < theta.sigma
    feat = true_feat * np.where(flips, -1, 1).astype(np.int8)
    return G, feat


def empirical_stats(G: nx.Graph, feat: np.ndarray) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if m == 0:
        return {"degree": 0.0, "clustering": 0.0, "homophily": np.nan}
    degree = 2 * m / n
    clustering = nx.average_clustering(G)
    same = sum(1 for u, v in G.edges if feat[u] == feat[v])
    homophily = same / m
    return {"degree": degree, "clustering": clustering, "homophily": homophily}


def wl_type_at_depth(G: nx.Graph, feat: np.ndarray, depth: int, v: int) -> tuple:
    """Cross-graph canonical WL type of vertex v at the given depth. Recursion:
      depth 0: feature.
      depth d: (feature, sorted tuple of (depth-(d-1) type of each neighbour)).
    Uses tuple-of-tuples so equality is structural and comparable across graphs.
    """
    if depth == 0:
        return (int(feat[v]),)
    return (
        int(feat[v]),
        tuple(sorted(wl_type_at_depth(G, feat, depth - 1, u) for u in G.neighbors(v))),
    )


def ensemble_type_distribution(
    graphs: list[tuple[nx.Graph, np.ndarray]], depth: int
) -> dict[tuple, float]:
    """Aggregate over all (graph, vertex) pairs; return normalised type histogram."""
    from collections import Counter

    counts: Counter = Counter()
    total = 0
    for G, f in graphs:
        for v in G.nodes:
            counts[wl_type_at_depth(G, f, depth, v)] += 1
            total += 1
    return {k: c / total for k, c in counts.items()}


def tv_distance(pa: dict[tuple, float], pb: dict[tuple, float]) -> float:
    keys = set(pa) | set(pb)
    return 0.5 * sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys)


def compare_ensembles_at_depth(
    graphs_a: list, graphs_b: list, depth: int
) -> dict:
    pa = ensemble_type_distribution(graphs_a, depth)
    pb = ensemble_type_distribution(graphs_b, depth)
    keys = set(pa) | set(pb)
    return {
        "depth": depth,
        "num_types_a": len(pa),
        "num_types_b": len(pb),
        "num_types_shared": sum(1 for k in keys if k in pa and k in pb),
        "tv_distance": tv_distance(pa, pb),
    }


def run(
    n: int = 200,
    n_samples: int = 200,
    s: float = 0.20,
    r1: float = 0.02,
    r2: float = 0.18,
    sigma1: float = 0.05,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    theta1, theta2 = solve_matched_pair(s=s, r1=r1, sigma1=sigma1, r2=r2)

    print(f"theta_1: p={theta1.p:.4f} q={theta1.q:.4f} sigma={theta1.sigma:.4f}")
    print(f"theta_2: p={theta2.p:.4f} q={theta2.q:.4f} sigma={theta2.sigma:.4f}")
    print(
        f"analytic homophily: {theta1.expected_homophily():.4f} vs {theta2.expected_homophily():.4f}"
    )
    print(
        f"analytic expected degree (n={n}): "
        f"{theta1.expected_degree(n):.2f} vs {theta2.expected_degree(n):.2f}"
    )

    stats1, stats2 = [], []
    ensemble1, ensemble2 = [], []

    for _ in range(n_samples):
        G1, f1 = sample_ppm(theta1, n, rng)
        G2, f2 = sample_ppm(theta2, n, rng)
        stats1.append(empirical_stats(G1, f1))
        stats2.append(empirical_stats(G2, f2))
        ensemble1.append((G1, f1))
        ensemble2.append((G2, f2))

    def summarise(rows: list[dict]) -> dict:
        return {
            k: float(np.mean([r[k] for r in rows]))
            for k in ("degree", "clustering", "homophily")
        }

    s1 = summarise(stats1)
    s2 = summarise(stats2)
    cmp_d0 = compare_ensembles_at_depth(ensemble1, ensemble2, 0)
    cmp_d1 = compare_ensembles_at_depth(ensemble1, ensemble2, 1)
    cmp_d2 = compare_ensembles_at_depth(ensemble1, ensemble2, 2)

    print("\nempirical means over samples:")
    print(f"  theta_1:  {s1}")
    print(f"  theta_2:  {s2}")
    print(f"  closure gap:  {s1['clustering'] - s2['clustering']:+.4f}")
    print(f"  homophily gap: {s1['homophily'] - s2['homophily']:+.4f}")
    print(f"  degree gap:   {s1['degree'] - s2['degree']:+.4f}")
    print("\ncross-graph 1-WL type distributions (TV distance between ensembles):")
    print(f"  depth 0 (feature only):                       tv={cmp_d0['tv_distance']:.4f}")
    print(
        f"  depth 1 (feature + neighbour features):      tv={cmp_d1['tv_distance']:.4f}"
        f"   shared_types={cmp_d1['num_types_shared']}/{cmp_d1['num_types_a']}(a)/{cmp_d1['num_types_b']}(b)"
    )
    print(
        f"  depth 2 (feature + 2-hop features):          tv={cmp_d2['tv_distance']:.4f}"
        f"   shared_types={cmp_d2['num_types_shared']}/{cmp_d2['num_types_a']}(a)/{cmp_d2['num_types_b']}(b)"
    )

    out = {
        "config": {
            "n": n, "n_samples": n_samples, "s": s, "r1": r1, "r2": r2, "sigma1": sigma1, "seed": seed,
        },
        "theta_1": asdict(theta1),
        "theta_2": asdict(theta2),
        "analytic": {
            "hom_1": theta1.expected_homophily(),
            "hom_2": theta2.expected_homophily(),
            "deg_1": theta1.expected_degree(n),
            "deg_2": theta2.expected_degree(n),
        },
        "empirical_mean_1": s1,
        "empirical_mean_2": s2,
        "wl_compare_depth_0": cmp_d0,
        "wl_compare_depth_1": cmp_d1,
        "wl_compare_depth_2": cmp_d2,
    }
    with (RESULTS_DIR / "run.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    run()
