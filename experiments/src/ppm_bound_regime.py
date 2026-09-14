"""Where the general ensemble bound bites: Delta c vs. TV_T on PPMs.

Theorem (general two-ensemble): MSE ≥ (1/4)·max(0, |Δc| − (b−a)·TV_T)²
The bound is non-trivial iff |Δc| > (b−a)·TV_T. This script measures
the ratio |Δc| / TV_T across a grid of PPM parameter pairs at K ∈
{2, 4, 8}, so the "regime of applicability" statement in the appendix
has empirical backing.

Setup:
  For each K, sweep p ∈ [0.01, 0.30], q ∈ [0.001, 0.10], holding
  mean degree ≈ target μ = 15.
  For each (p, q), sample N = 30 graphs of n = 500 vertices from
  PPM(n, K, p, q), record the empirical mean clustering.
  For each parameter pair (theta_1, theta_2), compute:
    Δc = |mean clustering diff|
    TV_1 = TV between depth-1 WL colour histograms
      (aggregated: empirical distribution of depth-1 colours across all
       vertices in all samples per ensemble; TV computed via joint
       canonical colour space obtained by running WL on the disjoint
       union of two representative samples).
  Report the fraction of pairs where Δc > TV_1 (bound bites).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "ppm_bound_regime"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sample_ppm(n: int, K: int, p: float, q: float, seed: int) -> nx.Graph:
    """Symmetric K-block PPM on n vertices."""
    rng = np.random.default_rng(seed)
    block = np.array([v * K // n for v in range(n)], dtype=int)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            pij = p if block[i] == block[j] else q
            if rng.random() < pij:
                G.add_edge(i, j)
    for v in range(n):
        G.nodes[v]["block"] = int(block[v])
    return G


def wl_colours_at_depth(G: nx.Graph, initial: dict, depth: int) -> Counter:
    """One-pass WL to fixed depth on a single graph, return canonical
    colour histogram. Colours across different graphs are NOT aligned
    by this function (use joint refinement for cross-graph comparison)."""
    colours = dict(initial)
    for _ in range(depth):
        proposed = {}
        for v in G.nodes():
            neigh = tuple(sorted(colours[u] for u in G.neighbors(v)))
            proposed[v] = hash((colours[v], neigh))
        remap = {c: i for i, c in enumerate(sorted(set(proposed.values())))}
        colours = {v: remap[proposed[v]] for v in G.nodes()}
    return Counter(colours.values())


def joint_ensemble_wl_hists(graphs1: list, graphs2: list, depth: int) -> tuple[Counter, Counter]:
    """Aggregate depth-T WL colour histograms across an ensemble of graphs,
    on a shared canonical colour space obtained by running WL on the disjoint
    union of all graphs together."""
    U = nx.Graph()
    offsets = []
    running = 0
    for G in graphs1 + graphs2:
        offsets.append(running)
        for v in G.nodes():
            U.add_node(v + running)
        for v in G.nodes():
            for u in G.neighbors(v):
                if u > v:
                    U.add_edge(v + running, u + running)
        running += G.number_of_nodes()
    init = {v: 0 for v in U.nodes()}
    colours = dict(init)
    for _ in range(depth):
        proposed = {}
        for v in U.nodes():
            neigh = tuple(sorted(colours[u] for u in U.neighbors(v)))
            proposed[v] = hash((colours[v], neigh))
        remap = {c: i for i, c in enumerate(sorted(set(proposed.values())))}
        colours = {v: remap[proposed[v]] for v in U.nodes()}
    h1 = Counter()
    h2 = Counter()
    for idx, G in enumerate(graphs1):
        off = offsets[idx]
        for v in G.nodes():
            h1[colours[v + off]] += 1
    for idx, G in enumerate(graphs2):
        off = offsets[len(graphs1) + idx]
        for v in G.nodes():
            h2[colours[v + off]] += 1
    return h1, h2


def tv_distance(h1: Counter, h2: Counter) -> float:
    n1 = sum(h1.values())
    n2 = sum(h2.values())
    if n1 == 0 or n2 == 0:
        return float("nan")
    keys = set(h1.keys()) | set(h2.keys())
    tv = 0.5 * sum(abs(h1.get(k, 0) / n1 - h2.get(k, 0) / n2) for k in keys)
    return float(tv)


def measure_ensemble(K: int, p: float, q: float, n: int = 500,
                      n_samples: int = 20, seed_base: int = 42) -> dict:
    graphs = [sample_ppm(n, K, p, q, seed=seed_base + i) for i in range(n_samples)]
    clus = [nx.average_clustering(G) for G in graphs]
    return {"K": K, "p": p, "q": q, "n": n, "n_samples": n_samples,
             "mean_clustering": float(np.mean(clus)),
             "std_clustering": float(np.std(clus)),
             "graphs": graphs}


def scan_K(K: int, mu_target: float = 15.0, n: int = 500,
             n_samples: int = 15, n_pairs: int = 15) -> dict:
    print(f"\n=== K = {K}: scanning parameter pairs ===", flush=True)
    # For each p, choose q so expected mean degree matches mu_target.
    # E[degree] ≈ (n/K - 1)*p + ((K-1)*n/K)*q. Solve for q given p.
    def q_of_p(p):
        return (mu_target - (n / K - 1) * p) / ((K - 1) * n / K)

    # Sample p values that give valid q ∈ (0, min(0.15, p)].
    p_vals = np.linspace(0.01, 0.15, n_pairs)
    ensembles = []
    for p in p_vals:
        q = q_of_p(p)
        if q < 1e-4 or q > p or q > 0.15:
            continue
        e = measure_ensemble(K, p, q, n=n, n_samples=n_samples)
        print(f"  p={p:.4f}, q={q:.4f}: mean deg ~ {mu_target}, "
              f"mean clus = {e['mean_clustering']:.4f}", flush=True)
        ensembles.append(e)

    # For every pair of ensembles, compute Δc and TV_1.
    pairs = []
    bite = 0
    for i in range(len(ensembles)):
        for j in range(i + 1, len(ensembles)):
            e1, e2 = ensembles[i], ensembles[j]
            dc = abs(e1["mean_clustering"] - e2["mean_clustering"])
            h1, h2 = joint_ensemble_wl_hists(e1["graphs"], e2["graphs"], depth=1)
            tv = tv_distance(h1, h2)
            pairs.append({"i": i, "j": j, "delta_c": dc, "tv_1": tv,
                           "bites": bool(dc > tv)})
            if dc > tv:
                bite += 1
            print(f"    ({i},{j}): Δc={dc:.4f}, TV_1={tv:.4f}, "
                  f"bites={dc > tv}", flush=True)

    return {"K": K, "n_ensembles": len(ensembles),
             "n_pairs": len(pairs),
             "n_bite": bite,
             "frac_bite": bite / max(len(pairs), 1),
             "pairs": pairs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n_samples", type=int, default=15)
    ap.add_argument("--n_pairs", type=int, default=8)
    args = ap.parse_args()

    out = {}
    for K in args.Ks:
        r = scan_K(K, n=args.n, n_samples=args.n_samples, n_pairs=args.n_pairs)
        # strip graph objects before serialising
        for e_pair in r["pairs"]:
            pass  # already clean
        out[str(K)] = {k: v for k, v in r.items()}

    print("\n=== Bound-regime summary ===", flush=True)
    print(f"{'K':<4} {'#pairs':<8} {'#bite':<8} {'frac bite':<10}", flush=True)
    for K, r in out.items():
        print(f"  {K:<3} {r['n_pairs']:<8} {r['n_bite']:<8} {r['frac_bite']:<10.3f}",
              flush=True)

    with (RESULTS_DIR / "regime.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {RESULTS_DIR / 'regime.json'}", flush=True)
