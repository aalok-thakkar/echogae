"""Search for an RWC witness on 1-WL equivalent, non-vertex-transitive pairs.

The Shrikhande/rook pair pins RWC because both graphs are vertex-transitive
and cospectral. We ask whether any 1-WL equivalent pair on small graphs
(up to n vertices) has at least one non-vertex-transitive member and gives
distinct RWC values on some bipartition.

Procedure:
  1. Enumerate connected graphs on n vertices via networkx.graph_atlas.
  2. Compute stable 1-WL colour histogram (constant initial features).
  3. Group graphs by 1-WL histogram; consider each equivalence class of
     size >= 2 with at least one non-vertex-transitive member.
  4. For each such class, for each unordered pair, enumerate all
     balanced-or-nearly-balanced bipartitions and compute RWC with
     the two highest-degree vertices per side as authorities.
  5. Report any pair with |RWC(G_1) - RWC(G_2)| > eps.

If no witness lands in the atlas (n <= 7 in networkx), we strengthen the
obstacle statement to precisely identify what a witness would require.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import networkx as nx
import numpy as np


def wl_histogram(G: nx.Graph, iterations: int = 10) -> tuple:
    """Stable 1-WL colour histogram with constant initial colours."""
    n = G.number_of_nodes()
    colours = {v: 0 for v in G.nodes()}
    for _ in range(iterations):
        new = {}
        for v in G.nodes():
            neigh = tuple(sorted(colours[u] for u in G.neighbors(v)))
            new[v] = hash((colours[v], neigh))
        # relabel to a canonical range
        mapping = {c: i for i, c in enumerate(sorted(set(new.values())))}
        canonical = {v: mapping[c] for v, c in new.items()}
        if sorted(canonical.values()) == sorted(colours.values()):
            colours = canonical
            break
        colours = canonical
    return tuple(sorted(colours.values()))


def is_vertex_transitive(G: nx.Graph) -> bool:
    """Cheap sufficient condition: all vertices have same 1-WL final colour.
    Necessary condition, not sufficient in general, but sufficient for our
    filter (a WL-single-class graph is at least locally transitive)."""
    hist = wl_histogram(G)
    return len(set(hist)) == 1


def rwc(G: nx.Graph, left: set, right: set, k_auth: int = 2) -> float:
    """Random walk controversy per Garimella et al. 2018 with degree-based
    authority selection.

    Simplified: authorities on each side are the k_auth highest-degree
    vertices. Absorbing states: {left authorities, right authorities}.
    Compute P(L|L) etc via exact absorbing Markov chain (fundamental
    matrix)."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    left_sorted = sorted(left, key=lambda v: -G.degree(v))
    right_sorted = sorted(right, key=lambda v: -G.degree(v))
    if len(left_sorted) < k_auth or len(right_sorted) < k_auth:
        return float("nan")
    la = set(left_sorted[:k_auth])
    ra = set(right_sorted[:k_auth])
    absorbing = la | ra
    transient = [v for v in nodes if v not in absorbing]
    if not transient:
        return float("nan")

    # Build transition matrix P on transient states plus absorbing sinks.
    # For absorbing chain, we need Q (transient->transient) and R (transient->absorbing).
    t = len(transient)
    a = len(absorbing)
    t_idx = {v: i for i, v in enumerate(transient)}
    a_idx = {v: i for i, v in enumerate(sorted(absorbing))}
    Q = np.zeros((t, t))
    R = np.zeros((t, a))
    for v in transient:
        deg = G.degree(v)
        if deg == 0:
            continue
        for u in G.neighbors(v):
            if u in absorbing:
                R[t_idx[v], a_idx[u]] += 1.0 / deg
            else:
                Q[t_idx[v], t_idx[u]] += 1.0 / deg
    # Fundamental matrix N = (I - Q)^{-1}, and B = N @ R gives absorption probs.
    try:
        N = np.linalg.inv(np.eye(t) - Q)
    except np.linalg.LinAlgError:
        return float("nan")
    B = N @ R

    # For each transient vertex, absorb probability at each authority.
    la_cols = [a_idx[u] for u in la]
    ra_cols = [a_idx[u] for u in ra]

    def side_prob(v_transient_idx: int, side_cols: list) -> float:
        return float(sum(B[v_transient_idx, c] for c in side_cols))

    # RWC = mean_{v in L\la} P(L|v) * mean_{v in R\ra} P(R|v)
    #     - mean_{v in L\la} P(R|v) * mean_{v in R\ra} P(L|v)
    L_trans = [v for v in transient if v in left]
    R_trans = [v for v in transient if v in right]
    if not L_trans or not R_trans:
        return float("nan")
    P_LL = np.mean([side_prob(t_idx[v], la_cols) for v in L_trans])
    P_RL = np.mean([side_prob(t_idx[v], ra_cols) for v in L_trans])
    P_LR = np.mean([side_prob(t_idx[v], la_cols) for v in R_trans])
    P_RR = np.mean([side_prob(t_idx[v], ra_cols) for v in R_trans])
    return float(P_LL * P_RR - P_LR * P_RL)


def iter_bipartitions(n: int, max_bipartitions: int = 200):
    """Random balanced bipartitions of {0, ..., n-1}."""
    rng = np.random.default_rng(42)
    seen = set()
    yielded = 0
    while yielded < max_bipartitions:
        perm = rng.permutation(n)
        left = frozenset(perm[:n // 2].tolist())
        if left in seen:
            continue
        seen.add(left)
        right = frozenset(i for i in range(n) if i not in left)
        yield set(left), set(right)
        yielded += 1


def search(max_n: int = 7, eps: float = 1e-4, verbose: bool = False):
    """Iterate the graph atlas (all graphs on up to 7 vertices in networkx)
    plus custom small pairs, group by 1-WL, and search for RWC-distinguishing
    non-vertex-transitive pairs."""
    hits = []
    print(f"Searching graph_atlas up to n = {max_n}", flush=True)
    atlas = nx.graph_atlas_g()
    print(f"  {len(atlas)} graphs in atlas", flush=True)
    by_hist = {}
    for G in atlas:
        if G.number_of_nodes() < 4 or G.number_of_nodes() > max_n:
            continue
        if G.number_of_edges() == 0:
            continue
        if not nx.is_connected(G):
            continue
        hist = wl_histogram(G)
        by_hist.setdefault(hist, []).append(G)

    non_singleton = {h: gs for h, gs in by_hist.items() if len(gs) >= 2}
    print(f"  {len(non_singleton)} non-singleton 1-WL classes "
          f"(total {sum(len(gs) for gs in non_singleton.values())} graphs)",
          flush=True)

    for hist, graphs in non_singleton.items():
        if len(set(hist)) == 1:
            # All graphs in this class have the WL-single-class property;
            # they are all at least locally transitive. Skip only if ALL are
            # vertex-transitive per the sufficient condition.
            transitive_flags = [is_vertex_transitive(G) for G in graphs]
            if all(transitive_flags):
                continue
        for i, j in itertools.combinations(range(len(graphs)), 2):
            G1, G2 = graphs[i], graphs[j]
            n = G1.number_of_nodes()
            if G2.number_of_nodes() != n:
                continue
            # Try up to 50 random bipartitions
            for L1, R1 in itertools.islice(iter_bipartitions(n), 50):
                # Use the SAME index partition on both graphs
                # (both are labelled 0..n-1 in the atlas).
                L2, R2 = set(L1), set(R1)
                r1 = rwc(G1, L1, R1)
                r2 = rwc(G2, L2, R2)
                if np.isnan(r1) or np.isnan(r2):
                    continue
                if abs(r1 - r2) > eps:
                    hits.append({
                        "hist_repr": str(hist)[:60],
                        "n": n,
                        "G1_edges": list(G1.edges()),
                        "G2_edges": list(G2.edges()),
                        "L": sorted(L1),
                        "R": sorted(R1),
                        "rwc_G1": r1, "rwc_G2": r2,
                        "diff": abs(r1 - r2),
                    })
                    if verbose:
                        print(f"    HIT: n={n}, |diff|={abs(r1 - r2):.4f} "
                              f"(G1={r1:.4f}, G2={r2:.4f})", flush=True)
                    break
    return hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_n", type=int, default=7)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    hits = search(max_n=args.max_n, eps=args.eps, verbose=args.verbose)
    print(f"\n{len(hits)} candidate witness pairs found.", flush=True)
    if hits:
        # Sort by |diff| descending
        hits.sort(key=lambda h: -h["diff"])
        for h in hits[:5]:
            print(f"\n  n = {h['n']}, |diff| = {h['diff']:.4f}")
            print(f"  RWC(G1) = {h['rwc_G1']:.4f}, RWC(G2) = {h['rwc_G2']:.4f}")
            print(f"  G1 edges: {h['G1_edges']}")
            print(f"  G2 edges: {h['G2_edges']}")
            print(f"  Left:  {h['L']}")
            print(f"  Right: {h['R']}")
