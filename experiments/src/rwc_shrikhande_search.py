"""Exhaustive RWC search on the Shrikhande vs. 4x4 rook pair.

The Shrikhande and 4x4 rook graphs are the canonical strongly-regular
1-WL indistinguishable pair on 16 vertices (SRG(16, 6, 2, 2)). Under a
single fixed bipartition and authority choice they gave identical RWC in
our earlier attempt. Here we sweep random bipartitions and authority
choices to test whether ANY (bipartition, authority) pair distinguishes.

Setup:
  * Both graphs on 16 vertices, indexed 0..15.
  * A bipartition (L, R) on {0..15} induces the same bipartition on
    both graphs via the identity index correspondence.
  * Authorities on each side: the k_auth highest-degree vertices; ties
    broken lexicographically. Since both graphs are 6-regular, the tie-
    break is what actually picks the authority.
  * Search over N random balanced bipartitions.
  * For each, compute RWC(G_shrikhande, L, R) and RWC(G_rook, L, R) via
    exact absorbing Markov chain, report if |diff| > eps.

If no pair distinguishes across many random tries, that is empirical
evidence that RWC is invariant under the identity correspondence between
these two graphs, and the witness search on Shrikhande vs. rook
specifically is done.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np


def stable_wl_histogram(G: nx.Graph, initial: dict, max_iter: int = 32) -> Counter:
    """Run WL until the induced partition stabilises; return the histogram
    on a canonical colour space (relabelled 0..K-1)."""
    colours = dict(initial)
    for _ in range(max_iter):
        proposed = {}
        for v in G.nodes():
            neigh_multiset = tuple(sorted(colours[u] for u in G.neighbors(v)))
            proposed[v] = hash((colours[v], neigh_multiset))
        remap = {c: i for i, c in enumerate(sorted(set(proposed.values())))}
        new_colours = {v: remap[proposed[v]] for v in G.nodes()}
        if sorted(new_colours.values()) == sorted(colours.values()):
            colours = new_colours
            break
        colours = new_colours
    return Counter(colours.values())


def joint_wl_histograms(G1: nx.Graph, G2: nx.Graph, initial1: dict,
                         initial2: dict, max_iter: int = 32) -> tuple[Counter, Counter]:
    """Run WL on the disjoint union of the two graphs so colour classes
    are aligned across G1 and G2. Return histograms on the shared canonical
    colour space."""
    n1 = G1.number_of_nodes()
    U = nx.Graph()
    U.add_nodes_from(G1.nodes())
    for v in G1.nodes():
        for u in G1.neighbors(v):
            if u > v:
                U.add_edge(v, u)
    U.add_nodes_from(v + n1 for v in G2.nodes())
    for v in G2.nodes():
        for u in G2.neighbors(v):
            if u > v:
                U.add_edge(v + n1, u + n1)
    init_U = {**initial1, **{v + n1: initial2[v] for v in G2.nodes()}}
    colours_U = dict(init_U)
    for _ in range(max_iter):
        proposed = {}
        for v in U.nodes():
            neigh = tuple(sorted(colours_U[u] for u in U.neighbors(v)))
            proposed[v] = hash((colours_U[v], neigh))
        remap = {c: i for i, c in enumerate(sorted(set(proposed.values())))}
        new = {v: remap[proposed[v]] for v in U.nodes()}
        if sorted(new.values()) == sorted(colours_U.values()):
            colours_U = new
            break
        colours_U = new
    h1 = Counter(colours_U[v] for v in G1.nodes())
    h2 = Counter(colours_U[v + n1] for v in G2.nodes())
    return h1, h2


def shrikhande_graph() -> nx.Graph:
    """Shrikhande graph on 16 vertices, indexed 0..15.
    Vertices arranged as a 4x4 grid on the torus. Vertex (i, j) at index
    4*i + j. Edges: (i, j) ~ (i', j') if the difference (i'-i, j'-j)
    mod 4 belongs to a specific set of 6 nonzero differences."""
    G = nx.Graph()
    G.add_nodes_from(range(16))
    diffs = [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1)]
    for i in range(4):
        for j in range(4):
            v = 4 * i + j
            for di, dj in diffs:
                ii = (i + di) % 4
                jj = (j + dj) % 4
                u = 4 * ii + jj
                if u > v:
                    G.add_edge(v, u)
    assert G.number_of_edges() == 48  # SRG(16, 6, 2, 2)
    return G


def rook_graph() -> nx.Graph:
    """4x4 rook graph on 16 vertices, indexed 0..15.
    Vertex (i, j) at index 4*i + j. Edges: same row or same column, not
    self."""
    G = nx.Graph()
    G.add_nodes_from(range(16))
    for i in range(4):
        for j in range(4):
            v = 4 * i + j
            for k in range(4):
                if k != j:
                    u = 4 * i + k
                    if u > v:
                        G.add_edge(v, u)
                if k != i:
                    u = 4 * k + j
                    if u > v:
                        G.add_edge(v, u)
    assert G.number_of_edges() == 48
    return G


def rwc(G: nx.Graph, left: set, right: set, k_auth: int = 2) -> float:
    """RWC per Garimella et al. 2018 with degree+lex authority selection.
    Absorbing Markov chain computation via fundamental matrix."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    left_sorted = sorted(left, key=lambda v: (-G.degree(v), v))
    right_sorted = sorted(right, key=lambda v: (-G.degree(v), v))
    if len(left_sorted) < k_auth or len(right_sorted) < k_auth:
        return float("nan")
    la = set(left_sorted[:k_auth])
    ra = set(right_sorted[:k_auth])
    absorbing = la | ra
    transient = [v for v in nodes if v not in absorbing]
    if not transient:
        return float("nan")

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
    try:
        N = np.linalg.inv(np.eye(t) - Q)
    except np.linalg.LinAlgError:
        return float("nan")
    B = N @ R

    la_cols = [a_idx[u] for u in la]
    ra_cols = [a_idx[u] for u in ra]

    def side_prob(v_transient_idx: int, side_cols: list) -> float:
        return float(sum(B[v_transient_idx, c] for c in side_cols))

    L_trans = [v for v in transient if v in left]
    R_trans = [v for v in transient if v in right]
    if not L_trans or not R_trans:
        return float("nan")
    P_LL = np.mean([side_prob(t_idx[v], la_cols) for v in L_trans])
    P_RL = np.mean([side_prob(t_idx[v], ra_cols) for v in L_trans])
    P_LR = np.mean([side_prob(t_idx[v], la_cols) for v in R_trans])
    P_RR = np.mean([side_prob(t_idx[v], ra_cols) for v in R_trans])
    return float(P_LL * P_RR - P_LR * P_RL)


def search_random(n_bipartitions: int = 5000, eps: float = 1e-4,
                   k_auth: int = 2, seed: int = 42) -> dict:
    S = shrikhande_graph()
    R = rook_graph()
    n = 16
    rng = np.random.default_rng(seed)

    hits = []
    all_diffs = []
    for i in range(n_bipartitions):
        perm = rng.permutation(n)
        L = set(perm[:8].tolist())
        Rset = set(int(x) for x in perm[8:])
        rs = rwc(S, L, Rset, k_auth=k_auth)
        rr = rwc(R, L, Rset, k_auth=k_auth)
        if np.isnan(rs) or np.isnan(rr):
            continue
        d = abs(rs - rr)
        all_diffs.append(d)
        if d > eps:
            hits.append({"L": sorted(L), "R": sorted(Rset),
                          "rwc_S": rs, "rwc_R": rr, "diff": d})

    all_diffs = np.array(all_diffs)
    return {"n_bipartitions": int(len(all_diffs)),
            "n_hits": len(hits),
            "max_diff": float(all_diffs.max()) if len(all_diffs) else float("nan"),
            "mean_diff": float(all_diffs.mean()) if len(all_diffs) else float("nan"),
            "hits": sorted(hits, key=lambda h: -h["diff"])[:10]}


def search_exhaustive(k_left: int = 8, eps: float = 1e-4, k_auth: int = 2,
                       limit: int = None) -> dict:
    """Exhaustive search over balanced (8, 8) bipartitions.
    Reports RWC difference AND whether the bipartition-featured graphs
    are 1-WL equivalent (i.e., whether their stable WL colour histograms
    coincide on the disjoint union). Only hits with equal WL histograms
    are valid witnesses for RWC not-in-F_WL."""
    S = shrikhande_graph()
    R = rook_graph()
    n = 16
    nodes = list(range(n))
    hits = []
    valid_witnesses = []
    all_diffs = []
    count = 0
    for L_tuple in combinations(nodes, k_left):
        L = set(L_tuple)
        Rset = set(v for v in nodes if v not in L)
        if 0 not in L:
            continue
        rs = rwc(S, L, Rset, k_auth=k_auth)
        rr = rwc(R, L, Rset, k_auth=k_auth)
        count += 1
        if np.isnan(rs) or np.isnan(rr):
            continue
        d = abs(rs - rr)
        all_diffs.append(d)
        if d > eps:
            initS = {v: (1 if v in L else 2) for v in range(n)}
            initR = {v: (1 if v in L else 2) for v in range(n)}
            hS, hR = joint_wl_histograms(S, R, initS, initR)
            wl_equal = (hS == hR)
            hit = {"L": sorted(L), "R": sorted(Rset),
                    "rwc_S": rs, "rwc_R": rr, "diff": d,
                    "wl_equivalent": wl_equal,
                    "n_colours_S": len(hS), "n_colours_R": len(hR)}
            hits.append(hit)
            if wl_equal:
                valid_witnesses.append(hit)
        if limit is not None and count >= limit:
            break

    all_diffs = np.array(all_diffs)
    return {"n_bipartitions": int(count),
            "n_hits": len(hits),
            "n_valid_witnesses": len(valid_witnesses),
            "max_diff": float(all_diffs.max()) if len(all_diffs) else float("nan"),
            "mean_diff": float(all_diffs.mean()) if len(all_diffs) else float("nan"),
            "top_hits": sorted(hits, key=lambda h: -h["diff"])[:5],
            "top_valid_witnesses": sorted(valid_witnesses, key=lambda h: -h["diff"])[:5]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "exhaustive"], default="exhaustive")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--k_auth", type=int, default=2)
    args = ap.parse_args()

    if args.mode == "random":
        print(f"Random search: {args.n} bipartitions, k_auth={args.k_auth}, eps={args.eps}")
        r = search_random(n_bipartitions=args.n, eps=args.eps, k_auth=args.k_auth)
    else:
        print(f"Exhaustive search over balanced (8, 8) bipartitions, k_auth={args.k_auth}, eps={args.eps}")
        r = search_exhaustive(eps=args.eps, k_auth=args.k_auth)

    print(f"\n{r['n_bipartitions']} bipartitions checked; "
          f"{r['n_hits']} give |RWC diff| > {args.eps}; "
          f"of those, {r.get('n_valid_witnesses', '?')} are 1-WL-equivalent "
          f"under the bipartition-as-feature colouring.")
    print(f"max |diff| = {r['max_diff']:.6f}, mean = {r['mean_diff']:.6f}")

    if r.get("top_valid_witnesses"):
        print("\nTop valid witnesses (1-WL equivalent, RWC distinguishes):")
        for h in r["top_valid_witnesses"]:
            print(f"  RWC_S = {h['rwc_S']:.6f}, RWC_R = {h['rwc_R']:.6f}, "
                  f"|diff| = {h['diff']:.6f}")
            print(f"    L = {h['L']}")
            print(f"    R = {h['R']}")
            print(f"    colours: S={h['n_colours_S']}, R={h['n_colours_R']}")
    elif r.get("top_hits"):
        print("\nTop RWC-distinguishing bipartitions (WL histograms NOT matched):")
        for h in r["top_hits"]:
            print(f"  RWC_S = {h['rwc_S']:.6f}, RWC_R = {h['rwc_R']:.6f}, "
                  f"|diff| = {h['diff']:.6f}, WL-equiv? {h.get('wl_equivalent')}")

    out = Path(__file__).resolve().parents[1] / "results" / "rwc_search"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "shrikhande_rook.json").open("w") as f:
        json.dump(r, f, indent=2)
    print(f"\nWritten to {out / 'shrikhande_rook.json'}")
