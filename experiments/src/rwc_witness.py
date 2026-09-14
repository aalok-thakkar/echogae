"""Exact random walk controversy computation on the Shrikhande / rook pair.

Both graphs are strongly regular with parameters (16, 6, 2, 2). Both are
1-WL indistinguishable in the unfeatured case. This script builds each
graph, defines a common bipartition and authority set, and computes the
random walk controversy score by solving the two-absorbing-state chain
exactly via linear algebra.

Output: exact rational or floating-point values of RWC on each graph.
If the values differ, RWC is not 1-WL invariant and Proposition 3(c)
in the paper is witnessed rigorously.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def rook_graph_4x4() -> nx.Graph:
    """The 4x4 rook's graph L_2(4). Vertices are (i, j) for i, j in 0..3.
    Two vertices adjacent iff they share a row (same i) or column (same j)."""
    G = nx.Graph()
    verts = [(i, j) for i in range(4) for j in range(4)]
    G.add_nodes_from(verts)
    for u in verts:
        for v in verts:
            if u == v:
                continue
            if u[0] == v[0] or u[1] == v[1]:
                G.add_edge(u, v)
    return G


def shrikhande_graph() -> nx.Graph:
    """The Shrikhande graph. Vertices are Z/4 x Z/4 = {(i, j) : 0 <= i, j <= 3}.
    Two vertices adjacent iff their difference mod 4 is one of
    +/-(1, 0), +/-(0, 1), +/-(1, 1)."""
    G = nx.Graph()
    verts = [(i, j) for i in range(4) for j in range(4)]
    G.add_nodes_from(verts)
    diffs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    for u in verts:
        for du, dv in diffs:
            v = ((u[0] + du) % 4, (u[1] + dv) % 4)
            if v != u:
                G.add_edge(u, v)
    return G


def check_srg(G: nx.Graph, name: str) -> None:
    assert G.number_of_nodes() == 16, name
    assert G.number_of_edges() == 48, f"{name} has {G.number_of_edges()} edges, expected 48"
    degs = [G.degree[v] for v in G.nodes]
    assert all(d == 6 for d in degs), name
    lambdas, mus = set(), set()
    for u in G.nodes:
        for v in G.nodes:
            if u == v:
                continue
            common = len(set(G.neighbors(u)) & set(G.neighbors(v)))
            if G.has_edge(u, v):
                lambdas.add(common)
            else:
                mus.add(common)
    assert lambdas == {2}, f"{name}: lambdas={lambdas}"
    assert mus == {2}, f"{name}: mus={mus}"
    print(f"  {name}: SRG(16, 6, 2, 2) verified")


def wl_fingerprint(G: nx.Graph, x: dict, rounds: int = 8) -> tuple:
    """Compute the graph-level 1-WL fingerprint as a canonical multiset."""
    colours = {v: x.get(v, 0) for v in G.nodes}
    for _ in range(rounds):
        new_colours = {}
        for v in G.nodes:
            neigh = tuple(sorted(colours[u] for u in G.neighbors(v)))
            new_colours[v] = hash((colours[v], neigh))
        if new_colours == colours:
            break
        colours = new_colours
    return tuple(sorted(colours.values()))


def compute_rwc(G: nx.Graph, S_plus: set, S_minus: set,
                 auth_plus, auth_minus) -> tuple[float, dict]:
    """Random walk controversy score. Simple random walk starting at a
    uniform vertex of a side terminates upon hitting either auth_plus or
    auth_minus. P_st = probability that a walk starting on side s hits
    the authority of side t first. RWC = P_++ P_-- - P_+- P_-+.

    Absorbing Markov chain: order nodes with non-absorbing first, then
    the two absorbing at the end. Solve (I - Q) X = R where Q is
    non-absorbing block, R is transition to absorbing. Column c of X is
    the vector of absorption probabilities at absorbing state c.
    """
    nodes = list(G.nodes)
    absorbers = [auth_plus, auth_minus]
    non_abs = [v for v in nodes if v not in absorbers]
    idx = {v: i for i, v in enumerate(non_abs)}
    n = len(non_abs)

    Q = np.zeros((n, n))
    R = np.zeros((n, 2))
    for v in non_abs:
        neighs = list(G.neighbors(v))
        d = len(neighs)
        if d == 0:
            continue
        p = 1.0 / d
        for u in neighs:
            if u == auth_plus:
                R[idx[v], 0] += p
            elif u == auth_minus:
                R[idx[v], 1] += p
            else:
                Q[idx[v], idx[u]] += p

    # X = (I - Q)^{-1} R
    X = np.linalg.solve(np.eye(n) - Q, R)

    # For each non-absorbing v, X[v, 0] = P(absorbed at auth_plus | start at v).
    P = {"plus_plus": [], "plus_minus": [], "minus_plus": [], "minus_minus": []}
    for v in non_abs:
        prob_plus = X[idx[v], 0]
        prob_minus = X[idx[v], 1]
        if v in S_plus:
            P["plus_plus"].append(prob_plus)
            P["plus_minus"].append(prob_minus)
        elif v in S_minus:
            P["minus_plus"].append(prob_plus)
            P["minus_minus"].append(prob_minus)

    P_pp = float(np.mean(P["plus_plus"]))
    P_pm = float(np.mean(P["plus_minus"]))
    P_mp = float(np.mean(P["minus_plus"]))
    P_mm = float(np.mean(P["minus_minus"]))
    rwc = P_pp * P_mm - P_pm * P_mp
    return rwc, {"P_++": P_pp, "P_+-": P_pm, "P_-+": P_mp, "P_--": P_mm}


def main() -> None:
    print("Building graphs...")
    R = rook_graph_4x4()
    S = shrikhande_graph()
    check_srg(R, "Rook 4x4")
    check_srg(S, "Shrikhande")

    # Both graphs are non-isomorphic though SRG-equivalent.
    print(f"  Rook   is triangle-free? {all(len(set(R.neighbors(u)) & set(R.neighbors(v))) == 2 for u, v in R.edges) and False}")
    # Cheaper: count K_4 subgraphs (rook has many, Shrikhande has zero rows/cols).
    from itertools import combinations
    def n_k4(G: nx.Graph) -> int:
        c = 0
        for quad in combinations(G.nodes, 4):
            if all(G.has_edge(u, v) for u, v in combinations(quad, 2)):
                c += 1
        return c
    print(f"  Rook has {n_k4(R)} K_4 subgraphs; Shrikhande has {n_k4(S)}")

    # Constant-feature 1-WL fingerprint: identical single colour class on both.
    zero = {v: 0 for v in R.nodes}
    fp_R = wl_fingerprint(R, zero)
    fp_S = wl_fingerprint(S, zero)
    assert fp_R == fp_S, "1-WL fingerprints differ; witness is wrong"
    print(f"  1-WL fingerprints match (both = {len(set(fp_R))} distinct colour class)")

    # Bipartition: S_+ = vertices with i in {0, 1}, S_- = vertices with i in {2, 3}.
    S_plus = {(i, j) for i in (0, 1) for j in range(4)}
    S_minus = {(i, j) for i in (2, 3) for j in range(4)}
    print(f"\nBipartition: |S_+| = {len(S_plus)}, |S_-| = {len(S_minus)}")

    # Verify featured 1-WL is still degenerate under this bipartition-as-feature.
    feat = {v: 1 if v in S_plus else -1 for v in R.nodes}
    fp_R_f = wl_fingerprint(R, feat)
    fp_S_f = wl_fingerprint(S, feat)
    match_featured = fp_R_f == fp_S_f
    print(f"  Featured 1-WL fingerprints match: {match_featured}")
    if not match_featured:
        # Count colour classes to see how far apart they are.
        print(f"  Rook features have {len(set(fp_R_f))} distinct colours; "
              f"Shrikhande has {len(set(fp_S_f))} distinct colours")

    # Authorities: k=1 per side.
    auth_plus_R = (0, 0)   # rook
    auth_minus_R = (2, 2)  # rook
    auth_plus_S = (0, 0)   # shrikhande
    auth_minus_S = (2, 2)  # shrikhande

    print("\nComputing RWC on rook 4x4...")
    rwc_R, P_R = compute_rwc(R, S_plus, S_minus, auth_plus_R, auth_minus_R)
    print(f"  P_++ = {P_R['P_++']:.6f}")
    print(f"  P_+- = {P_R['P_+-']:.6f}")
    print(f"  P_-+ = {P_R['P_-+']:.6f}")
    print(f"  P_-- = {P_R['P_--']:.6f}")
    print(f"  RWC(R) = {rwc_R:.6f}")

    print("\nComputing RWC on Shrikhande...")
    rwc_S, P_S = compute_rwc(S, S_plus, S_minus, auth_plus_S, auth_minus_S)
    print(f"  P_++ = {P_S['P_++']:.6f}")
    print(f"  P_+- = {P_S['P_+-']:.6f}")
    print(f"  P_-+ = {P_S['P_-+']:.6f}")
    print(f"  P_-- = {P_S['P_--']:.6f}")
    print(f"  RWC(S) = {rwc_S:.6f}")

    print(f"\nRWC gap: {abs(rwc_R - rwc_S):.6f}")
    if abs(rwc_R - rwc_S) > 1e-4:
        print("RWC separates the pair; Proposition 3(c) witnessed rigorously.")
    else:
        print("RWC does not separate this specific pair/bipartition; "
              "witness would need a different setup.")


if __name__ == "__main__":
    main()
