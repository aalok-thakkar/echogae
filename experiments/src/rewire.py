"""Degree-and-block-preserving rewiring that destroys triangle closure.

The audit test: given a featured graph (G, x), produce a rewired G' with
  - identical degree sequence,
  - identical block membership (features on each vertex unchanged),
  - identical within-block and cross-block edge counts,
  - reduced local clustering (triangles destroyed as much as possible).

Any MPNN estimator whose output depends only on 1-WL-invariant statistics of (G, x)
should give nearly identical outputs on G and G'. A closure-based polarization measure
(clustering, generalized Euclidean distance) should differ substantially.

Method: targeted double-edge swaps constrained to preserve block-conditional degrees.
An edge swap picks two edges (a, b) and (c, d) with a, b, c, d distinct and replaces
them with (a, c) and (b, d), which preserves every degree. Restrict swaps to pairs
where both new edges are within the same block-pair category as the corresponding old
edges (so within-vs-cross-block edge counts stay fixed). Prefer swaps that destroy
existing triangles.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def block_of(v: int, block: np.ndarray) -> int:
    return int(block[v])


def edge_category(u: int, v: int, block: np.ndarray) -> tuple[int, int]:
    """Unordered pair of block ids for an edge."""
    a, b = int(block[u]), int(block[v])
    return (min(a, b), max(a, b))


def count_triangles_containing(G: nx.Graph, u: int, v: int) -> int:
    """Number of triangles that include the edge (u, v)."""
    return len(set(G.neighbors(u)) & set(G.neighbors(v)))


def rewire_preserving_blocks(
    G: nx.Graph,
    block: np.ndarray,
    n_swaps: int = 10_000,
    seed: int = 0,
    prefer_triangle_destroy: bool = True,
    log_every: int = 0,
    max_seconds: float = 60.0,
) -> tuple[nx.Graph, dict]:
    """Perform degree-and-block-preserving double-edge swaps.

    Each swap picks two edges (a, b), (c, d) with matching edge-category (unordered
    block-pair) and rewires to (a, c), (b, d) provided
      - a, b, c, d are distinct,
      - (a, c) and (b, d) both have that same category (block preservation),
      - neither (a, c) nor (b, d) is already an edge (simple-graph preservation).
    When prefer_triangle_destroy is True, accept only if the swap strictly decreases
    the local triangle count. Caps: n_swaps accepted or max_seconds wall-time.

    Implementation note. Direct adj-dict access + set intersections; no graph copies
    in the inner loop.
    """
    import time

    G = G.copy()
    rng = np.random.default_rng(seed)
    # Precompute categories for every edge; store as list of ((u, v), category).
    edges = list(G.edges())
    # Neighbour sets: mutable, mirrored to G. Use these for triangle counts (fast).
    adj = {v: set(G.neighbors(v)) for v in G.nodes}

    def local_triangles(u: int, v: int) -> int:
        return len(adj[u] & adj[v])

    triangles_before = sum(nx.triangles(G).values()) // 3
    accepted = 0
    proposed = 0
    dropped_multi = 0
    dropped_no_decrease = 0
    start = time.time()

    n_edges = len(edges)
    while accepted < n_swaps and time.time() - start < max_seconds:
        proposed += 1
        # Sample two edges uniformly. Categories are checked implicitly by the
        # block-preservation test on the new edges.
        i = int(rng.integers(n_edges))
        j = int(rng.integers(n_edges))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        cat_ab = edge_category(a, b, block)
        cat_cd = edge_category(c, d, block)
        if cat_ab != cat_cd:
            continue
        if edge_category(a, c, block) != cat_ab:
            continue
        if edge_category(b, d, block) != cat_ab:
            continue
        if c in adj[a] or d in adj[b]:
            dropped_multi += 1
            continue

        if prefer_triangle_destroy:
            # Change in triangle count. Old edges (a, b), (c, d) removed; new (a, c),
            # (b, d) added. Compute triangle counts on each of these four edges
            # before and after using the current adj sets (before we mutate).
            t_before_ab = local_triangles(a, b)
            t_before_cd = local_triangles(c, d)
            # For (a, c) after removing (a, b) and (c, d): common neighbours of a
            # and c in the modified graph. Modification affects only vertices b, d
            # in adj[a] / adj[c]. Since a-b and c-d are removed, b is removed from
            # adj[a] and d from adj[c]. So common neighbours of a and c: (adj[a] -
            # {b}) & (adj[c] - {d}).
            common_ac = (adj[a] - {b}) & (adj[c] - {d})
            t_after_ac = len(common_ac)
            common_bd = (adj[b] - {a}) & (adj[d] - {c})
            t_after_bd = len(common_bd)
            delta = (t_after_ac + t_after_bd) - (t_before_ab + t_before_cd)
            if delta >= 0:
                dropped_no_decrease += 1
                continue

        # Accept: mutate adj and G together.
        adj[a].discard(b); adj[b].discard(a)
        adj[c].discard(d); adj[d].discard(c)
        adj[a].add(c); adj[c].add(a)
        adj[b].add(d); adj[d].add(b)
        G.remove_edge(a, b)
        G.remove_edge(c, d)
        G.add_edge(a, c)
        G.add_edge(b, d)
        edges[i] = (a, c)
        edges[j] = (b, d)
        accepted += 1
        if log_every and accepted % log_every == 0:
            t_now = sum(nx.triangles(G).values()) // 3
            print(f"  accepted={accepted}, triangles={t_now}, proposed={proposed}")

    triangles_after = sum(nx.triangles(G).values()) // 3
    # Connectivity metrics after rewiring.
    cc_after = list(nx.connected_components(G))
    lcc_after = max((len(c) for c in cc_after), default=0)
    stats = {
        "accepted": accepted,
        "proposed": proposed,
        "dropped_multi_edge": dropped_multi,
        "dropped_no_triangle_decrease": dropped_no_decrease,
        "triangles_before": triangles_before,
        "triangles_after": triangles_after,
        "elapsed_seconds": time.time() - start,
        "n_components_after": len(cc_after),
        "largest_component_size_after": lcc_after,
        "largest_component_fraction_after": (lcc_after / G.number_of_nodes()
                                              if G.number_of_nodes() > 0 else 0.0),
    }
    return G, stats


def connectivity_report(G_before: nx.Graph, G_after: nx.Graph,
                         labelled: set | None = None) -> dict:
    """Report connectivity change and (if labelled subset provided) whether
    all labelled vertices remain in the largest connected component after
    rewiring."""
    cc_before = list(nx.connected_components(G_before))
    cc_after = list(nx.connected_components(G_after))
    lcc_before = max(cc_before, key=len) if cc_before else set()
    lcc_after = max(cc_after, key=len) if cc_after else set()
    out = {
        "n_components_before": len(cc_before),
        "n_components_after": len(cc_after),
        "largest_component_size_before": len(lcc_before),
        "largest_component_size_after": len(lcc_after),
    }
    if labelled is not None:
        lset = set(labelled)
        out["labelled_in_lcc_before"] = int(len(lset & lcc_before))
        out["labelled_in_lcc_after"] = int(len(lset & lcc_after))
        out["labelled_total"] = int(len(lset))
    return out


def verify_preservation(G_before: nx.Graph, G_after: nx.Graph, block: np.ndarray) -> dict:
    from collections import Counter

    deg_before = Counter(dict(G_before.degree()).values())
    deg_after = Counter(dict(G_after.degree()).values())
    # Per-vertex degree equality.
    per_vertex_ok = all(G_before.degree[v] == G_after.degree[v] for v in G_before.nodes)
    # Edge category counts.
    cat_before = Counter(edge_category(u, v, block) for u, v in G_before.edges)
    cat_after = Counter(edge_category(u, v, block) for u, v in G_after.edges)
    return {
        "degree_sequence_matches": deg_before == deg_after,
        "per_vertex_degree_matches": per_vertex_ok,
        "edge_category_counts_match": cat_before == cat_after,
        "edge_categories_before": dict(cat_before),
        "edge_categories_after": dict(cat_after),
    }


if __name__ == "__main__":
    # Smoke test on a small planted-partition graph.
    rng = np.random.default_rng(0)
    n = 80
    half = n // 2
    block = np.zeros(n, dtype=np.int8)
    block[half:] = 1
    p, q = 0.25, 0.05
    prob = np.where(block[:, None] == block[None, :], p, q)
    upper = np.triu(rng.random((n, n)) < prob, k=1)
    A = upper | upper.T
    G = nx.from_numpy_array(A.astype(np.int8))

    print(f"Before: n={G.number_of_nodes()}, m={G.number_of_edges()}, "
          f"clus={nx.average_clustering(G):.4f}, "
          f"triangles={sum(nx.triangles(G).values()) // 3}")
    G_rw, stats = rewire_preserving_blocks(G, block, n_swaps=2000, log_every=500)
    print(f"After : n={G_rw.number_of_nodes()}, m={G_rw.number_of_edges()}, "
          f"clus={nx.average_clustering(G_rw):.4f}, "
          f"triangles={sum(nx.triangles(G_rw).values()) // 3}")
    print("Stats:", stats)
    print("Preservation:", verify_preservation(G, G_rw, block))
