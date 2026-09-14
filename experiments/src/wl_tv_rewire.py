"""Depth-T 1-WL total variation between original and rewired graphs.

For each of the four Twitter datasets, apply the exact same rewiring used
in Experiment 4 (degree- and Louvain-block-preserving), then compute the
total variation distance between the depth-T 1-WL colour histograms of
the original and rewired graphs for T = 1, 2, 3.

Two feature choices:
  (a) constant features: every vertex starts with colour 0. Depth-T WL
      colours then depend only on graph structure.
  (b) Louvain-block features: each vertex starts coloured by its
      Louvain-community id. This matches the setup EchoGAE sees when the
      community assignment enters as the label set.

The output pins what the rewiring preserves at each depth and grounds
Experiment 4's statement that the rewiring exactly preserves depth-1
statistics but not depth-2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np


ECHOGAE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
sys.path.insert(0, str(ECHOGAE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from echogae_reproduction import load_topic, compute_sbert_features, louvain_communities  # noqa: E402
from rewire import rewire_preserving_blocks  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "wl_tv_rewire"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def wl_colours_at_depth(G: nx.Graph, initial: dict, depth: int) -> dict:
    """Colour refinement for exactly `depth` iterations from `initial`.
    Returns a canonical relabelling into 0..K-1 so histograms can be compared
    across graphs by hash-equal colours."""
    colours = dict(initial)
    for _ in range(depth):
        proposed = {}
        for v in G.nodes():
            neigh_multiset = tuple(sorted(colours[u] for u in G.neighbors(v)))
            proposed[v] = hash((colours[v], neigh_multiset))
        # Canonicalise to 0..K-1 to keep hash values comparable across graphs.
        remap = {c: i for i, c in enumerate(sorted(set(proposed.values())))}
        colours = {v: remap[proposed[v]] for v in G.nodes()}
    return colours


def histogram(colours: dict) -> Counter:
    return Counter(colours.values())


def tv_distance(h1: Counter, h2: Counter, n_total: int) -> float:
    """Total variation between two histograms treated as distributions on
    the union of their colour vocabularies. Both histograms sum to n_total."""
    keys = set(h1.keys()) | set(h2.keys())
    tv = 0.5 * sum(abs(h1.get(k, 0) - h2.get(k, 0)) for k in keys) / n_total
    return float(tv)


def wl_colours_pair(G1: nx.Graph, G2: nx.Graph, initial1: dict, initial2: dict,
                    depth: int) -> tuple[Counter, Counter]:
    """Compute depth-T WL colour histograms for both graphs on a shared
    canonical colour space. To do this we run one round of WL refinement on
    the DISJOINT UNION of G1 and G2 at each step, so hash collisions align:
    the colour of a vertex in G1 at round t is the same as the colour of a
    vertex in G2 at round t iff their (colour_{t-1}, neighbour multiset)
    tuples are identical."""
    n1 = G1.number_of_nodes()
    # Relabel G2's nodes by offset to build a disjoint union.
    offset = n1
    U = nx.Graph()
    U.add_nodes_from(G1.nodes())
    for v in G1.nodes():
        for u in G1.neighbors(v):
            if u > v:
                U.add_edge(v, u)
    U.add_nodes_from(v + offset for v in G2.nodes())
    for v in G2.nodes():
        for u in G2.neighbors(v):
            if u > v:
                U.add_edge(v + offset, u + offset)
    initial_U = {**initial1, **{v + offset: initial2[v] for v in G2.nodes()}}
    colours_U = wl_colours_at_depth(U, initial_U, depth)
    c1 = {v: colours_U[v] for v in G1.nodes()}
    c2 = {v: colours_U[v + offset] for v in G2.nodes()}
    return histogram(c1), histogram(c2)


def prepare_graphs(topic: str, seed: int = 42, rewire_seconds: float = 90.0):
    print(f"\n=== {topic}: preparing graphs ===", flush=True)
    G = load_topic(topic)
    sbert = compute_sbert_features(topic)
    keep = [n for n in G.nodes if str(G.nodes[n]["user_id"]) in sbert]
    if len(keep) < G.number_of_nodes():
        Gs = G.subgraph(keep).copy()
        for n in Gs.nodes:
            Gs.nodes[n]["twitter_user_id"] = str(Gs.nodes[n]["user_id"])
        G = nx.convert_node_labels_to_integers(Gs, label_attribute="_orig")
    else:
        for n in G.nodes:
            G.nodes[n]["twitter_user_id"] = str(G.nodes[n]["user_id"])

    comm_ids = louvain_communities(G, resolution=0.05, seed=seed)
    unique, counts = np.unique(comm_ids, return_counts=True)
    order = np.argsort(-counts)
    top_ids = unique[order[:2]]
    block = np.zeros(G.number_of_nodes(), dtype=np.int8)
    for v in range(G.number_of_nodes()):
        c = comm_ids[v]
        if c == top_ids[0]:
            block[v] = 1
        elif c == top_ids[1]:
            block[v] = 2
        else:
            block[v] = 3

    print(f"  n = {G.number_of_nodes()}, m = {G.number_of_edges()}", flush=True)
    print("  Applying rewiring...", flush=True)
    G_rw, rw_stats = rewire_preserving_blocks(
        G, block, n_swaps=15000, seed=seed, max_seconds=rewire_seconds,
    )
    print(f"  triangles {rw_stats['triangles_before']} -> {rw_stats['triangles_after']}",
          flush=True)
    return G, G_rw, comm_ids.astype(int)


def measure_topic(topic: str, depths=(1, 2, 3), seed: int = 42,
                   rewire_seconds: float = 90.0) -> dict:
    G, G_rw, comm_ids = prepare_graphs(topic, seed=seed,
                                        rewire_seconds=rewire_seconds)
    n = G.number_of_nodes()

    out = {"topic": topic, "n_nodes": n, "n_edges": G.number_of_edges(),
           "n_edges_rw": G_rw.number_of_edges(),
           "depths": {}}

    # Constant features: initialise every vertex to 0.
    init_const = {v: 0 for v in G.nodes()}
    # Louvain-block features: initialise every vertex to its top-2 block id
    # (0 = "other", 1 = block A, 2 = block B) so WL sees the community
    # partition as the initial colouring.
    unique, counts = np.unique(comm_ids, return_counts=True)
    order = np.argsort(-counts)
    top_ids = unique[order[:2]]
    init_block = {}
    for v in G.nodes():
        c = comm_ids[v]
        if c == top_ids[0]:
            init_block[v] = 1
        elif c == top_ids[1]:
            init_block[v] = 2
        else:
            init_block[v] = 0

    for T in depths:
        h1_const, h2_const = wl_colours_pair(G, G_rw, init_const, init_const, T)
        tv_const = tv_distance(h1_const, h2_const, n)
        h1_block, h2_block = wl_colours_pair(G, G_rw, init_block, init_block, T)
        tv_block = tv_distance(h1_block, h2_block, n)
        out["depths"][T] = {
            "tv_constant": tv_const,
            "tv_block": tv_block,
            "n_colours_orig_const": len(h1_const),
            "n_colours_rw_const": len(h2_const),
            "n_colours_orig_block": len(h1_block),
            "n_colours_rw_block": len(h2_block),
        }
        print(f"  depth {T}: TV_const = {tv_const:.4f}, TV_block = {tv_block:.4f}, "
              f"|colours_const| = {len(h1_const)}/{len(h2_const)}, "
              f"|colours_block| = {len(h1_block)}/{len(h2_block)}",
              flush=True)

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+",
                    default=["gun", "abortion", "super_bowl", "sxsw"])
    ap.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rewire_seconds", type=float, default=90.0)
    args = ap.parse_args()

    all_out = {}
    for t in args.topics:
        try:
            all_out[t] = measure_topic(t, depths=tuple(args.depths),
                                        seed=args.seed,
                                        rewire_seconds=args.rewire_seconds)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"ERROR on {t}: {e}", flush=True)

    print("\n=== WL TV summary ===", flush=True)
    print(f"{'Topic':<12} {'T':<3} {'TV_const':<10} {'TV_block':<10}", flush=True)
    for t, r in all_out.items():
        for T, d in r["depths"].items():
            print(f"  {t:<10} {T:<3} {d['tv_constant']:<10.4f} {d['tv_block']:<10.4f}",
                  flush=True)

    with (RESULTS_DIR / "wl_tv.json").open("w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nWritten to {RESULTS_DIR / 'wl_tv.json'}", flush=True)
