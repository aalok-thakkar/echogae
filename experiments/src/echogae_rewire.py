"""EchoGAE rewiring invariance test.

Apply a degree- and block-preserving rewiring that destroys triangles, retrain EchoGAE
on the rewired graph, and compare the Echo Chamber Index. If EchoGAE is 1-WL invariant
(as the dichotomy theorem says any MPNN is), the ECI should be nearly unchanged even
though the graph's clustering coefficient collapses. A closure-based polarization
measure (e.g. GED) should move.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


ECHOGAE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
DATA_DIR = ECHOGAE_DIR / "data"
sys.path.insert(0, str(ECHOGAE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.EchoGAE import EchoGAE_algorithm
from src.echo_chamber_measure import EchoChamberMeasure
from rewire import rewire_preserving_blocks


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "echogae_rewire"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load(topic: str):
    G = nx.read_gml(DATA_DIR / topic / "graph.gml")
    if G.is_directed():
        G = G.to_undirected()
    G = nx.convert_node_labels_to_integers(G, label_attribute="user_id")
    labels = pd.read_feather(DATA_DIR / topic / "allsides.feather")
    return G, labels


def align_labels(G: nx.Graph, labels_df: pd.DataFrame):
    m = dict(zip(labels_df["user_id"].astype(str), labels_df["allsides_score"]))
    return {n: float(m[G.nodes[n]["user_id"]])
            for n in G.nodes if str(G.nodes[n]["user_id"]) in m}


def sign_block(G: nx.Graph, labeled: dict) -> np.ndarray:
    """Block = sign(allsides_score) for labeled nodes; 0 for unlabeled."""
    block = np.zeros(G.number_of_nodes(), dtype=np.int8)
    for v, s in labeled.items():
        block[v] = 1 if s > 0 else 2  # 1 = right, 2 = left; 0 = unlabeled
    return block


def evaluate_eci(G: nx.Graph, labeled: dict, epochs: int, seed: int = 42) -> dict:
    import torch
    torch.manual_seed(seed)
    embeddings = EchoGAE_algorithm(G, epochs=epochs, show_progress=False)
    label_idx = sorted(labeled.keys())
    label_arr = np.array([np.sign(labeled[v]) for v in label_idx])
    label_arr[label_arr == 0] = 1
    ec = EchoChamberMeasure(embeddings[label_idx], label_arr, metric="euclidean")
    overall = float(ec.echo_chamber_index())
    per_node = np.array([ec.metric(i) for i in range(len(label_idx))])
    return {"overall_eci": overall,
            "per_node_mean": float(per_node.mean()),
            "per_node_std": float(per_node.std()),
            "n_labeled": len(label_idx)}


def ged_hohmann(G: nx.Graph, labeled: dict, max_nodes: int = 1500) -> float:
    """Generalized Euclidean Distance of Hohmann, Devriendt, Coscia (Sci Adv 2023),
    delta = sqrt((o+ - o-)^T L^dagger (o+ - o-)), on the induced subgraph of labeled
    nodes' largest connected component. Restricted to at most max_nodes to keep the
    dense pseudoinverse tractable (O(n^3)); larger graphs use the induced subgraph
    on labeled nodes, then its largest connected component. Returns NaN if the
    subgraph is too small or disconnected in a way that leaves no signal.
    """
    if not labeled:
        return float("nan")
    H = G.subgraph(labeled.keys()).copy()
    if H.number_of_edges() == 0 or not any(True for _ in nx.connected_components(H)):
        return float("nan")
    cc = max(nx.connected_components(H), key=len)
    if len(cc) < 5:
        return float("nan")
    if len(cc) > max_nodes:
        # Truncate to a random sample of labeled nodes with a fixed seed for
        # comparability across before/after.
        import numpy as _np
        rng = _np.random.default_rng(0)
        sub = set(rng.choice(sorted(cc), size=max_nodes, replace=False).tolist())
        Hs = H.subgraph(sub).copy()
        cc2 = max(nx.connected_components(Hs), key=len)
        Hs = Hs.subgraph(cc2).copy()
    else:
        Hs = H.subgraph(cc).copy()
    nodes = sorted(Hs.nodes)
    o = np.array([1.0 if labeled[v] > 0 else -1.0 for v in nodes])
    o_pos = np.clip(o, 0, None)
    o_neg = np.clip(-o, 0, None)
    diff = o_pos - o_neg
    L = nx.laplacian_matrix(Hs, nodelist=nodes).astype(float).toarray()
    L_pinv = np.linalg.pinv(L)
    val = float(diff @ L_pinv @ diff)
    return float(np.sqrt(max(val, 0.0)))


def graph_stats(G: nx.Graph, labeled: dict) -> dict:
    m = G.number_of_edges()
    triangles = sum(nx.triangles(G).values()) // 3
    # Homophily on labeled edges.
    labeled_set = set(labeled.keys())
    labeled_edges = [(u, v) for u, v in G.edges if u in labeled_set and v in labeled_set]
    if labeled_edges:
        same = sum(1 for u, v in labeled_edges if np.sign(labeled[u]) == np.sign(labeled[v]))
        homophily = same / len(labeled_edges)
    else:
        homophily = float("nan")
    ged = ged_hohmann(G, labeled)
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": m,
        "avg_clustering": float(nx.average_clustering(G)),
        "n_triangles": triangles,
        "labeled_edge_count": len(labeled_edges),
        "edge_homophily_on_labeled": homophily,
        "ged": ged,
    }


def run_topic(topic: str, epochs: int, n_swaps: int, rewire_time: float,
              seed: int = 0) -> dict:
    print(f"\n=== Rewiring audit on {topic} ===")
    G, labels = load(topic)
    labeled = align_labels(G, labels)
    block = sign_block(G, labeled)
    stats_before = graph_stats(G, labeled)
    print(f"  Before: n={stats_before['n_nodes']}, m={stats_before['n_edges']}, "
          f"clus={stats_before['avg_clustering']:.4f}, triangles={stats_before['n_triangles']}, "
          f"homophily(labeled)={stats_before['edge_homophily_on_labeled']:.4f}")

    print(f"  Training EchoGAE (original)...")
    eci_before = evaluate_eci(G, labeled, epochs=epochs, seed=seed)
    print(f"  ECI (original): {eci_before['overall_eci']:.4f} (per-node std {eci_before['per_node_std']:.4f})")

    print(f"  Rewiring with n_swaps={n_swaps}, max_seconds={rewire_time}...")
    G_rw, rw_stats = rewire_preserving_blocks(
        G, block, n_swaps=n_swaps, seed=seed, max_seconds=rewire_time,
        prefer_triangle_destroy=True, log_every=0,
    )
    stats_after = graph_stats(G_rw, labeled)
    print(f"  After: n={stats_after['n_nodes']}, m={stats_after['n_edges']}, "
          f"clus={stats_after['avg_clustering']:.4f}, triangles={stats_after['n_triangles']}, "
          f"homophily(labeled)={stats_after['edge_homophily_on_labeled']:.4f}")
    print(f"  Rewire stats: {rw_stats['accepted']} accepted, "
          f"{rw_stats['dropped_no_triangle_decrease']} rejected (no decrease), "
          f"{rw_stats['dropped_multi_edge']} rejected (multi-edge)")

    print(f"  Training EchoGAE (rewired)...")
    eci_after = evaluate_eci(G_rw, labeled, epochs=epochs, seed=seed)
    print(f"  ECI (rewired):  {eci_after['overall_eci']:.4f} (per-node std {eci_after['per_node_std']:.4f})")

    delta_eci = eci_after['overall_eci'] - eci_before['overall_eci']
    delta_clus = stats_after['avg_clustering'] - stats_before['avg_clustering']
    delta_tri = stats_after['n_triangles'] - stats_before['n_triangles']

    print(f"\n  Deltas: ECI {delta_eci:+.4f}, clustering {delta_clus:+.4f}, triangles {delta_tri:+d}")
    if abs(delta_eci) < 0.02:
        print(f"  --> ECI moved by less than 0.02, closure collapsed by "
              f"{100 * abs(delta_clus) / max(stats_before['avg_clustering'], 1e-9):.0f}%. "
              f"Invariance consistent with dichotomy.")

    out = {
        "topic": topic, "epochs": epochs, "n_swaps": n_swaps,
        "before": {**stats_before, **eci_before},
        "after":  {**stats_after,  **eci_after},
        "rewire_stats": rw_stats,
        "deltas": {"eci": delta_eci, "clustering": delta_clus, "triangles": delta_tri},
    }
    with (RESULTS_DIR / f"{topic}.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+", default=["sxsw"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_swaps", type=int, default=20_000)
    ap.add_argument("--rewire_time", type=float, default=120.0)
    args = ap.parse_args()
    for t in args.topics:
        run_topic(t, epochs=args.epochs, n_swaps=args.n_swaps, rewire_time=args.rewire_time)
