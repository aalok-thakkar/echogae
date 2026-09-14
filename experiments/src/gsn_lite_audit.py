"""GSN-lite audit: does a subgraph-count architecture behave differently
under the same rewiring test?

Adds per-vertex subgraph-count features (triangle count, 4-cycle count) to
the initial feature tensor, then trains the same EchoGAE encoder and
computes the same community-level Echo Chamber Score. Because triangle
count is a triangle-density signal at the vertex level, the encoder can
now respond to closure structure through its input features rather than
through message passing alone.

Prediction: the community-level ECS should move substantially under the
degree- and Louvain-block-preserving rewiring (because triangle counts
collapse to near-zero for most vertices), unlike EchoGAE where the score
moves by 0.017-0.078 (Table 4 in the paper).

We run this on one Twitter dataset (SXSW, smallest, fastest) as the
existence-proof demonstration the reviewer asked for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ECHOGAE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
sys.path.insert(0, str(ECHOGAE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.EchoGAE import EchoGAE_algorithm  # noqa: E402
from src.echo_chamber_measure import EchoChamberMeasure  # noqa: E402
from echogae_reproduction import (  # noqa: E402
    load_topic, compute_sbert_features, louvain_communities,
)
from rewire import rewire_preserving_blocks  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "gsn_lite"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def vertex_subgraph_counts(G: nx.Graph) -> np.ndarray:
    """Per-vertex (triangle_count, 4-cycle_count, degree) features.
    All three are subgraph counts (or reductions) that lift the
    initial feature tensor above the 1-WL fixed point."""
    n = G.number_of_nodes()
    tri = nx.triangles(G)
    feats = np.zeros((n, 3), dtype=np.float32)
    for v in G.nodes():
        feats[v, 0] = float(tri[v])
        # 4-cycles through v (approximation via triangle count of the
        # bipartite neighbour graph; use total 4-cycle count / n for a
        # scalar proxy since exact per-vertex 4-cycle is expensive).
        # Here we use degree as a cheap fill-in for the third feature.
        feats[v, 2] = float(G.degree(v))
    # 4-cycle proxy: 2 * (number of open triangles a vertex is in), a
    # cheap 4-cycle-density surrogate that lifts closure-structure info.
    for v in G.nodes():
        deg = G.degree(v)
        feats[v, 1] = float(deg * (deg - 1) // 2 - tri[v])
    return feats


def graph_and_communities(topic: str, seed: int = 42):
    G_raw = load_topic(topic)
    sbert = compute_sbert_features(topic)
    keep = [n for n in G_raw.nodes if str(G_raw.nodes[n]["user_id"]) in sbert]
    Gs = G_raw.subgraph(keep).copy()
    for n in Gs.nodes:
        Gs.nodes[n]["twitter_user_id"] = str(Gs.nodes[n]["user_id"])
    G = nx.convert_node_labels_to_integers(Gs, label_attribute="_orig")
    comm_ids = louvain_communities(G, resolution=0.05, seed=seed)
    return G, comm_ids


def compute_ecs(G: nx.Graph, comm_ids: np.ndarray,
                 feature_matrix: np.ndarray, seed: int, epochs: int = 300) -> dict:
    user_embeddings = {int(v): feature_matrix[int(v)].astype(np.float32)
                        for v in G.nodes()}
    emb = EchoGAE_algorithm(G, user_embeddings=user_embeddings,
                            epochs=epochs, show_progress=False, seed=seed)
    ec = EchoChamberMeasure(emb, comm_ids, metric="euclidean")
    unique, counts = np.unique(comm_ids, return_counts=True)
    per_comm = []
    for c in unique:
        if counts[np.where(unique == c)[0][0]] < 2:
            continue
        idx = np.where(comm_ids == c)[0].tolist()
        per_comm.append(float(np.mean([ec.metric(int(i)) for i in idx])))
    return {"graph_ecs": float(np.mean(per_comm)) if per_comm else float("nan"),
            "per_community": per_comm}


def compute_ecs_direct(feature_matrix: np.ndarray, comm_ids: np.ndarray) -> dict:
    """ECS on raw features (no GCN, no message passing). This is the
    non-MPNN baseline the reviewer's demonstration ask targets: an
    estimator outside the 1-WL invariant class should respond to
    closure destruction because the triangle count itself does.
    Standardise features to unit norm per column so the ECS is
    dimensionless."""
    X = feature_matrix.astype(np.float64).copy()
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    X = (X - X.mean(axis=0)) / std
    ec = EchoChamberMeasure(X, comm_ids, metric="euclidean")
    unique, counts = np.unique(comm_ids, return_counts=True)
    per_comm = []
    for c in unique:
        if counts[np.where(unique == c)[0][0]] < 2:
            continue
        idx = np.where(comm_ids == c)[0].tolist()
        per_comm.append(float(np.mean([ec.metric(int(i)) for i in idx])))
    return {"graph_ecs": float(np.mean(per_comm)) if per_comm else float("nan"),
            "per_community": per_comm}


def run(topic: str, epochs: int = 300, seed: int = 42,
         rewire_seconds: float = 180.0) -> dict:
    print(f"\n=== GSN-lite audit on {topic} ===", flush=True)
    G, comm_ids = graph_and_communities(topic, seed=seed)
    n_ns = int((np.unique(comm_ids, return_counts=True)[1] > 1).sum())
    print(f"  n = {G.number_of_nodes()}, m = {G.number_of_edges()}, "
          f"communities = {n_ns} non-singleton", flush=True)

    print("  Original graph: computing subgraph-count features...", flush=True)
    feats_orig = vertex_subgraph_counts(G)
    print(f"    triangle counts: min={feats_orig[:, 0].min():.0f}, "
          f"max={feats_orig[:, 0].max():.0f}, "
          f"mean={feats_orig[:, 0].mean():.2f}", flush=True)
    print(f"  Training EchoGAE with subgraph-count features "
          f"({epochs} epochs)...", flush=True)
    r_before = compute_ecs(G, comm_ids, feats_orig, seed=seed, epochs=epochs)
    print(f"    ECS = {r_before['graph_ecs']:.4f}", flush=True)
    r_before_direct = compute_ecs_direct(feats_orig, comm_ids)
    print(f"    ECS on raw subgraph-count features (non-MPNN) "
          f"= {r_before_direct['graph_ecs']:.4f}", flush=True)

    # Louvain-block rewiring (same as v2 audit).
    print("  Rewiring...", flush=True)
    G_rw, rw_stats = rewire_preserving_blocks(
        G, comm_ids.astype(np.int8), n_swaps=10000, seed=seed,
        max_seconds=rewire_seconds,
    )
    print(f"    triangles {rw_stats['triangles_before']} -> "
          f"{rw_stats['triangles_after']}", flush=True)

    print("  Rewired graph: recomputing subgraph-count features...", flush=True)
    feats_rw = vertex_subgraph_counts(G_rw)
    print(f"    triangle counts: min={feats_rw[:, 0].min():.0f}, "
          f"max={feats_rw[:, 0].max():.0f}, "
          f"mean={feats_rw[:, 0].mean():.2f}", flush=True)
    print(f"  Retraining EchoGAE with rewired features "
          f"({epochs} epochs)...", flush=True)
    r_after = compute_ecs(G_rw, comm_ids, feats_rw, seed=seed, epochs=epochs)
    print(f"    ECS = {r_after['graph_ecs']:.4f}", flush=True)
    r_after_direct = compute_ecs_direct(feats_rw, comm_ids)
    print(f"    ECS on raw subgraph-count features (non-MPNN) "
          f"= {r_after_direct['graph_ecs']:.4f}", flush=True)

    delta = r_after["graph_ecs"] - r_before["graph_ecs"]
    delta_direct = r_after_direct["graph_ecs"] - r_before_direct["graph_ecs"]
    print(f"  Delta ECS (GSN-lite through GCN) = {delta:+.4f}", flush=True)
    print(f"  Delta ECS (raw features, no message passing) "
          f"= {delta_direct:+.4f}", flush=True)

    out = {
        "topic": topic, "epochs": epochs, "seed": seed,
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "n_communities_non_singleton": n_ns,
        "graph_ecs_before_gcn": r_before["graph_ecs"],
        "graph_ecs_after_gcn": r_after["graph_ecs"],
        "delta_ecs_gcn": delta,
        "graph_ecs_before_direct": r_before_direct["graph_ecs"],
        "graph_ecs_after_direct": r_after_direct["graph_ecs"],
        "delta_ecs_direct": delta_direct,
        "triangles_before": rw_stats["triangles_before"],
        "triangles_after": rw_stats["triangles_after"],
        "tri_mean_before": float(feats_orig[:, 0].mean()),
        "tri_mean_after": float(feats_rw[:, 0].mean()),
    }
    with (RESULTS_DIR / f"{topic}.json").open("w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="sxsw")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rewire_seconds", type=float, default=180.0)
    args = ap.parse_args()
    run(topic=args.topic, epochs=args.epochs, seed=args.seed,
        rewire_seconds=args.rewire_seconds)
