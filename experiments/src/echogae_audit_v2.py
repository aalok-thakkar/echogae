"""EchoGAE audit v2: mirrors the paper's pipeline exactly.

Differences from `echogae_audit.py`:
- Uses Louvain communities (resolution=0.05, seed=42) as the label set,
  not AllSides signs.
- Uses Sentence-BERT tweet embeddings as node features, not the identity
  matrix.
- Includes all users, not just AllSides-labelled ones.
- Reports the paper's exact graph-level ECS: unweighted mean of
  per-community ECS* over non-singleton communities.

Runs three experiments:
  1. Report reproduction against the paper's Table 2.
  2. Graph-blind reproduction: fit a small per-community regression from
     graph-blind features to per-community ECS*.
  3. Rewiring: preserve Louvain-block edge counts, retrain EchoGAE,
     recompute ECS. Keep the ORIGINAL Louvain partition as the fixed
     label set during rewiring (so the audit isolates rewiring's effect
     on the embedding, not on community reassignment).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


ECHOGAE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
DATA_DIR = ECHOGAE_DIR / "data"
sys.path.insert(0, str(ECHOGAE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.EchoGAE import EchoGAE_algorithm
from src.echo_chamber_measure import EchoChamberMeasure
from rewire import rewire_preserving_blocks, connectivity_report
from echogae_reproduction import (
    load_topic, compute_sbert_features, louvain_communities,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "echogae_audit_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def per_community_ecs(embeddings: np.ndarray, comm_ids: np.ndarray) -> dict[int, float]:
    """Per-community ECS* on non-singleton communities."""
    unique, counts = np.unique(comm_ids, return_counts=True)
    ns_mask = counts > 1
    ec = EchoChamberMeasure(embeddings, comm_ids, metric="euclidean")
    per = {}
    for c, keep in zip(unique, ns_mask):
        if not keep:
            continue
        idx = np.where(comm_ids == c)[0].tolist()
        per[int(c)] = float(np.mean([ec.metric(int(i)) for i in idx]))
    return per


def graph_level_ecs(per: dict[int, float]) -> float:
    return float(np.mean(list(per.values()))) if per else float("nan")


def community_features(G: nx.Graph, comm_ids: np.ndarray) -> pd.DataFrame:
    """Graph-blind features per non-singleton community: size, mean degree,
    internal edge fraction, external edge count, average clustering restricted
    to the community. All are 1-WL invariant given the community assignment."""
    rows = []
    unique, counts = np.unique(comm_ids, return_counts=True)
    for c, sz in zip(unique, counts):
        if sz < 2:
            continue
        members = np.where(comm_ids == c)[0].tolist()
        member_set = set(members)
        degs = [G.degree[v] for v in members]
        internal = 0
        external = 0
        for v in members:
            for u in G.neighbors(v):
                if u in member_set:
                    internal += 1
                else:
                    external += 1
        internal //= 2
        sub_G = G.subgraph(members)
        rows.append({
            "community": int(c),
            "size": int(sz),
            "mean_deg": float(np.mean(degs)),
            "median_deg": float(np.median(degs)),
            "internal_edges": int(internal),
            "external_edges": int(external),
            "internal_edge_frac": float(internal / max(internal + external, 1)),
            "sub_clustering": float(nx.average_clustering(sub_G)),
        })
    return pd.DataFrame(rows)


def rewire_with_louvain_blocks(G: nx.Graph, comm_ids: np.ndarray, seed: int,
                                 rewire_seconds: float) -> tuple[nx.Graph, dict]:
    """Rewire using Louvain communities (top-2 by size) as blocks. Vertices
    not in the top-2 communities are pooled into a third block."""
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
    return rewire_preserving_blocks(G, block, n_swaps=15000, seed=seed,
                                     max_seconds=rewire_seconds)


def run_topic(topic: str, epochs: int = 300, seed: int = 42,
               rewire_seconds: float = 120.0) -> dict:
    print(f"\n=== EchoGAE v2 audit on {topic} ===")
    G = load_topic(topic)
    print(f"  Graph: n={G.number_of_nodes()}, m={G.number_of_edges()}")

    print("  Computing Sentence-BERT features...")
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
    user_embeddings = {n: sbert[G.nodes[n]["twitter_user_id"]] for n in G.nodes}
    print(f"  Users with SBERT features: {G.number_of_nodes()}")

    print("  Running Louvain (resolution=0.05)...")
    comm_ids = louvain_communities(G, resolution=0.05, seed=seed)
    unique, counts = np.unique(comm_ids, return_counts=True)
    ns = int((counts > 1).sum())
    print(f"  Louvain: {len(unique)} total, {ns} non-singleton "
          f"(sizes: {sorted([int(c) for c in counts if c > 1], reverse=True)[:5]})")

    print(f"  Training EchoGAE for {epochs} epochs (SBERT features)...")
    embeddings = EchoGAE_algorithm(G, user_embeddings=user_embeddings,
                                     epochs=epochs, show_progress=False)

    per_comm = per_community_ecs(embeddings, comm_ids)
    ecs = graph_level_ecs(per_comm)
    print(f"  Graph-level ECS: {ecs:.4f}")

    # Graph-blind reproduction on per-community ECS*.
    feat_df = community_features(G, comm_ids)
    y_map = feat_df["community"].map(per_comm).values
    X = feat_df.drop(columns=["community"]).values

    # With few communities, use leave-one-out and report the correlation-
    # squared of held-out predictions against true values (Pearson^2 on the
    # LOO predictions). This avoids the per-fold R^2 being ill-defined when
    # a fold has one sample.
    if len(feat_df) >= 3:
        preds = np.zeros_like(y_map, dtype=float)
        for i in range(len(feat_df)):
            mask = np.ones(len(feat_df), dtype=bool); mask[i] = False
            preds[i] = Ridge(alpha=1e-3).fit(X[mask], y_map[mask]).predict(X[i:i+1])[0]
        if np.std(y_map) > 1e-6:
            r2_mean = float(np.corrcoef(preds, y_map)[0, 1] ** 2)
        else:
            r2_mean = float("nan")
        r2_std = 0.0
    else:
        # Not enough communities; report in-sample fit.
        model = Ridge(alpha=1e-3).fit(X, y_map)
        r2_mean = float(r2_score(y_map, model.predict(X)))
        r2_std = 0.0
    print(f"  Graph-blind R^2 on {len(feat_df)} communities: "
          f"{r2_mean:.4f} +- {r2_std:.4f}")

    # Rewiring test.
    print(f"  Rewiring...")
    clus_before = float(nx.average_clustering(G))
    G_rw, rw_stats = rewire_with_louvain_blocks(G, comm_ids, seed=seed,
                                                  rewire_seconds=rewire_seconds)
    clus_after = float(nx.average_clustering(G_rw))
    # Connectivity: pool the two largest Louvain communities as "labelled".
    labelled_pool = set()
    unique_c, counts_c = np.unique(comm_ids, return_counts=True)
    top2 = unique_c[np.argsort(-counts_c)[:2]]
    for v in range(len(comm_ids)):
        if comm_ids[v] in top2:
            labelled_pool.add(v)
    conn = connectivity_report(G, G_rw, labelled=labelled_pool)
    print(f"  Clustering: {clus_before:.4f} -> {clus_after:.4f}")
    print(f"  Triangles: {rw_stats['triangles_before']} -> {rw_stats['triangles_after']}")
    print(f"  Components: {conn['n_components_before']} -> {conn['n_components_after']}, "
          f"LCC size: {conn['largest_component_size_before']} -> "
          f"{conn['largest_component_size_after']}, "
          f"top-2-community vertices in LCC after: "
          f"{conn['labelled_in_lcc_after']}/{conn['labelled_total']}")
    print("  Retraining EchoGAE on rewired graph...")
    embeddings_rw = EchoGAE_algorithm(G_rw, user_embeddings=user_embeddings,
                                        epochs=epochs, show_progress=False)
    per_comm_rw = per_community_ecs(embeddings_rw, comm_ids)  # same labels
    ecs_rw = graph_level_ecs(per_comm_rw)
    print(f"  Graph-level ECS after rewiring: {ecs_rw:.4f} "
          f"(delta {ecs_rw - ecs:+.4f})")

    out = {
        "topic": topic,
        "epochs": epochs,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_communities_non_singleton": int(len(feat_df)),
        "graph_level_ecs": ecs,
        "per_community_ecs": per_comm,
        "graph_blind_r2_mean": r2_mean,
        "graph_blind_r2_std": r2_std,
        "clus_before": clus_before,
        "clus_after": clus_after,
        "triangles_before": rw_stats["triangles_before"],
        "triangles_after": rw_stats["triangles_after"],
        "graph_level_ecs_rewired": ecs_rw,
        "delta_ecs": ecs_rw - ecs,
        "connectivity": conn,
    }
    with (RESULTS_DIR / f"{topic}.json").open("w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+",
                    default=["gun", "abortion", "super_bowl", "sxsw"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rewire_seconds", type=float, default=120.0)
    args = ap.parse_args()
    results = []
    for t in args.topics:
        try:
            results.append(run_topic(t, epochs=args.epochs, seed=args.seed,
                                      rewire_seconds=args.rewire_seconds))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR on {t}: {e}")

    print("\n=== Summary ===")
    print(f"{'Topic':<12} {'Paper':<7} {'ECS':<7} {'R^2':<7} {'dECS(rewire)':<12} "
          f"{'clus_bf':<8} {'clus_af':<8}")
    published = {"gun": 0.714, "abortion": 0.626, "super_bowl": 0.485, "sxsw": 0.465}
    for r in results:
        p = published.get(r["topic"], float("nan"))
        print(f"  {r['topic']:<10} {p:<7.3f} {r['graph_level_ecs']:<7.3f} "
              f"{r['graph_blind_r2_mean']:<7.3f} {r['delta_ecs']:<+12.3f} "
              f"{r['clus_before']:<8.3f} {r['clus_after']:<8.3f}")
