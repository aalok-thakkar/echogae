"""Reddit-native audit.

Uses:
- SNAP soc-RedditHyperlinks (source-subreddit -> target-subreddit edges from
  posts, Jan 2014 - Apr 2017) as the interaction graph.
- Waller-Anderson (Nature 2021) partisan scores as node labels.

The audit runs the same three tests as the EchoGAE audit chapter:
(1) train EchoGAE on the constructed subreddit-subreddit graph; report the
community-level Echo Chamber Score; (2) fit a graph-blind reproduction; (3)
apply degree- and block-preserving rewiring; report score movement.
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


WA_DIR = Path(__file__).resolve().parents[1] / "vendor" / "reddit" / "wa" / "data"
SNAP_DIR = Path(__file__).resolve().parents[1] / "vendor" / "reddit"
ECHOGAE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
sys.path.insert(0, str(ECHOGAE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.EchoGAE import EchoGAE_algorithm
from src.echo_chamber_measure import EchoChamberMeasure
from rewire import rewire_preserving_blocks


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "reddit_snap"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_wa_scores() -> pd.DataFrame:
    scores = pd.read_csv(WA_DIR / "scores.csv")
    meta = pd.read_csv(WA_DIR / "embedding-metadata.tsv", sep="\t")
    scores["community"] = scores["community"].str.lower()
    meta["community"] = meta["community"].str.lower()
    m = scores.merge(meta[["community"]], on="community", how="inner")
    return m


def load_snap_hyperlinks() -> pd.DataFrame:
    """Return dataframe with SOURCE_SUBREDDIT, TARGET_SUBREDDIT, LINK_SENTIMENT."""
    # File is TSV with header.
    path = SNAP_DIR / "hyperlinks-body.tsv"
    if not path.exists():
        raise FileNotFoundError(f"SNAP hyperlinks file not found at {path}. "
                                 "Download from https://snap.stanford.edu/data/soc-RedditHyperlinks.html")
    df = pd.read_csv(path, sep="\t")
    df["SOURCE_SUBREDDIT"] = df["SOURCE_SUBREDDIT"].str.lower()
    df["TARGET_SUBREDDIT"] = df["TARGET_SUBREDDIT"].str.lower()
    return df


def build_subreddit_graph(hyperlinks: pd.DataFrame, wa: pd.DataFrame,
                          min_edge_weight: int = 3) -> tuple[nx.Graph, dict]:
    """Build undirected graph on the intersection of SNAP hyperlinks' subreddits
    and Waller-Anderson's scored subreddits, weighted by edge multiplicity."""
    wa_subs = set(wa["community"].tolist())
    hl = hyperlinks[hyperlinks["SOURCE_SUBREDDIT"].isin(wa_subs) &
                    hyperlinks["TARGET_SUBREDDIT"].isin(wa_subs)]
    edge_counts = (hl.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"])
                    .size().reset_index(name="w"))
    edge_counts = edge_counts[edge_counts["w"] >= min_edge_weight]
    # Symmetrise by summing.
    pairs = edge_counts.copy()
    pairs["u"] = pairs[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].min(axis=1)
    pairs["v"] = pairs[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].max(axis=1)
    pairs = pairs.groupby(["u", "v"]).agg(w=("w", "sum")).reset_index()
    G = nx.Graph()
    for _, r in pairs.iterrows():
        G.add_edge(r["u"], r["v"], weight=r["w"])
    # Restrict to largest connected component.
    if G.number_of_nodes() > 0:
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(cc).copy()
    G = nx.convert_node_labels_to_integers(G, label_attribute="subreddit")
    # Attach partisan score.
    sub_to_score = dict(zip(wa["community"], wa["partisan"]))
    partisan = {n: float(sub_to_score.get(G.nodes[n]["subreddit"], 0.0))
                for n in G.nodes}
    return G, partisan


def audit(min_edge_weight: int = 3, epochs: int = 200,
          rewire_seconds: float = 120.0, seed: int = 42) -> dict:
    print(f"=== Reddit-native audit ===")
    wa = load_wa_scores()
    print(f"  Waller-Anderson: {len(wa)} scored subreddits")
    hl = load_snap_hyperlinks()
    print(f"  SNAP hyperlinks: {len(hl)} rows")
    G, partisan = build_subreddit_graph(hl, wa, min_edge_weight=min_edge_weight)
    print(f"  Subreddit graph: n={G.number_of_nodes()}, m={G.number_of_edges()}, "
          f"avg_deg={2*G.number_of_edges()/max(G.number_of_nodes(),1):.2f}")

    if G.number_of_nodes() < 100:
        raise RuntimeError("Constructed graph too small; increase edge weight or drop filter.")

    labeled = {v: s for v, s in partisan.items() if abs(s) > 1e-6}
    print(f"  Labelled (partisan != 0) subset: {len(labeled)}")

    print("  Training EchoGAE...")
    features = {v: np.array([partisan[v], 1.0], dtype=np.float32) for v in G.nodes}
    emb = EchoGAE_algorithm(G, user_embeddings=features, epochs=epochs,
                             show_progress=False)
    labeled_idx = sorted(labeled.keys())
    sub_emb = emb[labeled_idx]
    sign_labels = np.array([1 if partisan[v] > 0 else -1 for v in labeled_idx])
    ec = EchoChamberMeasure(sub_emb, sign_labels)
    community_score = float(ec.echo_chamber_index())
    per_node = np.array([ec.metric(i) for i in range(len(labeled_idx))])
    print(f"  Community-level ECS: {community_score:.4f} (per-node std {per_node.std():.4f})")

    # Graph-blind reproduction.
    rows = []
    for v in labeled_idx:
        s = partisan[v]
        neighs = list(G.neighbors(v))
        neigh_scores = [partisan[u] for u in neighs if u in labeled]
        rows.append({
            "s": s, "sign": np.sign(s),
            "deg": G.degree[v],
            "labeled_neigh": len(neigh_scores),
            "mean_neigh_s": float(np.mean(neigh_scores)) if neigh_scores else 0.0,
            "std_neigh_s": float(np.std(neigh_scores)) if neigh_scores else 0.0,
            "same_sign_frac": (sum(1 for x in neigh_scores if np.sign(x) == np.sign(s))
                                / max(1, len(neigh_scores))),
            "s_deg": s * G.degree[v],
        })
    X = pd.DataFrame(rows).values
    y = per_node
    r2s = []
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        r2s.append(r2_score(y[te], Ridge(alpha=1e-3).fit(X[tr], y[tr]).predict(X[te])))
    r2_mean = float(np.mean(r2s))
    r2_std = float(np.std(r2s))
    print(f"  Graph-blind R^2: {r2_mean:.4f} +- {r2_std:.4f}")

    # Rewiring.
    print("  Rewiring...")
    block = np.zeros(G.number_of_nodes(), dtype=np.int8)
    for v, s in labeled.items():
        block[v] = 1 if s > 0 else 2
    clus_before = float(nx.average_clustering(G))
    G_rw, rw_stats = rewire_preserving_blocks(
        G, block, n_swaps=10000, seed=seed, max_seconds=rewire_seconds,
    )
    clus_after = float(nx.average_clustering(G_rw))
    print(f"  Clustering: {clus_before:.4f} -> {clus_after:.4f}")
    print(f"  Triangles: {rw_stats['triangles_before']} -> {rw_stats['triangles_after']}")
    print("  Retraining EchoGAE on rewired graph...")
    emb2 = EchoGAE_algorithm(G_rw, user_embeddings=features, epochs=epochs,
                              show_progress=False)
    sub_emb2 = emb2[labeled_idx]
    ec2 = EchoChamberMeasure(sub_emb2, sign_labels)
    community2 = float(ec2.echo_chamber_index())
    delta = community2 - community_score
    print(f"  Community score before/after: {community_score:.4f} -> {community2:.4f} "
          f"(delta {delta:+.4f})")

    out = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_labeled": len(labeled),
        "min_edge_weight": min_edge_weight,
        "epochs": epochs,
        "community_score": community_score,
        "per_node_std": float(per_node.std()),
        "graph_blind_r2_mean": r2_mean,
        "graph_blind_r2_std": r2_std,
        "community_score_rewired": community2,
        "delta_score": delta,
        "clus_before": clus_before,
        "clus_after": clus_after,
        "triangles_before": rw_stats["triangles_before"],
        "triangles_after": rw_stats["triangles_after"],
    }
    with (RESULTS_DIR / "audit.json").open("w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_edge_weight", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--rewire_seconds", type=float, default=120.0)
    args = ap.parse_args()
    audit(min_edge_weight=args.min_edge_weight, epochs=args.epochs,
          rewire_seconds=args.rewire_seconds)
