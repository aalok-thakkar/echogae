"""Reproduce EchoGAE's Table 2 exactly.

The reproduction below matches the setup in Alatawi et al. (ASONAM 2023):

  1. Communities come from Louvain on the (undirected) retweet graph.
  2. ECS is computed over ALL vertices, not just the AllSides-labelled subset.
  3. Distances are Euclidean in the EchoGAE embedding space.
  4. The graph-level ECS is the unweighted mean of per-community ECS*.

Features: this version uses the identity matrix (EchoGAE's fallback path
when user_embeddings is None). Sentence-BERT tweet features are a follow-up.

The published Table 2 values are:
    Gun 0.714, Abortion 0.626, Super Bowl 0.485, SXSW 0.465.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ECHOGAE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
DATA_DIR = ECHOGAE_DIR / "data"
sys.path.insert(0, str(ECHOGAE_DIR))

from src.EchoGAE import EchoGAE_algorithm
from src.echo_chamber_measure import EchoChamberMeasure


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "echogae_reproduction"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_topic(topic: str) -> nx.Graph:
    G = nx.read_gml(DATA_DIR / topic / "graph.gml")
    if G.is_directed():
        G = G.to_undirected()
    G = nx.convert_node_labels_to_integers(G, label_attribute="user_id")
    return G


def compute_sbert_features(topic: str, cache: bool = True) -> dict[str, np.ndarray]:
    """Encode each user's tweets via Sentence-BERT (all-MiniLM-L6-v2) and
    average per user. Returns dict from user_id (str) to 384-dim vector.
    Uses EchoGAE's own tweet-preprocessing pipeline.
    """
    cache_path = DATA_DIR / topic / "sbert_embeddings.npz"
    if cache and cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as f:
            return dict(zip(f["user_ids"].astype(str), f["embeddings"]))

    from sentence_transformers import SentenceTransformer
    sys.path.insert(0, str(ECHOGAE_DIR))
    from src.tweet_preprocessing import preprocess_tweet_for_bert

    df = pd.read_feather(DATA_DIR / topic / "tweets.feather")
    print(f"    loaded {len(df)} users' tweets")

    def preprocess(tweets: list[str]) -> list[str]:
        out = []
        for tw in tweets:
            processed = preprocess_tweet_for_bert(tw)
            if len(processed) > 1:
                out.append(" ".join(processed))
        return out

    df["tweets"] = df["tweets"].apply(preprocess)
    df = df[df["tweets"].apply(len) > 0].reset_index(drop=True)
    print(f"    {len(df)} users after preprocessing")

    print("    loading Sentence-BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("    encoding tweets...")
    embs = []
    for _, row in df.iterrows():
        tweets = row["tweets"]
        enc = model.encode(tweets, show_progress_bar=False)
        embs.append(np.mean(enc, axis=0))
    embs = np.stack(embs)

    if cache:
        np.savez(cache_path, user_ids=df["user_id"].astype(str).values, embeddings=embs)
    return dict(zip(df["user_id"].astype(str).values, embs))


def louvain_communities(G: nx.Graph, resolution: float = 0.05,
                         seed: int = 42) -> np.ndarray:
    """Run Louvain on the LCC with the resolution used by EchoGAE's
    load_data.py (0.05, much coarser than the networkx default of 1.0).
    Vertices outside the LCC get their own singleton ids.
    """
    from networkx.algorithms.community.louvain import louvain_communities as louv

    n = G.number_of_nodes()
    comm_ids = -np.ones(n, dtype=np.int64)

    cc = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(cc).copy()
    parts = list(louv(Gc, resolution=resolution, seed=seed))
    for i, part in enumerate(parts):
        for v in part:
            comm_ids[v] = i

    next_id = len(parts)
    for v in range(n):
        if comm_ids[v] == -1:
            comm_ids[v] = next_id
            next_id += 1
    return comm_ids


def run_topic(topic: str, epochs: int = 300, resolution: float = 0.05,
               seed: int = 42, use_sbert: bool = True) -> dict:
    print(f"=== EchoGAE reproduction on {topic} ===")
    G = load_topic(topic)
    print(f"  Graph: n={G.number_of_nodes()}, m={G.number_of_edges()}")

    user_embeddings = None
    if use_sbert:
        print("  Computing Sentence-BERT tweet embeddings...")
        sbert = compute_sbert_features(topic)
        # Filter graph to users with SBERT features. user_id in G.nodes[n]
        # is the original string from graph.gml (set by load_topic).
        keep = [n for n in G.nodes if str(G.nodes[n]["user_id"]) in sbert]
        if len(keep) < G.number_of_nodes():
            print(f"    Restricting graph to {len(keep)} of {G.number_of_nodes()} "
                  "users with SBERT features")
            Gs = G.subgraph(keep).copy()
            # Preserve the original user_id as 'twitter_user_id' before
            # relabelling (which would overwrite 'user_id').
            for n in Gs.nodes:
                Gs.nodes[n]["twitter_user_id"] = str(Gs.nodes[n]["user_id"])
            G = nx.convert_node_labels_to_integers(Gs, label_attribute="_orig")
        else:
            for n in G.nodes:
                G.nodes[n]["twitter_user_id"] = str(G.nodes[n]["user_id"])
        user_embeddings = {n: sbert[G.nodes[n]["twitter_user_id"]] for n in G.nodes}

    # Louvain communities matching EchoGAE's load_data.py default.
    print(f"  Running Louvain (resolution={resolution})...")
    comm_ids = louvain_communities(G, resolution=resolution, seed=seed)
    unique, counts = np.unique(comm_ids, return_counts=True)
    # Filter singleton communities from the ECS average per Louvain convention.
    non_singleton = counts > 1
    ns_count = int(non_singleton.sum())
    print(f"  Louvain: {len(unique)} total communities, "
          f"{ns_count} non-singleton (sizes: {sorted(counts[non_singleton].tolist())[-10:]})")

    # Train EchoGAE with Sentence-BERT features (paper default) or identity
    # (fallback for a features-off ablation).
    feat_desc = "Sentence-BERT" if user_embeddings is not None else "identity"
    print(f"  Training EchoGAE for {epochs} epochs ({feat_desc} features)...")
    embeddings = EchoGAE_algorithm(G, user_embeddings=user_embeddings,
                                     epochs=epochs, show_progress=False)
    print(f"  Embedding shape: {embeddings.shape}")

    # Compute ECS using Louvain communities as the label set, on ALL vertices.
    ec = EchoChamberMeasure(embeddings, comm_ids, metric="euclidean")

    # Per-community ECS*: mean of per-vertex metric() within community.
    per_comm = {}
    for c in unique:
        if not non_singleton[np.where(unique == c)[0][0]]:
            continue
        member_idx = np.where(comm_ids == c)[0].tolist()
        if len(member_idx) < 2:
            continue
        scores = [ec.metric(int(i)) for i in member_idx]
        per_comm[int(c)] = float(np.mean(scores))

    # Graph-level ECS: unweighted average over non-singleton communities.
    graph_ecs = float(np.mean(list(per_comm.values()))) if per_comm else float("nan")

    print(f"  Graph-level ECS (mean over {len(per_comm)} non-singleton "
          f"communities): {graph_ecs:.4f}")
    print(f"  Per-community ECS* values: "
          f"min={min(per_comm.values()):.4f}, "
          f"max={max(per_comm.values()):.4f}, "
          f"median={np.median(list(per_comm.values())):.4f}")

    out = {
        "topic": topic,
        "epochs": epochs,
        "seed": seed,
        "resolution": resolution,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_communities_louvain": int(len(unique)),
        "n_communities_non_singleton": ns_count,
        "graph_level_ecs": graph_ecs,
        "per_community_ecs": per_comm,
        "features": feat_desc,
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
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--no_sbert", action="store_true",
                    help="Use identity features instead of Sentence-BERT")
    args = ap.parse_args()
    out = []
    for t in args.topics:
        try:
            out.append(run_topic(t, epochs=args.epochs, seed=args.seed,
                                  resolution=args.resolution,
                                  use_sbert=not args.no_sbert))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR on {t}: {e}")

    print("\n=== Reproduction Summary ===")
    print(f"{'Topic':<12} {'Their ECS':<11} {'Our ECS':<10} {'Communities':<12}")
    published = {"gun": 0.714, "abortion": 0.626, "super_bowl": 0.485, "sxsw": 0.465}
    for r in out:
        p = published.get(r["topic"], float("nan"))
        print(f"  {r['topic']:<10} {p:<11.4f} {r['graph_level_ecs']:<10.4f} "
              f"{r['n_communities_non_singleton']}")
