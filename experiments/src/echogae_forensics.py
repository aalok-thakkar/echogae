"""Reproduction forensics for EchoGAE Table 2.

For each of the four EchoGAE datasets:
  1. Node/edge counts on the release's graph.gml vs. what the paper's
     Table 1 reports.
  2. Degree distribution: mean, median, max, and the fraction of edges
     touching the top-1% degree vertices.
  3. Louvain partition sizes (seed=42, resolution=0.05) vs. what the
     paper implies (its per-community ECS* count).
  4. Feature-vertex overlap: number of vertices in the graph that carry
     SBERT features (the release ships the graph but not the aggregated
     tweet text; our reproduction re-computes SBERT from the shipped
     'tweets.feather').
  5. Per-community ECS* on Abortion at seed=42 (the topic our
     reproduction agreed with the paper on), which is the fair
     comparison point to look for shape agreement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" /
                        "echo-chamber-score"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.EchoGAE import EchoGAE_algorithm  # noqa: E402
from src.echo_chamber_measure import EchoChamberMeasure  # noqa: E402
from echogae_reproduction import (  # noqa: E402
    load_topic, compute_sbert_features, louvain_communities,
)

DATA = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score" / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "echogae_forensics"
RESULTS.mkdir(parents=True, exist_ok=True)

PAPER_TABLE_1 = {
    "gun": {"n": 7905, "m": 18908},
    "abortion": {"n": 5556, "m": 11697},
    "super_bowl": {"n": 5473, "m": 8754},
    "sxsw": {"n": 2438, "m": 5329},
}

PAPER_TABLE_2_ECS = {
    "gun": 0.714, "abortion": 0.626, "super_bowl": 0.485, "sxsw": 0.465,
}


def summarise_graph(topic: str) -> dict:
    G_raw = load_topic(topic)
    n_raw, m_raw = G_raw.number_of_nodes(), G_raw.number_of_edges()
    sbert = compute_sbert_features(topic)
    keep = [n for n in G_raw.nodes if str(G_raw.nodes[n]["user_id"]) in sbert]
    Gs = G_raw.subgraph(keep).copy()
    for n in Gs.nodes:
        Gs.nodes[n]["twitter_user_id"] = str(Gs.nodes[n]["user_id"])
    G = nx.convert_node_labels_to_integers(Gs, label_attribute="_orig")
    degs = np.array([d for _, d in G.degree()])
    top1_cut = int(np.ceil(0.01 * len(degs)))
    top1_idx = np.argsort(degs)[-top1_cut:] if top1_cut > 0 else []
    top1_edge_touch = sum(1 for u, v in G.edges
                           if u in set(top1_idx.tolist()) or
                           v in set(top1_idx.tolist()))
    ci = louvain_communities(G, resolution=0.05, seed=42)
    _, cs = np.unique(ci, return_counts=True)
    ns = int((cs > 1).sum())
    return {
        "topic": topic,
        "paper_n": PAPER_TABLE_1[topic]["n"],
        "paper_m": PAPER_TABLE_1[topic]["m"],
        "release_n": int(n_raw),
        "release_m": int(m_raw),
        "sbert_covered_n": int(G.number_of_nodes()),
        "sbert_covered_m": int(G.number_of_edges()),
        "n_gap": int(PAPER_TABLE_1[topic]["n"] - G.number_of_nodes()),
        "m_gap": int(PAPER_TABLE_1[topic]["m"] - G.number_of_edges()),
        "deg_mean": float(degs.mean()),
        "deg_median": float(np.median(degs)),
        "deg_max": int(degs.max()),
        "top1pct_edge_touch_frac":
            float(top1_edge_touch) / max(G.number_of_edges(), 1),
        "louvain_n_communities": int(len(cs)),
        "louvain_n_non_singleton": ns,
        "louvain_block_sizes":
            sorted([int(c) for c in cs if c > 1], reverse=True)[:8],
    }


def ecs_per_community_abortion() -> dict:
    """The one topic where our mean reproduces the paper. Return the
    per-community ECS* values so a reader can compare shapes."""
    G_raw = load_topic("abortion")
    sbert = compute_sbert_features("abortion")
    keep = [n for n in G_raw.nodes if str(G_raw.nodes[n]["user_id"]) in sbert]
    Gs = G_raw.subgraph(keep).copy()
    for n in Gs.nodes:
        Gs.nodes[n]["twitter_user_id"] = str(Gs.nodes[n]["user_id"])
    G = nx.convert_node_labels_to_integers(Gs, label_attribute="_orig")
    emb = {n: sbert[G.nodes[n]["twitter_user_id"]] for n in G.nodes}
    ci = louvain_communities(G, resolution=0.05, seed=42)
    E = EchoGAE_algorithm(G, user_embeddings=emb, epochs=300,
                           show_progress=False, seed=42)
    ec = EchoChamberMeasure(E, ci, metric="euclidean")
    u, c = np.unique(ci, return_counts=True)
    out = []
    for cid, sz in zip(u, c):
        if sz <= 1:
            continue
        idx = np.where(ci == cid)[0]
        m = float(np.mean([ec.metric(int(i)) for i in idx]))
        out.append({"community_id": int(cid), "size": int(sz),
                     "ecs_star": m})
    out.sort(key=lambda r: -r["size"])
    return {"topic": "abortion", "seed": 42, "n_communities": len(out),
             "per_community": out,
             "graph_ecs_mean": float(np.mean([r["ecs_star"] for r in out]))}


def main():
    out = {}
    for t in ("gun", "abortion", "super_bowl", "sxsw"):
        print(f"\n== {t}", flush=True)
        s = summarise_graph(t)
        out[t] = s
        print(json.dumps(s, indent=2), flush=True)
    (RESULTS / "graph_stats.json").write_text(json.dumps(out, indent=2))

    print("\n== per-community ECS* on abortion (seed 42)", flush=True)
    pc = ecs_per_community_abortion()
    print(json.dumps(pc, indent=2), flush=True)
    (RESULTS / "abortion_per_community.json").write_text(
        json.dumps(pc, indent=2))
    print("\nWritten to results/echogae_forensics/", flush=True)


if __name__ == "__main__":
    main()
