"""Feature-subset ablation for the graph-blind audit.

Reports how the cross-validated R^2 of the graph-blind reproduction varies
as we trim the feature set. If R^2 is already high with 2 or 3 features,
the "eight features are cherry-picked" objection is defused.

Runs two ablations:
  (A) per-vertex v1 (AllSides labels, per-vertex ECS): 2, 3, 5, 8 features.
  (B) per-community v2 (Louvain communities, per-community ECS*): 1, 2, 3, 5, 7
      features (bounded by number of non-singleton communities per topic).
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
from echogae_reproduction import (
    load_topic, compute_sbert_features, louvain_communities,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "feature_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----- v1: per-vertex on AllSides labels ---------------------------------

V1_FEATURE_SETS = {
    "1_score":      ["s"],
    "2_score_deg":  ["s", "deg"],
    "3_add_meanN":  ["s", "deg", "mean_neigh_s"],
    "5_hom_full":   ["s", "sign", "deg", "same_sign_frac", "mean_neigh_s"],
    "8_full":       ["s", "sign", "deg", "labeled_neigh",
                     "mean_neigh_s", "std_neigh_s", "same_sign_frac", "s_deg"],
}


def v1_features(G: nx.Graph, labeled: dict) -> pd.DataFrame:
    rows = []
    for v, s in labeled.items():
        neighs = list(G.neighbors(v))
        neigh_scores = [labeled.get(u, np.nan) for u in neighs]
        neigh_scores = [x for x in neigh_scores if not np.isnan(x)]
        rows.append({
            "node": v, "s": s, "sign": np.sign(s), "deg": G.degree[v],
            "labeled_neigh": len(neigh_scores),
            "mean_neigh_s": float(np.mean(neigh_scores)) if neigh_scores else 0.0,
            "std_neigh_s": float(np.std(neigh_scores)) if neigh_scores else 0.0,
            "same_sign_frac": (sum(1 for x in neigh_scores if np.sign(x) == np.sign(s))
                                / max(1, len(neigh_scores))),
            "s_deg": s * G.degree[v],
        })
    return pd.DataFrame(rows).set_index("node")


def run_v1(topic: str, epochs: int, seed: int = 42) -> dict:
    print(f"\n=== v1 feature ablation on {topic} ===")
    G = nx.read_gml(DATA_DIR / topic / "graph.gml")
    if G.is_directed():
        G = G.to_undirected()
    G = nx.convert_node_labels_to_integers(G, label_attribute="user_id")
    labels_df = pd.read_feather(DATA_DIR / topic / "allsides.feather")
    m = dict(zip(labels_df["user_id"].astype(str), labels_df["allsides_score"]))
    labeled = {n: float(m[G.nodes[n]["user_id"]])
               for n in G.nodes if str(G.nodes[n]["user_id"]) in m}

    print(f"  Training EchoGAE (identity features) for {epochs} epochs...")
    emb = EchoGAE_algorithm(G, epochs=epochs, show_progress=False)
    labeled_idx = sorted(labeled.keys())
    sign_labels = np.array([1 if labeled[v] > 0 else -1 for v in labeled_idx])
    ec = EchoChamberMeasure(emb[labeled_idx], sign_labels)
    y = np.array([ec.metric(i) for i in range(len(labeled_idx))])

    feats = v1_features(G, labeled).loc[labeled_idx]
    results = {}
    for name, cols in V1_FEATURE_SETS.items():
        X = feats[cols].values
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        r2s = []
        for tr, te in kf.split(X):
            m = Ridge(alpha=1e-3).fit(X[tr], y[tr])
            r2s.append(r2_score(y[te], m.predict(X[te])))
        results[name] = {"mean": float(np.mean(r2s)), "std": float(np.std(r2s)),
                          "n_features": len(cols)}
        print(f"  {name:20s} ({len(cols)}f): R^2 = {np.mean(r2s):+.4f} +- {np.std(r2s):.4f}")
    return results


# ----- v2: per-community on Louvain ------------------------------------

V2_FEATURE_SETS = {
    "1_size":         ["size"],
    "2_size_meandeg": ["size", "mean_deg"],
    "3_add_internal": ["size", "mean_deg", "internal_edge_frac"],
    "5_add_edges":    ["size", "mean_deg", "internal_edges", "external_edges",
                       "internal_edge_frac"],
    "7_full":         ["size", "mean_deg", "median_deg", "internal_edges",
                       "external_edges", "internal_edge_frac", "sub_clustering"],
}


def v2_features(G: nx.Graph, comm_ids: np.ndarray) -> pd.DataFrame:
    rows = []
    unique, counts = np.unique(comm_ids, return_counts=True)
    for c, sz in zip(unique, counts):
        if sz < 2:
            continue
        members = np.where(comm_ids == c)[0].tolist()
        member_set = set(members)
        degs = [G.degree[v] for v in members]
        internal = external = 0
        for v in members:
            for u in G.neighbors(v):
                if u in member_set:
                    internal += 1
                else:
                    external += 1
        internal //= 2
        sub_G = G.subgraph(members)
        rows.append({"community": int(c), "size": int(sz),
                      "mean_deg": float(np.mean(degs)),
                      "median_deg": float(np.median(degs)),
                      "internal_edges": int(internal),
                      "external_edges": int(external),
                      "internal_edge_frac": float(internal / max(internal + external, 1)),
                      "sub_clustering": float(nx.average_clustering(sub_G))})
    return pd.DataFrame(rows)


def per_community_ecs(embeddings: np.ndarray, comm_ids: np.ndarray) -> dict:
    unique, counts = np.unique(comm_ids, return_counts=True)
    ec = EchoChamberMeasure(embeddings, comm_ids, metric="euclidean")
    per = {}
    for c, sz in zip(unique, counts):
        if sz < 2:
            continue
        idx = np.where(comm_ids == c)[0].tolist()
        per[int(c)] = float(np.mean([ec.metric(int(i)) for i in idx]))
    return per


def loo_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out squared Pearson correlation."""
    if len(y) < 3 or np.std(y) < 1e-6:
        return float("nan")
    preds = np.zeros_like(y, dtype=float)
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool); mask[i] = False
        preds[i] = Ridge(alpha=1e-3).fit(X[mask], y[mask]).predict(X[i:i+1])[0]
    return float(np.corrcoef(preds, y)[0, 1] ** 2)


def run_v2(topic: str, epochs: int, seed: int = 42) -> dict:
    print(f"\n=== v2 feature ablation on {topic} ===")
    G = load_topic(topic)
    print("  Loading SBERT features (cached)...")
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

    comm_ids = louvain_communities(G, resolution=0.05, seed=seed)
    n_ns = int((np.unique(comm_ids, return_counts=True)[1] > 1).sum())
    print(f"  Louvain: {n_ns} non-singleton communities")

    print(f"  Training EchoGAE for {epochs} epochs...")
    emb = EchoGAE_algorithm(G, user_embeddings=user_embeddings, epochs=epochs,
                             show_progress=False)
    per = per_community_ecs(emb, comm_ids)
    feats = v2_features(G, comm_ids)
    y = feats["community"].map(per).values

    results = {"n_communities": n_ns}
    for name, cols in V2_FEATURE_SETS.items():
        if len(cols) >= n_ns:
            r2 = float("nan")
            note = "under-determined"
        else:
            X = feats[cols].values
            r2 = loo_r2(X, y)
            note = ""
        results[name] = {"loo_r2": r2, "n_features": len(cols), "note": note}
        print(f"  {name:20s} ({len(cols)}f): LOO R^2 = {r2:+.4f}  {note}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v1", "v2", "both"], default="both")
    ap.add_argument("--topics", nargs="+",
                    default=["gun", "abortion", "super_bowl", "sxsw"])
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    out = {}
    for t in args.topics:
        try:
            entry = {}
            if args.variant in ("v1", "both"):
                entry["v1"] = run_v1(t, epochs=args.epochs)
            if args.variant in ("v2", "both"):
                entry["v2"] = run_v2(t, epochs=args.epochs)
            out[t] = entry
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"ERROR on {t}: {e}")
    with (RESULTS_DIR / "ablation.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {RESULTS_DIR / 'ablation.json'}")
