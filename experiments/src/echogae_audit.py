"""EchoGAE audit: fit a graph-blind reproduction estimator to EchoGAE's per-node score.

If a small linear regression on 1-WL-invariant scalars (per-node degree, own label,
neighbourhood homophily, degree-weighted neighbour-label mean) reproduces EchoGAE's
per-node Echo Chamber Score to high R^2, that is evidence that whatever EchoGAE is
measuring on this dataset does not require the graph structure beyond the first-order
neighbour-composition statistics.
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


ECHOGAE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
DATA_DIR = ECHOGAE_DIR / "data"
sys.path.insert(0, str(ECHOGAE_DIR))

from src.EchoGAE import EchoGAE_algorithm
from src.echo_chamber_measure import EchoChamberMeasure


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "echogae_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_topic(topic: str) -> tuple[nx.Graph, pd.DataFrame]:
    G = nx.read_gml(DATA_DIR / topic / "graph.gml")
    if G.is_directed():
        G = G.to_undirected()
    G = nx.convert_node_labels_to_integers(G, label_attribute="user_id")
    labels = pd.read_feather(DATA_DIR / topic / "allsides.feather")
    return G, labels


def align_labels(G: nx.Graph, labels_df: pd.DataFrame) -> dict[int, float]:
    """Map node index -> allsides_score for the subset of nodes present in the label table."""
    label_map = dict(zip(labels_df["user_id"].astype(str), labels_df["allsides_score"]))
    return {n: float(label_map[G.nodes[n]["user_id"]])
            for n in G.nodes if str(G.nodes[n]["user_id"]) in label_map}


def per_node_features(G: nx.Graph, labeled: dict[int, float]) -> pd.DataFrame:
    """Graph-blind 1-WL-invariant features per labeled node.

    Features: own_label, degree, same_label_neighbours, diff_label_neighbours,
              neighbour_label_mean, neighbour_label_std, own_label * degree
    """
    rows = []
    label_arr = np.zeros(G.number_of_nodes())
    for v, s in labeled.items():
        label_arr[v] = s
    for v, s in labeled.items():
        neighs = list(G.neighbors(v))
        neigh_labels = [labeled.get(u, 0.0) for u in neighs]
        neigh_labeled = [l for l in neigh_labels if l != 0.0]
        rows.append({
            "node": v,
            "own_label": s,
            "sign_label": np.sign(s),
            "degree": G.degree[v],
            "labeled_neighbours": len(neigh_labeled),
            "mean_neighbour_label": float(np.mean(neigh_labeled)) if neigh_labeled else 0.0,
            "std_neighbour_label": float(np.std(neigh_labeled)) if neigh_labeled else 0.0,
            "same_sign_neigh_frac": (
                sum(1 for l in neigh_labeled if np.sign(l) == np.sign(s)) / len(neigh_labeled)
                if neigh_labeled else 0.5
            ),
            "label_x_degree": s * G.degree[v],
        })
    return pd.DataFrame(rows)


def run_audit(topic: str, epochs: int = 100, seed: int = 42,
              output_prefix: str = "audit") -> dict:
    print(f"=== Audit on topic: {topic} ===")
    G, labels_df = load_topic(topic)
    labeled = align_labels(G, labels_df)
    print(f"  Graph: n={G.number_of_nodes()}, m={G.number_of_edges()}")
    print(f"  Labeled subset: {len(labeled)} nodes")

    # Train EchoGAE.
    print(f"  Training EchoGAE for {epochs} epochs...")
    embeddings = EchoGAE_algorithm(G, epochs=epochs, show_progress=False)
    print(f"  Embedding shape: {embeddings.shape}")

    # Compute ECI using binary sign labels on the labeled subset only.
    label_idx = sorted(labeled.keys())
    label_arr = np.array([np.sign(labeled[v]) for v in label_idx])
    label_arr[label_arr == 0] = 1
    sub_embeddings = embeddings[label_idx]

    ec = EchoChamberMeasure(sub_embeddings, label_arr, metric="euclidean")
    overall_eci = float(ec.echo_chamber_index())
    per_node_eci = np.array([ec.metric(i) for i in range(len(label_idx))])
    print(f"  Overall ECI: {overall_eci:.4f}")
    print(f"  Per-node ECI: mean {per_node_eci.mean():.4f}, std {per_node_eci.std():.4f}")

    # Graph-blind features.
    features_df = per_node_features(G, labeled)
    features_df = features_df.set_index("node").loc[label_idx].reset_index()
    features_df["eci"] = per_node_eci

    # Fit graph-blind regression.
    feature_cols = ["own_label", "sign_label", "degree", "labeled_neighbours",
                    "mean_neighbour_label", "std_neighbour_label",
                    "same_sign_neigh_frac", "label_x_degree"]
    X = features_df[feature_cols].values
    y = features_df["eci"].values

    # 5-fold CV.
    cv_r2 = []
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X):
        model = Ridge(alpha=1e-3)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        cv_r2.append(r2_score(y[test_idx], pred))
    cv_r2_mean = float(np.mean(cv_r2))
    cv_r2_std = float(np.std(cv_r2))

    # Full-fit for coefficient interpretation.
    full_model = Ridge(alpha=1e-3).fit(X, y)
    coeffs = dict(zip(feature_cols, full_model.coef_))
    print(f"  Graph-blind reproduction: CV R^2 = {cv_r2_mean:.4f} +- {cv_r2_std:.4f}")
    print(f"  Coefficients: {coeffs}")

    # Baseline: predict ECI from only 'own_label' and 'sign_label'.
    baseline_r2 = []
    for train_idx, test_idx in kf.split(X):
        m = Ridge(alpha=1e-3).fit(X[train_idx][:, :2], y[train_idx])
        baseline_r2.append(r2_score(y[test_idx], m.predict(X[test_idx][:, :2])))
    print(f"  Own-label-only baseline: CV R^2 = {np.mean(baseline_r2):.4f}")

    out = {
        "topic": topic,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_labeled": len(labeled),
        "epochs": epochs,
        "overall_eci": overall_eci,
        "per_node_eci_mean": float(per_node_eci.mean()),
        "per_node_eci_std": float(per_node_eci.std()),
        "graph_blind_cv_r2_mean": cv_r2_mean,
        "graph_blind_cv_r2_std": cv_r2_std,
        "own_label_only_cv_r2_mean": float(np.mean(baseline_r2)),
        "coefficients": coeffs,
        "feature_cols": feature_cols,
    }
    with (RESULTS_DIR / f"{output_prefix}_{topic}.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+", default=["gun", "abortion", "super_bowl", "sxsw"])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    all_results = []
    for topic in args.topics:
        try:
            r = run_audit(topic, epochs=args.epochs)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR on {topic}: {e}")
    print("\n=== Summary ===")
    for r in all_results:
        print(f"  {r['topic']:12s}: ECI={r['overall_eci']:+.4f}, "
              f"graph-blind R^2={r['graph_blind_cv_r2_mean']:.4f}, "
              f"own-label R^2={r['own_label_only_cv_r2_mean']:.4f}")
