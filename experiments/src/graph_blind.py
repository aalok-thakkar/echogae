"""Graph-blind reproducibility estimator, per Prop 4 (theory/characterization.tex).

The theorem: any MPNN estimator's output on a featured graph factors through the
per-vertex tuple (feature, degree, sorted neighbour-feature multiset). Given a trained
MPNN estimator M and a training graph distribution D, we can fit a graph-blind statistic
M^star that approximates M(G, x) uniformly on D.

Two variants provided:

1. Naive: return the edge-homophily scalar. Zero training needed. Tests the hypothesis
   'the GNN is measuring homophily'.

2. Fitted: train a linear (or low-degree polynomial) regression from a small feature
   vector of 1-WL-invariant scalars to the MPNN's output on the training distribution.
   Predict the MPNN's output on held-out graphs without touching the graph structure.

The audit result: correlation between M(G, x) and M^star(G, x) on held-out graphs. High
correlation is evidence that M is measuring a graph-blind homophily-shaped functional.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

import networkx as nx
import numpy as np
from sklearn.linear_model import Ridge


# -------- 1-WL-invariant scalar features on a featured graph --------

def edge_homophily(G: nx.Graph, feat: np.ndarray) -> float:
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    same = sum(1 for u, v in G.edges if feat[u] == feat[v])
    return same / m


def assortativity(G: nx.Graph, feat: np.ndarray) -> float:
    """Newman attribute assortativity coefficient over the feature attribute."""
    if G.number_of_edges() == 0:
        return 0.0
    d = {int(v): int(feat[v]) for v in G.nodes}
    nx.set_node_attributes(G, d, name="_feat")
    try:
        return float(nx.attribute_assortativity_coefficient(G, "_feat"))
    finally:
        for v in G.nodes:
            del G.nodes[v]["_feat"]


def mean_degree(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    return 2 * G.number_of_edges() / G.number_of_nodes()


def per_feature_degree(G: nx.Graph, feat: np.ndarray, target_feature: int) -> float:
    idx = [v for v in G.nodes if int(feat[v]) == target_feature]
    if not idx:
        return 0.0
    return float(np.mean([G.degree[v] for v in idx]))


def neighbour_feature_entropy(G: nx.Graph, feat: np.ndarray) -> float:
    """Average Shannon entropy of the neighbour-feature multiset."""
    ent_sum = 0.0
    counted = 0
    for v in G.nodes:
        if G.degree[v] == 0:
            continue
        c = Counter(int(feat[u]) for u in G.neighbors(v))
        total = sum(c.values())
        h = 0.0
        for k in c:
            p = c[k] / total
            if p > 0:
                h -= p * np.log(p)
        ent_sum += h
        counted += 1
    return ent_sum / max(counted, 1)


def degree_feature_covariance(G: nx.Graph, feat: np.ndarray) -> float:
    degs = np.array([G.degree[v] for v in G.nodes], dtype=float)
    fs = np.array(feat, dtype=float)
    if degs.std() == 0 or fs.std() == 0:
        return 0.0
    return float(np.corrcoef(degs, fs)[0, 1])


# The default 1-WL invariant feature vector. Ordered.
DEFAULT_FEATURES: list[tuple[str, Callable[[nx.Graph, np.ndarray], float]]] = [
    ("edge_homophily", edge_homophily),
    ("assortativity", assortativity),
    ("mean_degree", lambda G, f: mean_degree(G)),
    ("mean_deg_pos", lambda G, f: per_feature_degree(G, f, +1)),
    ("mean_deg_neg", lambda G, f: per_feature_degree(G, f, -1)),
    ("neigh_feat_entropy", neighbour_feature_entropy),
    ("deg_feat_cov", degree_feature_covariance),
]


def compute_feature_vector(G: nx.Graph, feat: np.ndarray,
                            features=DEFAULT_FEATURES) -> np.ndarray:
    return np.array([fn(G, feat) for _, fn in features])


# -------- Graph-blind estimators --------

@dataclass
class HomophilyEstimator:
    """M^star(G, x) := edge_homophily(G, x). Zero training."""

    def predict(self, G: nx.Graph, feat: np.ndarray) -> float:
        return edge_homophily(G, feat)


@dataclass
class FittedGraphBlindEstimator:
    """Ridge regression from a fixed vector of 1-WL-invariant scalars to a target MPNN output."""
    features: list = None
    alpha: float = 1e-3
    _model: Ridge | None = None
    _feature_names: list[str] | None = None

    def __post_init__(self):
        if self.features is None:
            self.features = DEFAULT_FEATURES
        self._feature_names = [name for name, _ in self.features]

    def fit(self, graphs: Sequence[tuple[nx.Graph, np.ndarray]],
            targets: Sequence[float]):
        X = np.stack([compute_feature_vector(G, f, self.features) for G, f in graphs])
        y = np.asarray(targets, dtype=float)
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, y)
        return self

    def predict(self, G: nx.Graph, feat: np.ndarray) -> float:
        if self._model is None:
            raise RuntimeError("fit() first")
        x = compute_feature_vector(G, feat, self.features).reshape(1, -1)
        return float(self._model.predict(x)[0])

    def coefficients(self) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("fit() first")
        return dict(zip(self._feature_names, self._model.coef_))


# -------- Audit metrics --------

def audit_correlation(mpnn_outputs: Sequence[float], blind_outputs: Sequence[float]) -> dict:
    a = np.asarray(mpnn_outputs, dtype=float)
    b = np.asarray(blind_outputs, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return {"pearson": 0.0, "spearman": 0.0, "l1_err": float(np.mean(np.abs(a - b))),
                "l_inf_err": float(np.max(np.abs(a - b)))}
    from scipy.stats import spearmanr

    return {
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(spearmanr(a, b).statistic),
        "l1_err": float(np.mean(np.abs(a - b))),
        "l_inf_err": float(np.max(np.abs(a - b))),
    }


if __name__ == "__main__":
    # Smoke test on synthetic graphs.
    rng = np.random.default_rng(0)
    graphs = []
    targets = []
    for _ in range(50):
        G = nx.erdos_renyi_graph(50, 0.1, seed=int(rng.integers(1_000)))
        feat = rng.choice([-1, 1], size=50)
        graphs.append((G, feat))
        # Synthetic 'MPNN target' = homophily + small noise
        targets.append(edge_homophily(G, feat) + 0.02 * rng.normal())

    est = FittedGraphBlindEstimator().fit(graphs, targets)
    print("Fitted coefficients:")
    for k, v in est.coefficients().items():
        print(f"  {k:22s} = {v:+.4f}")

    preds = [est.predict(G, f) for G, f in graphs]
    print("\nAudit correlation (in-sample):")
    print(" ", audit_correlation(targets, preds))
