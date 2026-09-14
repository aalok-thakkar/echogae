"""InfoVGAE-style audit.

Reimplements the essential InfoVGAE architecture (variational GAE with a
3-layer GCN encoder, ReLU on the sampled latent, inner-product decoder, link
reconstruction loss with a PID-controlled KL weight) as a self-contained
adapter over any NetworkX graph. Runs the same graph-blind reproducibility
and rewiring tests as `echogae_audit.py` / `echogae_rewire.py`.

The score derived from InfoVGAE's 2-D latent is z[:, 0] - z[:, 1] per node
(a bipolar analogue of the paper's clustering-based per-node score).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


DATA_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score" / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rewire import rewire_preserving_blocks


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "infovgae_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Model ---------------------------------------------------------------

def glorot(in_dim: int, out_dim: int) -> nn.Parameter:
    r = (6.0 / (in_dim + out_dim)) ** 0.5
    return nn.Parameter(torch.rand(in_dim, out_dim) * 2 * r - r)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, adj: torch.Tensor, activation):
        super().__init__()
        self.W = glorot(in_dim, out_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(torch.mm(self.adj, torch.mm(x, self.W)))


class InfoVGAELite(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent: int, adj: torch.Tensor):
        super().__init__()
        self.base = GCNLayer(in_dim, hidden, adj, F.relu)
        self.mean_layer = GCNLayer(hidden, latent, adj, lambda z: z)
        self.log_std_layer = GCNLayer(hidden, latent, adj, lambda z: z)
        self.latent = latent

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.base(x)
        mu = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        eps = torch.randn(x.size(0), self.latent)
        z = F.relu(eps * torch.exp(log_std) + mu)
        return z, mu, log_std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, log_std = self.encode(x)
        return torch.sigmoid(torch.mm(z, z.t())), mu, log_std


class PIDControl:
    """PID controller for KL weight per InfoVGAE."""

    def __init__(self, kp: float = 0.001, ki: float = -0.001):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.last = 0.0

    def step(self, target: float, current: float) -> float:
        err = target - current
        self.integral += err
        w = self.last + self.kp * err + self.ki * self.integral
        w = max(w, 0.0)
        self.last = w
        return w


# --- Data ----------------------------------------------------------------

def load_topic(topic: str) -> tuple[nx.Graph, dict[int, float]]:
    G = nx.read_gml(DATA_DIR / topic / "graph.gml")
    if G.is_directed():
        G = G.to_undirected()
    G = nx.convert_node_labels_to_integers(G, label_attribute="user_id")
    labels_df = pd.read_feather(DATA_DIR / topic / "allsides.feather")
    m = dict(zip(labels_df["user_id"].astype(str), labels_df["allsides_score"]))
    labeled = {n: float(m[G.nodes[n]["user_id"]])
               for n in G.nodes if str(G.nodes[n]["user_id"]) in m}
    return G, labeled


def build_features(G: nx.Graph, labeled: dict[int, float]) -> np.ndarray:
    """Sparse identity augmented with the AllSides score for labelled nodes."""
    n = G.number_of_nodes()
    feat = np.zeros((n, 2), dtype=np.float32)
    for v, s in labeled.items():
        feat[v, 0] = float(s)
        feat[v, 1] = 1.0
    return feat


def normalize_adj(A: sp.spmatrix) -> torch.Tensor:
    A = sp.coo_matrix(A) + sp.eye(A.shape[0])
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    D = sp.diags(d_inv_sqrt)
    A_norm = D @ A @ D
    A_norm = A_norm.tocoo()
    idx = torch.LongTensor(np.vstack([A_norm.row, A_norm.col]))
    val = torch.FloatTensor(A_norm.data)
    return torch.sparse_coo_tensor(idx, val, torch.Size(A_norm.shape)).coalesce()


# --- Training ------------------------------------------------------------

def train_infovgae(G: nx.Graph, features_np: np.ndarray, epochs: int = 200,
                    hidden: int = 32, latent: int = 2, seed: int = 42,
                    lr: float = 0.01, verbose: bool = False) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, format="coo")
    A.setdiag(0)  # ensure no self-loops so A + I stays in {0, 1}
    A.eliminate_zeros()
    adj_norm = normalize_adj(A)
    adj_label = (A + sp.eye(n)).astype(np.float32)
    adj_dense_np = adj_label.toarray()
    adj_dense_np = np.clip(adj_dense_np, 0.0, 1.0)
    adj_dense_label = torch.FloatTensor(adj_dense_np)

    pos = float(A.sum())
    total = float(n * n)
    pos_weight = (total - pos) / max(pos, 1.0)
    norm_factor = total / (2 * (total - pos))

    features = torch.FloatTensor(features_np)
    model = InfoVGAELite(features_np.shape[1], hidden, latent, adj_norm)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    pid = PIDControl()
    target_kl = 0.005

    weight_mask = adj_dense_label.view(-1) == 1
    weight = torch.ones(weight_mask.size(0))
    weight[weight_mask] = pos_weight

    for epoch in range(epochs):
        model.train()
        A_pred, mu, log_std = model(features)
        recon_loss = norm_factor * F.binary_cross_entropy(
            A_pred.view(-1), adj_dense_label.view(-1), weight=weight,
        )
        kl = 0.5 / n * (1 + 2 * log_std - mu ** 2 - torch.exp(log_std) ** 2).sum(1).mean()
        w_kl = pid.step(target_kl, kl.item())
        loss = recon_loss - w_kl * kl
        optim.zero_grad()
        loss.backward()
        optim.step()
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch}: recon={recon_loss.item():.4f} kl={kl.item():.4f} w={w_kl:.4f}")

    model.eval()
    with torch.no_grad():
        z, _, _ = model.encode(features)
    return z.numpy()


# --- Audit ---------------------------------------------------------------

def per_node_score(embedding: np.ndarray) -> np.ndarray:
    """The paper's per-node bipolar score for InfoVGAE: z[:, 0] - z[:, 1]."""
    return embedding[:, 0] - embedding[:, 1]


def graph_blind_features(G: nx.Graph, labeled: dict) -> pd.DataFrame:
    rows = []
    for v, s in labeled.items():
        neighs = list(G.neighbors(v))
        neigh_scores = [labeled.get(u, np.nan) for u in neighs]
        neigh_scores = [x for x in neigh_scores if not np.isnan(x)]
        rows.append({
            "node": v,
            "s": s,
            "sign": np.sign(s),
            "deg": G.degree[v],
            "labeled_neigh": len(neigh_scores),
            "mean_neigh_s": float(np.mean(neigh_scores)) if neigh_scores else 0.0,
            "std_neigh_s": float(np.std(neigh_scores)) if neigh_scores else 0.0,
            "same_sign_frac": (
                sum(1 for x in neigh_scores if np.sign(x) == np.sign(s)) / len(neigh_scores)
                if neigh_scores else 0.5
            ),
            "s_deg": s * G.degree[v],
        })
    return pd.DataFrame(rows)


def audit_topic(topic: str, epochs: int, do_rewire: bool = True,
                 rewire_seconds: float = 60.0) -> dict:
    print(f"\n=== InfoVGAE audit on {topic} ===")
    G, labeled = load_topic(topic)
    print(f"  n={G.number_of_nodes()}, m={G.number_of_edges()}, labeled={len(labeled)}")

    features_np = build_features(G, labeled)
    print(f"  Training InfoVGAE for {epochs} epochs...")
    z = train_infovgae(G, features_np, epochs=epochs)
    scores = per_node_score(z)

    labeled_idx = sorted(labeled.keys())
    per_node = scores[labeled_idx]
    community_score = float(np.mean(per_node))
    per_node_std = float(np.std(per_node))
    print(f"  Community-level score: {community_score:+.4f} (per-node std {per_node_std:.4f})")

    # Graph-blind reproduction.
    feats = graph_blind_features(G, labeled)
    feats = feats.set_index("node").loc[labeled_idx].reset_index()
    X = feats.drop(columns=["node"]).values
    y = per_node
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s = []
    for tr, te in kf.split(X):
        r2s.append(r2_score(y[te], Ridge(alpha=1e-3).fit(X[tr], y[tr]).predict(X[te])))
    r2_mean = float(np.mean(r2s))
    r2_std = float(np.std(r2s))
    print(f"  Graph-blind reproduction R^2: {r2_mean:.4f} +- {r2_std:.4f}")

    out: dict = {
        "topic": topic,
        "epochs": epochs,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_labeled": len(labeled),
        "community_score": community_score,
        "per_node_std": per_node_std,
        "graph_blind_r2_mean": r2_mean,
        "graph_blind_r2_std": r2_std,
    }

    if do_rewire:
        print("  Rewiring...")
        block = np.zeros(G.number_of_nodes(), dtype=np.int8)
        for v, s in labeled.items():
            block[v] = 1 if s > 0 else 2
        G_rw, rw_stats = rewire_preserving_blocks(
            G, block, n_swaps=8000, seed=0, max_seconds=rewire_seconds,
        )
        print(f"    triangles {rw_stats['triangles_before']} -> {rw_stats['triangles_after']}")
        print(f"  Retraining InfoVGAE on rewired graph...")
        z2 = train_infovgae(G_rw, features_np, epochs=epochs)
        scores2 = per_node_score(z2)
        per_node2 = scores2[labeled_idx]
        community2 = float(np.mean(per_node2))
        delta = community2 - community_score
        print(f"  Community score before/after: {community_score:+.4f} -> {community2:+.4f} "
              f"(delta {delta:+.4f})")
        out["community_score_rewired"] = community2
        out["delta_score"] = delta
        out["triangles_before"] = rw_stats["triangles_before"]
        out["triangles_after"] = rw_stats["triangles_after"]
        out["clus_before"] = float(nx.average_clustering(G))
        out["clus_after"] = float(nx.average_clustering(G_rw))

    with (RESULTS_DIR / f"{topic}.json").open("w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+", default=["sxsw"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--no_rewire", action="store_true")
    ap.add_argument("--rewire_seconds", type=float, default=60.0)
    args = ap.parse_args()
    results = []
    for t in args.topics:
        results.append(audit_topic(t, epochs=args.epochs, do_rewire=not args.no_rewire,
                                    rewire_seconds=args.rewire_seconds))
    print("\n=== Summary ===")
    for r in results:
        line = f"  {r['topic']:12s}: score={r['community_score']:+.4f}, R^2={r['graph_blind_r2_mean']:.4f}"
        if "delta_score" in r:
            line += f", delta_score={r['delta_score']:+.4f}"
        print(line)
