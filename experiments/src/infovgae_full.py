"""InfoVGAE-full audit.

An adapter that runs the actual InfoVGAE training loop from
https://github.com/jinningli/InfoVGAE on an arbitrary NetworkX graph
without the paper's bill/tweet dataset-specific preprocessing.

Architectural changes from InfoVGAE-lite:
  - Adds the discriminator on the latent z.
  - Adds the total-correlation loss term with gamma weighting.
  - Adds the PID-controlled KL weight (matches PID.py from the release).
The encoder/decoder/base loss remain the same.

This audit runs the same three experiments as the EchoGAE v2 audit and
the InfoVGAE-lite audit for comparison.
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


ECHOGAE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
DATA_DIR = ECHOGAE_DIR / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rewire import rewire_preserving_blocks, connectivity_report


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "infovgae_full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Model (matches InfoVGAE/model.py architecture) ---------------------

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


class InfoVGAE(nn.Module):
    """Matches InfoVGAE/model.py: 3 GCN layers, ReLU-latent, inner-product
    decoder. Unipartite case (no user/asser masks)."""

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


class Discriminator(nn.Module):
    """Matches InfoVGAE/model.py Discriminator: 3-layer MLP with 1000-unit
    hidden layers, LeakyReLU. Output dim 2 for logistic-CE against zeros/ones."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000), nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000), nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze()


class PIDControl:
    """Matches InfoVGAE/PID.py."""

    def __init__(self, kp: float = 0.001, ki: float = -0.001):
        self.kp = kp; self.ki = ki
        self.integral = 0.0; self.last = 0.0

    def step(self, target: float, current: float) -> float:
        err = target - current
        self.integral += err
        w = self.last + self.kp * err + self.ki * self.integral
        w = max(w, 0.0)
        self.last = w
        return w


def permute_dims(z: torch.Tensor) -> torch.Tensor:
    """Random per-dimension permutation, per InfoVGAE/model_trainer.py."""
    B, D = z.size()
    perm_z = []
    for j in range(D):
        perm = torch.randperm(B)
        perm_z.append(z[:, j:j+1][perm])
    return torch.cat(perm_z, 1)


# --- Data ---------------------------------------------------------------

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


def build_features(G: nx.Graph, labeled: dict) -> np.ndarray:
    n = G.number_of_nodes()
    feat = np.zeros((n, 2), dtype=np.float32)
    for v, s in labeled.items():
        feat[v, 0] = float(s); feat[v, 1] = 1.0
    return feat


def normalize_adj(A: sp.spmatrix) -> torch.Tensor:
    A = sp.coo_matrix(A) + sp.eye(A.shape[0])
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(d, dtype=float)
    mask = d > 0
    d_inv_sqrt[mask] = np.power(d[mask], -0.5)
    D = sp.diags(d_inv_sqrt)
    A_norm = D @ A @ D
    A_norm = A_norm.tocoo()
    idx = torch.LongTensor(np.vstack([A_norm.row, A_norm.col]))
    val = torch.FloatTensor(A_norm.data)
    return torch.sparse_coo_tensor(idx, val, torch.Size(A_norm.shape)).coalesce()


# --- Training (matches InfoVGAE/model_trainer.py loop) -----------------

def train_infovgae_full(G: nx.Graph, features_np: np.ndarray, epochs: int = 500,
                          hidden: int = 32, latent: int = 2, seed: int = 42,
                          lr: float = 0.01, lr_D: float = 1e-3,
                          gamma: float = 1e-3, verbose: bool = False) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, format="coo")
    A.setdiag(0); A.eliminate_zeros()
    adj_norm = normalize_adj(A)
    adj_label = (A + sp.eye(n)).astype(np.float32)
    adj_dense_np = np.clip(adj_label.toarray(), 0.0, 1.0)
    adj_dense_label = torch.FloatTensor(adj_dense_np)

    pos = float(A.sum()); total = float(n * n)
    pos_weight = (total - pos) / max(pos, 1.0)
    norm_factor = total / (2 * (total - pos))

    features = torch.FloatTensor(features_np)
    model = InfoVGAE(features_np.shape[1], hidden, latent, adj_norm)
    disc = Discriminator(latent)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt_D = torch.optim.Adam(disc.parameters(), lr=lr_D, betas=(0.5, 0.9))
    pid = PIDControl(); target_kl = 0.005

    weight_mask = adj_dense_label.view(-1) == 1
    weight = torch.ones(weight_mask.size(0))
    weight[weight_mask] = pos_weight

    ones = torch.ones(n, dtype=torch.long)
    zeros = torch.zeros(n, dtype=torch.long)

    for epoch in range(epochs):
        # VAE step.
        model.train()
        z, mu, log_std = model.encode(features)
        A_pred = torch.sigmoid(torch.mm(z, z.t()))
        D_z = disc(z)

        recon_loss = norm_factor * F.binary_cross_entropy(
            A_pred.view(-1), adj_dense_label.view(-1), weight=weight,
        )
        kl = 0.5 / n * (1 + 2 * log_std - mu ** 2 - torch.exp(log_std) ** 2).sum(1).mean()
        # TC term: encourage disc to fail on true z (i.e., pretend independent).
        if D_z.dim() == 2 and D_z.size(1) == 2:
            tc_loss = (D_z[:, 0] - D_z[:, 1]).mean() * gamma
        else:
            tc_loss = torch.tensor(0.0)
        w_kl = pid.step(target_kl, kl.item())
        vae_loss = recon_loss - w_kl * kl + tc_loss
        opt.zero_grad(); vae_loss.backward(); opt.step()

        # Discriminator step.
        z2, _, _ = model.encode(features)
        z2 = z2.detach()
        D_z2 = disc(z2)
        z_perm = permute_dims(z2).detach()
        D_zp = disc(z_perm)
        d_loss = 0.5 * (F.cross_entropy(D_z2, zeros) + F.cross_entropy(D_zp, ones))
        opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        if verbose and epoch % 50 == 0:
            print(f"    epoch {epoch}: recon={recon_loss.item():.4f} kl={kl.item():.4f} "
                  f"tc={tc_loss.item():.4f} d={d_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z, _, _ = model.encode(features)
    return z.numpy()


# --- Audit --------------------------------------------------------------

def per_node_score(z: np.ndarray) -> np.ndarray:
    return z[:, 0] - z[:, 1]


def audit_topic(topic: str, epochs: int, seed: int = 42,
                 rewire_seconds: float = 120.0) -> dict:
    print(f"\n=== InfoVGAE-full audit on {topic} ===")
    G, labeled = load_topic(topic)
    print(f"  Graph: n={G.number_of_nodes()}, m={G.number_of_edges()}, "
          f"labeled={len(labeled)}")

    feat = build_features(G, labeled)
    print(f"  Training InfoVGAE-full for {epochs} epochs...")
    z = train_infovgae_full(G, feat, epochs=epochs, seed=seed)
    scores = per_node_score(z)
    labeled_idx = sorted(labeled.keys())
    per_node = scores[labeled_idx]
    community = float(np.mean(per_node))
    std = float(np.std(per_node))
    print(f"  Community score: {community:+.4f} (std {std:.4f})")

    # Rewiring.
    block = np.zeros(G.number_of_nodes(), dtype=np.int8)
    for v, s in labeled.items():
        block[v] = 1 if s > 0 else 2
    G_rw, rw_stats = rewire_preserving_blocks(
        G, block, n_swaps=10000, seed=seed, max_seconds=rewire_seconds,
    )
    conn = connectivity_report(G, G_rw, labelled=set(labeled.keys()))
    print(f"  Rewire: triangles {rw_stats['triangles_before']} -> "
          f"{rw_stats['triangles_after']}, components "
          f"{conn['n_components_before']} -> {conn['n_components_after']}, "
          f"labeled in LCC after: {conn['labelled_in_lcc_after']}/{conn['labelled_total']}")
    z2 = train_infovgae_full(G_rw, feat, epochs=epochs, seed=seed)
    scores2 = per_node_score(z2)
    per_node2 = scores2[labeled_idx]
    community2 = float(np.mean(per_node2))
    delta = community2 - community
    print(f"  Community score after: {community2:+.4f} (delta {delta:+.4f})")

    out = {
        "topic": topic, "epochs": epochs,
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "n_labeled": len(labeled),
        "community_score": community, "per_node_std": std,
        "community_score_rewired": community2, "delta_score": delta,
        "triangles_before": rw_stats["triangles_before"],
        "triangles_after": rw_stats["triangles_after"],
        "connectivity": conn,
    }
    with (RESULTS_DIR / f"{topic}.json").open("w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+", default=["sxsw"])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--rewire_seconds", type=float, default=120.0)
    args = ap.parse_args()
    results = []
    for t in args.topics:
        try:
            results.append(audit_topic(t, epochs=args.epochs,
                                         rewire_seconds=args.rewire_seconds))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"ERROR on {t}: {e}")

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['topic']:12s}: score={r['community_score']:+.4f}, "
              f"std={r['per_node_std']:.4f}, delta={r['delta_score']:+.4f}")
