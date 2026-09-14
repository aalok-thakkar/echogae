"""Multi-seed InfoVGAE audit with training-loss trajectories.

For each of the four Twitter topics:
  1. Train InfoVGAE-full at K seeds on the original graph.
  2. Apply the degree- and block-preserving rewiring (fixed seed).
  3. Train InfoVGAE-full at K seeds on the rewired graph.
  4. Log the reconstruction, KL, TC, and discriminator losses at every
     epoch checkpoint so posterior collapse can be identified.
  5. Report mean $\pm$ std of the community score across seeds, on
     both graphs, and the rewire delta (of means).

The purpose is to separate the reviewer's two hypotheses for the
InfoVGAE results:
  (a) posterior collapse under KL/TC weighting (score = 0 across all seeds);
  (b) training instability (score varies across seeds by more than the
      apparent rewire delta).

If per-seed std within one graph is comparable to the mean delta between
original and rewired graphs, the rewire delta is not evidence for a
polarisation-identifiability failure.
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
import torch.nn.functional as F

ECHOGAE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
DATA_DIR = ECHOGAE_DIR / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from infovgae_full import (  # noqa: E402
    InfoVGAE, Discriminator, PIDControl, permute_dims,
    load_topic, build_features, normalize_adj,
)
from rewire import rewire_preserving_blocks  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "infovgae_multi_seed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_log(G: nx.Graph, features_np: np.ndarray, epochs: int, seed: int,
                   log_every: int = 25) -> dict:
    """Train InfoVGAE-full, return final embeddings and loss trajectory."""
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
    model = InfoVGAE(features_np.shape[1], hidden=32, latent=2, adj=adj_norm)
    disc = Discriminator(2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    opt_D = torch.optim.Adam(disc.parameters(), lr=1e-3, betas=(0.5, 0.9))
    pid = PIDControl(); target_kl = 0.005
    gamma = 1e-3

    weight_mask = adj_dense_label.view(-1) == 1
    weight = torch.ones(weight_mask.size(0))
    weight[weight_mask] = pos_weight
    ones = torch.ones(n, dtype=torch.long)
    zeros = torch.zeros(n, dtype=torch.long)

    traj = []
    for epoch in range(epochs):
        model.train()
        z, mu, log_std = model.encode(features)
        A_pred = torch.sigmoid(torch.mm(z, z.t()))
        D_z = disc(z)
        recon = norm_factor * F.binary_cross_entropy(
            A_pred.view(-1), adj_dense_label.view(-1), weight=weight,
        )
        kl = 0.5 / n * (1 + 2 * log_std - mu ** 2 - torch.exp(log_std) ** 2).sum(1).mean()
        if D_z.dim() == 2 and D_z.size(1) == 2:
            tc = (D_z[:, 0] - D_z[:, 1]).mean() * gamma
        else:
            tc = torch.tensor(0.0)
        w_kl = pid.step(target_kl, kl.item())
        vae_loss = recon - w_kl * kl + tc
        opt.zero_grad(); vae_loss.backward(); opt.step()

        z2, _, _ = model.encode(features); z2 = z2.detach()
        D_z2 = disc(z2)
        z_perm = permute_dims(z2).detach()
        D_zp = disc(z_perm)
        d_loss = 0.5 * (F.cross_entropy(D_z2, zeros) + F.cross_entropy(D_zp, ones))
        opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            traj.append({"epoch": int(epoch),
                          "recon": float(recon.item()),
                          "kl": float(kl.item()),
                          "tc": float(tc.item()),
                          "d_loss": float(d_loss.item()),
                          "w_kl": float(w_kl)})

    model.eval()
    with torch.no_grad():
        z_final, _, _ = model.encode(features)
    return {"embedding": z_final.numpy(), "trajectory": traj}


def one_run(G: nx.Graph, feat: np.ndarray, labeled_idx: list, seed: int,
             epochs: int) -> dict:
    r = train_and_log(G, feat, epochs=epochs, seed=seed)
    scores = r["embedding"][:, 0] - r["embedding"][:, 1]
    per_node = scores[labeled_idx]
    return {"seed": int(seed),
            "community_score": float(np.mean(per_node)),
            "per_node_std": float(np.std(per_node)),
            "trajectory": r["trajectory"]}


def audit_topic_multi(topic: str, n_seeds: int, epochs: int,
                       rewire_seconds: float = 180.0) -> dict:
    print(f"\n=== InfoVGAE multi-seed on {topic} ===", flush=True)
    G, labeled = load_topic(topic)
    feat = build_features(G, labeled)
    labeled_idx = sorted(labeled.keys())
    print(f"  Graph: n={G.number_of_nodes()}, m={G.number_of_edges()}, "
          f"labeled={len(labeled)}", flush=True)

    block = np.zeros(G.number_of_nodes(), dtype=np.int8)
    for v, s in labeled.items():
        block[v] = 1 if s > 0 else 2
    print(f"  Rewiring (fixed seed=42)...", flush=True)
    G_rw, rw_stats = rewire_preserving_blocks(
        G, block, n_swaps=10000, seed=42, max_seconds=rewire_seconds,
    )
    print(f"    triangles {rw_stats['triangles_before']} -> {rw_stats['triangles_after']}", flush=True)

    orig_runs = []
    rw_runs = []
    for i, seed in enumerate(range(42, 42 + n_seeds)):
        print(f"  seed {seed}: original graph...", flush=True)
        orig_runs.append(one_run(G, feat, labeled_idx, seed=seed, epochs=epochs))
        print(f"    score = {orig_runs[-1]['community_score']:+.4f}", flush=True)
    for i, seed in enumerate(range(42, 42 + n_seeds)):
        print(f"  seed {seed}: rewired graph...", flush=True)
        rw_runs.append(one_run(G_rw, feat, labeled_idx, seed=seed, epochs=epochs))
        print(f"    score = {rw_runs[-1]['community_score']:+.4f}", flush=True)

    orig_scores = np.array([r["community_score"] for r in orig_runs])
    rw_scores = np.array([r["community_score"] for r in rw_runs])

    summary = {
        "topic": topic,
        "n_seeds": n_seeds, "epochs": epochs,
        "orig_mean": float(orig_scores.mean()),
        "orig_std": float(orig_scores.std(ddof=1)),
        "orig_range": [float(orig_scores.min()), float(orig_scores.max())],
        "rw_mean": float(rw_scores.mean()),
        "rw_std": float(rw_scores.std(ddof=1)),
        "rw_range": [float(rw_scores.min()), float(rw_scores.max())],
        "mean_delta": float(rw_scores.mean() - orig_scores.mean()),
        "delta_over_std": float((rw_scores.mean() - orig_scores.mean())
                                 / max(orig_scores.std(ddof=1), rw_scores.std(ddof=1), 1e-9)),
        "triangles_before": rw_stats["triangles_before"],
        "triangles_after": rw_stats["triangles_after"],
        "orig_runs": orig_runs,
        "rw_runs": rw_runs,
    }
    with (RESULTS_DIR / f"{topic}.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Original: {summary['orig_mean']:+.3f} +/- {summary['orig_std']:.3f}, "
          f"range [{summary['orig_range'][0]:+.3f}, {summary['orig_range'][1]:+.3f}]",
          flush=True)
    print(f"  Rewired : {summary['rw_mean']:+.3f} +/- {summary['rw_std']:.3f}, "
          f"range [{summary['rw_range'][0]:+.3f}, {summary['rw_range'][1]:+.3f}]",
          flush=True)
    print(f"  Delta   : {summary['mean_delta']:+.3f} "
          f"(delta / max(std) = {summary['delta_over_std']:+.2f})", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+",
                    default=["gun", "abortion", "super_bowl", "sxsw"])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--rewire_seconds", type=float, default=180.0)
    args = ap.parse_args()

    all_out = {}
    for t in args.topics:
        try:
            all_out[t] = audit_topic_multi(t, n_seeds=args.n_seeds,
                                            epochs=args.epochs,
                                            rewire_seconds=args.rewire_seconds)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"ERROR on {t}: {e}", flush=True)

    print("\n=== Multi-seed InfoVGAE summary ===", flush=True)
    print(f"{'Topic':<12} {'Orig mean +/- std':<22} {'Rewired mean +/- std':<24} "
          f"{'Delta':<9} {'D/std':<6}", flush=True)
    for t, s in all_out.items():
        print(f"  {t:<10} {s['orig_mean']:+.3f} +/- {s['orig_std']:.3f}        "
              f"{s['rw_mean']:+.3f} +/- {s['rw_std']:.3f}         "
              f"{s['mean_delta']:+.3f}   {s['delta_over_std']:+.2f}", flush=True)

    with (RESULTS_DIR / "summary.json").open("w") as f:
        json.dump(all_out, f, indent=2)
