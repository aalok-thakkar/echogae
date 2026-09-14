"""InfoVGAE audit in its authors' own setting.

Runs the released InfoVGAE pipeline (vendor/InfoVGAE: TwitterDataset,
InfoVGAETrainer, Evaluator) on its own Election and Eurovision
user-claim bipartite graphs, under its own config, at several seeds.
Reports the metrics the paper reports (user F1 and purity from a
2-means clustering of the latent, claim F1 and purity), across seeds,
for three conditions:

  orig       the released pipeline unchanged (adjacency with counts);
  binarised  the same graph with every user-claim count clipped to 1;
  rewired    a bipartite double-edge swap of the binarised graph that
             preserves every user's degree, every claim's degree, and the
             (user label, claim label) block of every edge, and
             randomises which user posted which claim subject to that.

Butterflies (4-cycles: two users sharing two claims) are the bipartite
closure statistic reported before and after the rewire.

Two patches to the release are needed under current torch: BCE targets
clipped to [0, 1] (model_trainer.py), and the seed set after the config
is applied (the config file overwrites --seed with 0).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "InfoVGAE"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "infovgae_native"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(VENDOR)
sys.path.insert(0, str(VENDOR))

from config import update_arg_with_config_name  # noqa: E402
from dataset import TwitterDataset  # noqa: E402
from model_trainer import InfoVGAETrainer  # noqa: E402
from evaluate import Evaluator  # noqa: E402


def default_args(config_name: str, num_process: int) -> Namespace:
    a = Namespace(config_name=config_name, model="InfoVGAE", epochs=500,
                  learning_rate=0.01, device="cpu", num_process=num_process,
                  seed=0, use_cuda=False, dataset=None, add_self_loop=True,
                  directed=False, data_path=None, data_json_path=None,
                  follow_path=None, use_follow=False, stopword_path=None,
                  keyword_path="N", kthreshold=5, uthreshold=3,
                  hidden1_dim=32, hidden2_dim=2, use_feature=True,
                  num_user=None, num_assertion=None, pos_weight_lambda=1.0,
                  gamma=1e-3, lr_D=1e-3, beta1_D=0.5, beta2_D=0.9,
                  output_path="./output")
    update_arg_with_config_name(a, config_name, phrase="train")
    a.output_path = str(RESULTS_DIR / f"work_{config_name}")
    os.makedirs(a.output_path, exist_ok=True)
    return a


def build_dataset(args: Namespace):
    ds = TwitterDataset(csv_path=args.data_path, keyword_path=args.keyword_path,
                        stopword_path=args.stopword_path, mode="multiply",
                        kthreshold=args.kthreshold, uthreshold=args.uthreshold,
                        num_process=args.num_process, add_self_loop=True,
                        directed=False, args=args)
    het = ds.build()
    args.num_user = ds.num_user
    args.num_assertion = ds.num_assertion
    ds.dump_label()
    return ds, het


def tweeting_block(het, nu: int, na: int) -> np.ndarray:
    return np.asarray(het[:nu, nu:nu + na].todense())


def het_from_block(B: np.ndarray, nu: int, na: int):
    n = nu + na
    H = sp.lil_matrix((n, n))
    H[:nu, nu:nu + na] = sp.lil_matrix(B)
    H[nu:nu + na, :nu] = sp.lil_matrix(B.T)
    return H


def butterflies(B: np.ndarray) -> int:
    M = B @ B.T
    np.fill_diagonal(M, 0)
    return int((M * (M - 1) / 2).sum() / 2)


def rewire_bipartite(B: np.ndarray, ulab: np.ndarray, clab: np.ndarray,
                     n_swaps: int, rng: np.random.Generator,
                     max_seconds: float) -> tuple[np.ndarray, dict]:
    edges = [(int(u), int(c)) for u, c in zip(*np.nonzero(B))]
    exists = set(edges)
    m = len(edges)
    accepted = proposed = 0
    start = time.time()
    while accepted < n_swaps and time.time() - start < max_seconds:
        proposed += 1
        i, j = int(rng.integers(m)), int(rng.integers(m))
        if i == j:
            continue
        u1, c1 = edges[i]
        u2, c2 = edges[j]
        if u1 == u2 or c1 == c2:
            continue
        if ulab[u1] != ulab[u2] or clab[c1] != clab[c2]:
            continue
        if (u1, c2) in exists or (u2, c1) in exists:
            continue
        exists.discard((u1, c1)); exists.discard((u2, c2))
        exists.add((u1, c2)); exists.add((u2, c1))
        edges[i] = (u1, c2); edges[j] = (u2, c1)
        accepted += 1
    B2 = np.zeros_like(B)
    for u, c in edges:
        B2[u, c] = 1
    return B2, {"accepted": accepted, "proposed": proposed,
                "elapsed_seconds": time.time() - start, "n_edges": m}


def train_eval(args: Namespace, het, ds, seed: int) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    args.seed = seed
    feature = sp.diags([1.0], shape=(ds.num_nodes, ds.num_nodes))
    args.input_dim = ds.num_nodes
    trainer = InfoVGAETrainer(het, feature, args, ds)
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.train()
    ev = Evaluator()
    ev.init_from_value(trainer.result_embedding, ds.user_label.copy(),
                       ds.asser_label.copy(), ds.name_list.copy(),
                       ds.asserlist.copy(), output_dir=args.output_path)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.run_clustering()
        _, user_f1, asser_f1 = ev.numerical_evaluate()
    um = ev.user_label != 0
    user_purity = ev.purity_score(ev.user_label[um], ev.user_clustering_pred[um])
    am = ev.asser_label != 1000
    asser_purity = ev.purity_score(ev.asser_label[am], ev.asser_clustering_pred[am])
    return {"seed": seed, "user_f1": float(user_f1),
            "user_purity": float(user_purity),
            "claim_f1": float(asser_f1), "claim_purity": float(asser_purity)}


def summarise(runs: list[dict]) -> dict:
    out = {}
    for k in ("user_f1", "user_purity", "claim_f1", "claim_purity"):
        v = np.array([r[k] for r in runs])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                  "min": float(v.min()), "max": float(v.max())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="InfoVGAE_election_3D")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--start_seed", type=int, default=42)
    ap.add_argument("--swaps_per_edge", type=float, default=10.0)
    ap.add_argument("--max_seconds", type=float, default=300.0)
    ap.add_argument("--num_process", type=int, default=4)
    args = ap.parse_args()

    a = default_args(args.config, args.num_process)
    with contextlib.redirect_stdout(io.StringIO()):
        ds, het = build_dataset(a)
    nu, na = ds.num_user, ds.num_assertion
    B = tweeting_block(het, nu, na)
    n_multi = int((B > 1).sum())
    Bb = (B > 0).astype(np.float64)
    ulab = np.asarray(ds.user_label)
    clab = np.asarray(ds.asser_label)
    print(f"{args.config}: users={nu}, claims={na}, user-claim edges="
          f"{int(Bb.sum())}, entries with count>1: {n_multi}; labelled "
          f"users={int((ulab != 0).sum())}, labelled claims="
          f"{int((clab != 0).sum())}", flush=True)
    bf_orig = butterflies(Bb)
    print(f"butterflies (binarised): {bf_orig}", flush=True)

    het_bin = het_from_block(Bb, nu, na)
    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    runs = {"orig": [], "binarised": [], "rewired": []}
    rewire_stats = []
    for s in seeds:
        print(f"\n=== seed {s} ===", flush=True)
        r = train_eval(a, het, ds, s); runs["orig"].append(r)
        print(f"  orig      user_f1={r['user_f1']:.4f} purity={r['user_purity']:.4f}", flush=True)
        r = train_eval(a, het_bin, ds, s); runs["binarised"].append(r)
        print(f"  binarised user_f1={r['user_f1']:.4f} purity={r['user_purity']:.4f}", flush=True)
        rng = np.random.default_rng(s)
        B2, st = rewire_bipartite(Bb, ulab, clab,
                                  n_swaps=int(args.swaps_per_edge * Bb.sum()),
                                  rng=rng, max_seconds=args.max_seconds)
        st["butterflies_before"] = bf_orig
        st["butterflies_after"] = butterflies(B2)
        rewire_stats.append(st)
        print(f"  rewire: {st['accepted']}/{st['proposed']} swaps, butterflies "
              f"{bf_orig} -> {st['butterflies_after']}", flush=True)
        r = train_eval(a, het_from_block(B2, nu, na), ds, s); runs["rewired"].append(r)
        print(f"  rewired   user_f1={r['user_f1']:.4f} purity={r['user_purity']:.4f}", flush=True)

    summary = {k: summarise(v) for k, v in runs.items()}
    print("\n=== InfoVGAE native summary ===", flush=True)
    for cond, s in summary.items():
        print(f"{cond:<10} user_f1 {s['user_f1']['mean']:.4f} +/- {s['user_f1']['std']:.4f} "
              f"[{s['user_f1']['min']:.4f}, {s['user_f1']['max']:.4f}]   "
              f"purity {s['user_purity']['mean']:.4f} +/- {s['user_purity']['std']:.4f}",
              flush=True)
    out = {"config": args.config, "n_users": nu, "n_claims": na,
           "n_edges": int(Bb.sum()), "n_multi_entries": n_multi,
           "epochs": a.epochs, "seeds": seeds,
           "summary": summary, "runs": runs, "rewire_stats": rewire_stats}
    out_path = RESULTS_DIR / f"{args.config}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {out_path}", flush=True)


if __name__ == "__main__":
    main()
