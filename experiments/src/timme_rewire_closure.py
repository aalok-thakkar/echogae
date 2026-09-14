"""Closure-removing rewire of TIMME's binarised relation graphs.

The label-composition shuffle in timme_rewire.py operates on weighted
records; because TIMME binarises each relation, that shuffle disperses
parallel records and raises closure. This script tests the EchoGAE-style
null instead: binarise each relation first (which is what TIMME sees),
then apply directed double-edge swaps that preserve per-vertex in- and
out-degree and the label-block pair of every edge, accepting only swaps
that reduce the undirected triangle count on the relation. Degrees and
neighbourhood label composition are then preserved exactly on the graph
TIMME consumes, and closure is reduced as far as the swap budget allows.

Reports, per seed: TIMME test accuracy on the original and rewired
data, and per-relation unique edges, triangles, and average local
clustering before and after.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timme_rewire import (  # noqa: E402
    TIMME_DATA, RELATIONS, load_labels_for_blocks, rewire_directed_blocks,
    run_timme,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "timme_rewire"


def closure_stats(dataset: str, ids: np.ndarray) -> dict:
    """Unique directed edges, triangles, and average local clustering on
    the undirected simple projection of each relation, plus the union."""
    idx = {int(t): i for i, t in enumerate(ids)}
    n = len(ids)
    U = np.zeros((n, n), dtype=np.float32)
    out = {}

    def one(A):
        d = A.sum(1)
        tri_v = ((A @ A) * A).sum(1)
        wedges = d * (d - 1)
        m = wedges > 0
        return {"triangles": int(tri_v.sum() / 6),
                "avg_clustering": float(np.mean(
                    np.where(m, tri_v / np.maximum(wedges, 1), 0)))}

    for rel in RELATIONS:
        df = pd.read_csv(TIMME_DATA / dataset / rel, sep="\t")
        a = df.iloc[:, 0].map(idx).values
        b = df.iloc[:, 1].map(idx).values
        A = np.zeros((n, n), dtype=np.float32)
        A[a, b] = 1
        n_unique = int(A.sum())
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0)
        U = np.maximum(U, A)
        out[rel] = {"unique_directed_edges": n_unique, **one(A)}
    out["union"] = one(U)
    return out


def build_closure_rewired(src_dataset: str, dst_dataset: str, seed: int,
                          swaps_per_edge: float, max_seconds: float) -> dict:
    src = TIMME_DATA / src_dataset
    dst = TIMME_DATA / dst_dataset
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in ("dict.csv", "all_twitter_ids.csv", "features.npz",
              "tweet_features.npz", "readme.md"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    block = load_labels_for_blocks(src_dataset)
    rng = np.random.default_rng(seed)
    per_rel = {}
    for rel in RELATIONS:
        df = pd.read_csv(src / rel, sep="\t")
        cols = list(df.columns)
        # Binarise: one record per distinct directed pair, count 1.
        pairs = sorted({(int(a), int(b)) for a, b in
                        zip(df[cols[0]].values, df[cols[1]].values)})
        n_target = int(swaps_per_edge * len(pairs))
        new_edges, stats = rewire_directed_blocks(
            pairs, block, n_swaps=n_target, rng=rng,
            prefer_triangle_destroy=True, max_seconds=max_seconds)
        rows = [(a, b, 1) for (a, b) in new_edges]
        pd.DataFrame(rows, columns=cols[:3]).to_csv(dst / rel, sep="\t",
                                                    index=False)
        per_rel[rel] = {"n_unique_edges": len(pairs), **stats}
        print(f"  {rel}: {stats['accepted']}/{stats['proposed']} swaps "
              f"({stats['elapsed_seconds']:.1f}s)", flush=True)
    return per_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="PureP")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--start_seed", type=int, default=42)
    ap.add_argument("--swaps_per_edge", type=float, default=10.0)
    ap.add_argument("--max_seconds", type=float, default=600.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--reuse_orig", type=str, default=None,
                    help="JSON from timme_multi_seed with per-seed original "
                         "accuracies, to skip re-running the original.")
    args = ap.parse_args()

    ids = pd.read_csv(TIMME_DATA / args.dataset / "all_twitter_ids.csv",
                      sep="\t")["twitter_id"].values
    orig_lookup = {}
    if args.reuse_orig:
        d = json.load(open(args.reuse_orig))
        orig_lookup = {r["seed"]: r for r in d["per_seed"]}

    print(f"closure stats on original {args.dataset} ...", flush=True)
    stats_orig = closure_stats(args.dataset, ids)
    print(json.dumps(stats_orig), flush=True)

    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    per_seed = []
    for s in seeds:
        print(f"\n=== seed {s} ===", flush=True)
        dst = f"{args.dataset}_closure_seed{s}"
        rw = build_closure_rewired(args.dataset, dst, seed=s,
                                   swaps_per_edge=args.swaps_per_edge,
                                   max_seconds=args.max_seconds)
        st = closure_stats(dst, ids)
        print("  closure after:", json.dumps({k: (v["triangles"],
                                               v["avg_clustering"])
                                          for k, v in st.items()}),
              flush=True)
        if s in orig_lookup:
            r_orig = {"seed": s, "dataset": args.dataset,
                      "accuracy": orig_lookup[s]["accuracy"],
                      "f1": orig_lookup[s].get("f1_score", float("nan"))}
        else:
            r_orig = run_timme(args.dataset, s, epochs=args.epochs)
        r_rw = run_timme(dst, s, epochs=args.epochs)
        print(f"  orig    : acc={r_orig['accuracy']:.4f}", flush=True)
        print(f"  rewired : acc={r_rw['accuracy']:.4f}", flush=True)
        per_seed.append({"seed": s, "orig": r_orig, "rewired": r_rw,
                         "rewire_stats": rw, "closure_after": st})

    acc_o = np.array([r["orig"]["accuracy"] for r in per_seed])
    acc_r = np.array([r["rewired"]["accuracy"] for r in per_seed])
    dl = acc_r - acc_o
    print("\n=== TIMME closure-rewire summary ===", flush=True)
    print(f"dataset  = {args.dataset}", flush=True)
    print(f"acc orig = {acc_o.mean():+.4f} +/- {acc_o.std(ddof=1):.4f}", flush=True)
    print(f"acc rw   = {acc_r.mean():+.4f} +/- {acc_r.std(ddof=1):.4f}", flush=True)
    print(f"delta    = {dl.mean():+.4f} +/- {dl.std(ddof=1):.4f}, "
          f"range [{dl.min():+.4f}, {dl.max():+.4f}]", flush=True)
    out = {"dataset": args.dataset, "n_seeds": len(seeds),
           "swaps_per_edge": args.swaps_per_edge,
           "max_seconds": args.max_seconds,
           "closure_orig": stats_orig,
           "acc_orig_mean": float(acc_o.mean()),
           "acc_orig_std": float(acc_o.std(ddof=1)),
           "acc_rewired_mean": float(acc_r.mean()),
           "acc_rewired_std": float(acc_r.std(ddof=1)),
           "delta_mean": float(dl.mean()), "delta_std": float(dl.std(ddof=1)),
           "delta_range": [float(dl.min()), float(dl.max())],
           "per_seed": per_seed}
    out_path = RESULTS_DIR / f"closure_rewire_{args.dataset}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {out_path}", flush=True)


if __name__ == "__main__":
    main()
