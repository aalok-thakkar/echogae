"""Rewire audit on TIMME's PureP relations.

For each of the five directed relations (retweet, mention, friend, reply,
favorite) we perform a degree- and block-preserving edge swap on the
multigraph. Blocks are ideology labels (D, R, unlabeled). Each accepted
swap picks two directed edges (a, b), (c, d) with matching block pairs
for (a, b) and (c, d) and rewires to (a, d), (c, b) provided the new
edges also match the same category and are not already present as
same-source-same-target duplicates in this relation. Directed in- and
out-degrees of every vertex are preserved exactly.

The rewired dataset is written under
../data/PureP_rewired_<seed>/ with all non-relation files copied over
verbatim (dict.csv, all_twitter_ids.csv, features.npz,
tweet_features.npz) and the five relation CSVs replaced by their
rewired counterparts. TIMME is then invoked on that dataset directly.

Report: for each seed, the delta between TIMME test accuracy on PureP
and on PureP_rewired.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


TIMME_CODE = Path(__file__).resolve().parents[1] / "vendor" / "TIMME" / "code"
TIMME_DATA = Path(__file__).resolve().parents[1] / "vendor" / "TIMME" / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "timme_rewire"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RELATIONS = ["retweet_list.csv", "mention_list.csv", "friend_list.csv",
              "reply_list.csv", "favorite_list.csv"]


def load_labels_for_blocks(dataset: str) -> dict:
    """Return dict {twitter_id: block_id} where block is 0/1 for D/R,
    2 for unlabeled. Uses dict.csv + additional labels, ignoring "I"."""
    data_dir = TIMME_DATA / dataset
    label_df = pd.read_csv(data_dir / "dict.csv", sep="\t")
    add_df = pd.read_csv(TIMME_DATA / "additional_labels"
                          / "new_dict_cleaned.csv", sep="\t")
    parts = {}
    for df in (label_df, add_df):
        for tid, party in zip(df["twitter_id"].values, df["party"].values):
            if party in ("D", "R"):
                parts[int(tid)] = 0 if party == "D" else 1
    return parts


def rewire_directed_blocks(edges: list[tuple[int, int]],
                             block: dict[int, int],
                             n_swaps: int,
                             rng: np.random.Generator,
                             prefer_triangle_destroy: bool,
                             max_seconds: float) -> tuple[list[tuple[int, int]], dict]:
    """Directed double-edge swap preserving per-vertex in/out degree and
    the block category of each edge. edges is a list of (a, b) directed
    pairs (may contain duplicates for multi-edges). Blocks: any vertex
    not in the dict maps to sentinel 2 (unlabeled)."""
    def bl(v: int) -> int:
        return block.get(v, 2)

    def cat(a: int, b: int) -> tuple[int, int]:
        return (bl(a), bl(b))

    # Adjacency for existence check on directed edges: set of (a,b) with
    # multiplicity handled via a Counter.
    from collections import Counter
    mult = Counter(edges)
    # Compact edge list for swap sampling: unique pairs
    edges = list(edges)
    n_edges = len(edges)

    # Triangle destruction: for directed graphs, we approximate by taking
    # the underlying simple graph and counting undirected triangles that
    # touch a swapped edge. Cheap proxy.
    adj_undir = defaultdict(set)
    for a, b in edges:
        adj_undir[a].add(b); adj_undir[b].add(a)

    accepted = 0
    proposed = 0
    dropped_multi = 0
    dropped_cat = 0
    dropped_no_decrease = 0
    start = time.time()

    while accepted < n_swaps and time.time() - start < max_seconds:
        proposed += 1
        i = int(rng.integers(n_edges))
        j = int(rng.integers(n_edges))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if len({a, b, c, d}) < 4:
            continue
        if cat(a, b) != cat(c, d):
            dropped_cat += 1
            continue
        # Directed swap: (a,b), (c,d) -> (a,d), (c,b). Preserves per-vertex
        # in/out degree exactly. Check new edges' categories.
        if cat(a, d) != cat(a, b):
            dropped_cat += 1
            continue
        if cat(c, b) != cat(a, b):
            dropped_cat += 1
            continue
        # Multigraph: check that the target edges do not already exist
        # in this relation.
        if mult[(a, d)] > 0 or mult[(c, b)] > 0:
            dropped_multi += 1
            continue

        if prefer_triangle_destroy:
            # Undirected triangle count on the four edges before/after.
            def tris(u: int, v: int) -> int:
                return len(adj_undir[u] & adj_undir[v])
            t_before = tris(a, b) + tris(c, d)
            # After the swap: undirected adjacency changes ambiguously if
            # there are multi-edges. Use a conservative estimate: pretend
            # a-b and c-d are removed from adj, a-d and c-b added, then
            # count. This overestimates destruction but suffices as a
            # ranking heuristic.
            common_ad = (adj_undir[a] - {b}) & (adj_undir[d] - {c})
            common_cb = (adj_undir[c] - {d}) & (adj_undir[b] - {a})
            t_after = len(common_ad) + len(common_cb)
            if t_after >= t_before:
                dropped_no_decrease += 1
                continue

        mult[(a, b)] -= 1; mult[(c, d)] -= 1
        mult[(a, d)] += 1; mult[(c, b)] += 1
        edges[i] = (a, d); edges[j] = (c, b)
        # Update undirected adjacency using multiplicity-aware toggles.
        if mult[(a, b)] == 0 and mult[(b, a)] == 0:
            adj_undir[a].discard(b); adj_undir[b].discard(a)
        if mult[(c, d)] == 0 and mult[(d, c)] == 0:
            adj_undir[c].discard(d); adj_undir[d].discard(c)
        adj_undir[a].add(d); adj_undir[d].add(a)
        adj_undir[c].add(b); adj_undir[b].add(c)
        accepted += 1

    stats = {"accepted": accepted, "proposed": proposed,
              "dropped_cat_mismatch": dropped_cat,
              "dropped_multi_edge": dropped_multi,
              "dropped_no_triangle_decrease": dropped_no_decrease,
              "elapsed_seconds": time.time() - start}
    return edges, stats


def build_rewired_dataset(src_dataset: str, dst_dataset: str,
                            seed: int, n_swaps: int, max_seconds: float,
                            prefer_triangle_destroy: bool) -> dict:
    src = TIMME_DATA / src_dataset
    dst = TIMME_DATA / dst_dataset
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Copy invariants: labels, id list, features
    for f in ("dict.csv", "all_twitter_ids.csv", "features.npz",
                "tweet_features.npz", "readme.md"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    block = load_labels_for_blocks(src_dataset)
    per_rel_stats = {}
    rng = np.random.default_rng(seed)
    for rel in RELATIONS:
        df = pd.read_csv(src / rel, sep="\t")
        col_from, col_to, col_cnt = df.columns[0], df.columns[1], df.columns[2]
        edges = []
        for a, b, c in zip(df[col_from].values, df[col_to].values,
                             df[col_cnt].values):
            for _ in range(int(c)):
                edges.append((int(a), int(b)))
        n_target = min(n_swaps, len(edges) // 2)
        new_edges, stats = rewire_directed_blocks(
            edges, block, n_swaps=n_target, rng=rng,
            prefer_triangle_destroy=prefer_triangle_destroy,
            max_seconds=max_seconds)
        from collections import Counter
        c = Counter(new_edges)
        rows = [(a, b, cnt) for (a, b), cnt in c.items()]
        new_df = pd.DataFrame(rows, columns=[col_from, col_to, col_cnt])
        new_df.to_csv(dst / rel, sep="\t", index=False)
        per_rel_stats[rel] = {"n_edges": len(edges),
                                "n_unique_edges": len(rows),
                                **stats}
        print(f"  {rel}: {stats['accepted']}/{stats['proposed']} swaps"
              f" ({stats['elapsed_seconds']:.1f}s)", flush=True)
    return per_rel_stats


LINE_TEST = re.compile(
    r"test results:\s+accuracy:\s+([0-9.]+)\s+f1_score:\s+([0-9.]+)")


def run_timme(dataset: str, seed: int, epochs: int = 300) -> dict:
    import os as _os
    cmd = [sys.executable, "main.py", "-e", str(epochs), "-d", dataset,
           "-t", "Classification", "--seed", str(seed)]
    env = dict(_os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    proc = subprocess.run(cmd, cwd=TIMME_CODE, capture_output=True,
                            text=True, timeout=1800, env=env)
    m = LINE_TEST.search(proc.stdout)
    if not m:
        return {"seed": seed, "dataset": dataset, "accuracy": float("nan"),
                 "f1": float("nan"),
                 "stderr_tail": "\n".join(proc.stderr.splitlines()[-10:])}
    return {"seed": seed, "dataset": dataset,
             "accuracy": float(m.group(1)),
             "f1": float(m.group(2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--start_seed", type=int, default=42)
    ap.add_argument("--n_swaps", type=int, default=200_000)
    ap.add_argument("--max_seconds", type=float, default=90.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--no_triangle_destroy", action="store_true",
                     help="Configuration-model rewire without triangle-destroy preference.")
    args = ap.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    all_results = []
    for s in seeds:
        print(f"\n=== seed {s} ===", flush=True)
        dst_name = f"PureP_rewired_seed{s}"
        rewire_stats = build_rewired_dataset(
            "PureP", dst_name, seed=s, n_swaps=args.n_swaps,
            max_seconds=args.max_seconds,
            prefer_triangle_destroy=not args.no_triangle_destroy)
        r_orig = run_timme("PureP", s, epochs=args.epochs)
        r_rw = run_timme(dst_name, s, epochs=args.epochs)
        all_results.append({"seed": s, "orig": r_orig, "rewired": r_rw,
                              "rewire_stats": rewire_stats})
        print(f"  orig    : acc={r_orig['accuracy']:.4f} f1={r_orig['f1']:.4f}",
              flush=True)
        print(f"  rewired : acc={r_rw['accuracy']:.4f} f1={r_rw['f1']:.4f}",
              flush=True)

    acc_orig = np.array([r["orig"]["accuracy"] for r in all_results])
    acc_rw = np.array([r["rewired"]["accuracy"] for r in all_results])
    deltas = acc_rw - acc_orig
    print("\n=== TIMME rewire audit summary ===", flush=True)
    print(f"n_seeds  = {len(seeds)}", flush=True)
    print(f"acc orig = {acc_orig.mean():+.4f} +/- {acc_orig.std(ddof=1):.4f}",
          flush=True)
    print(f"acc rw   = {acc_rw.mean():+.4f} +/- {acc_rw.std(ddof=1):.4f}",
          flush=True)
    print(f"delta    = {deltas.mean():+.4f} +/- {deltas.std(ddof=1):.4f},"
          f" range [{deltas.min():+.4f}, {deltas.max():+.4f}]", flush=True)

    out = {"n_seeds": len(seeds), "n_swaps_target": args.n_swaps,
             "max_seconds": args.max_seconds,
             "no_triangle_destroy": args.no_triangle_destroy,
             "acc_orig_mean": float(acc_orig.mean()),
             "acc_orig_std": float(acc_orig.std(ddof=1)),
             "acc_rewired_mean": float(acc_rw.mean()),
             "acc_rewired_std": float(acc_rw.std(ddof=1)),
             "delta_mean": float(deltas.mean()),
             "delta_std": float(deltas.std(ddof=1)),
             "delta_range": [float(deltas.min()), float(deltas.max())],
             "per_seed": all_results}
    out_path = RESULTS_DIR / "rewire_audit.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {out_path}", flush=True)


if __name__ == "__main__":
    main()
