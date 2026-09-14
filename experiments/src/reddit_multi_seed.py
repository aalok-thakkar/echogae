"""Multi-seed Reddit audit.

Runs `reddit_audit.audit` at N seeds on the SNAP soc-RedditHyperlinks x
Waller-Anderson subreddit graph. Reports mean +/- std of the community-level
ECS on the original graph and of the rewire delta.

Motivation: our critique of single-seed publication practice should not
itself use single-seed numbers on the Reddit supplement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reddit_audit import audit


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "reddit_multi_seed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--rewire_seconds", type=float, default=90.0)
    args = ap.parse_args()

    seeds = list(range(42, 42 + args.n_seeds))
    print(f"Running {len(seeds)} seeds: {seeds}", flush=True)

    results = []
    for s in seeds:
        try:
            print(f"\n=== seed {s} ===", flush=True)
            r = audit(min_edge_weight=3, epochs=args.epochs,
                       rewire_seconds=args.rewire_seconds, seed=s)
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"seed {s} failed: {e}", flush=True)

    ecs_before = np.array([r["graph_level_ecs"] for r in results])
    ecs_after = np.array([r["graph_level_ecs_rewired"] for r in results])
    deltas = ecs_after - ecs_before
    r2 = np.array([r.get("graph_blind_r2_mean", float("nan")) for r in results])

    print("\n=== Multi-seed Reddit summary ===", flush=True)
    print(f"n_seeds = {len(results)}", flush=True)
    print(f"ECS (original):    {ecs_before.mean():+.4f} +/- {ecs_before.std(ddof=1):.4f}, "
          f"range [{ecs_before.min():+.4f}, {ecs_before.max():+.4f}]", flush=True)
    print(f"ECS (rewired):     {ecs_after.mean():+.4f} +/- {ecs_after.std(ddof=1):.4f}, "
          f"range [{ecs_after.min():+.4f}, {ecs_after.max():+.4f}]", flush=True)
    print(f"Delta ECS:         {deltas.mean():+.4f} +/- {deltas.std(ddof=1):.4f}, "
          f"range [{deltas.min():+.4f}, {deltas.max():+.4f}]", flush=True)
    print(f"Graph-blind R^2:   {np.nanmean(r2):+.4f} +/- {np.nanstd(r2, ddof=1):.4f}",
          flush=True)

    out = {"n_seeds": len(results),
            "ecs_original_mean": float(ecs_before.mean()),
            "ecs_original_std": float(ecs_before.std(ddof=1)),
            "ecs_original_range": [float(ecs_before.min()), float(ecs_before.max())],
            "ecs_rewired_mean": float(ecs_after.mean()),
            "ecs_rewired_std": float(ecs_after.std(ddof=1)),
            "delta_mean": float(deltas.mean()),
            "delta_std": float(deltas.std(ddof=1)),
            "delta_range": [float(deltas.min()), float(deltas.max())],
            "r2_mean": float(np.nanmean(r2)),
            "r2_std": float(np.nanstd(r2, ddof=1)),
            "seed_results": results}
    with (RESULTS_DIR / "multi_seed.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {RESULTS_DIR / 'multi_seed.json'}", flush=True)
