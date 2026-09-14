"""Multi-seed TIMME reproduction on the PureP dataset.

For each seed s in a list of N seeds, invokes TIMME (the KDD 2020 model)
end to end via its own main.py entry point on ../data/PureP with the
default hyperparameters used in the published paper. Captures the final
test accuracy and F1 on the ideology-classification head. Reports the
across-seed mean, std, and range.

Purpose: exactly the same reproducibility axis we already applied to
EchoGAE and InfoVGAE. TIMME is supervised (ideology labels present in
PureP's dict.csv), so 'reproduction' here is the seed-level variance of
the reported classification accuracy on the fixed 80/10/10 label split
after TIMME's own random shuffle.

Under TIMME's default settings the model uses a random 80/10/10 label
partition (from utils.multi_relation_load, random.shuffle). Fixing the
Python/random/numpy/torch seeds across the whole loop fixes both the
label split and the model init. Varying the seed varies both.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


TIMME_CODE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "TIMME" / "code"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "timme_multi_seed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


LINE_TEST = re.compile(
    r"test results:\s+accuracy:\s+([0-9.]+)\s+f1_score:\s+([0-9.]+)")


def run_one(seed: int, dataset: str = "PureP", epochs: int = 300,
             task: str = "Classification") -> dict:
    """Run TIMME once at a given seed. Return {accuracy, f1_score, stdout_tail}.

    Sets PYTHONHASHSEED in the subprocess env so that TIMME's
    hash-order-dependent id-to-row mapping (utils.multi_relation_load
    consumes set iteration order) is stable across invocations.
    """
    import os as _os
    cmd = [
        sys.executable, "main.py",
        "-e", str(epochs),
        "-d", dataset,
        "-t", task,
        "--seed", str(seed),
    ]
    env = dict(_os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    proc = subprocess.run(
        cmd, cwd=TIMME_CODE_DIR, capture_output=True, text=True,
        timeout=1800, env=env)
    tail = "\n".join(proc.stdout.strip().split("\n")[-15:])
    if proc.returncode != 0:
        return {"seed": seed, "returncode": proc.returncode,
                "stderr_tail": "\n".join(proc.stderr.strip().split("\n")[-15:]),
                "stdout_tail": tail}
    m = LINE_TEST.search(proc.stdout)
    if not m:
        return {"seed": seed, "accuracy": float("nan"),
                "f1_score": float("nan"), "stdout_tail": tail,
                "note": "regex did not match"}
    return {"seed": seed,
             "accuracy": float(m.group(1)),
             "f1_score": float(m.group(2)),
             "stdout_tail": tail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--start_seed", type=int, default=42)
    ap.add_argument("--dataset", type=str, default="PureP")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--task", type=str, default="Classification")
    args = ap.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    print(f"Running TIMME on {args.dataset} across {len(seeds)} seeds:"
          f" {seeds}", flush=True)

    per_seed = []
    for s in seeds:
        print(f"\n=== seed {s} ===", flush=True)
        r = run_one(s, dataset=args.dataset, epochs=args.epochs, task=args.task)
        print(json.dumps({k: v for k, v in r.items()
                            if k not in ("stdout_tail", "stderr_tail")}),
              flush=True)
        per_seed.append(r)

    acc = np.array([r.get("accuracy", float("nan")) for r in per_seed])
    f1 = np.array([r.get("f1_score", float("nan")) for r in per_seed])
    good = ~np.isnan(acc)

    print("\n=== Multi-seed TIMME summary ===", flush=True)
    print(f"dataset  = {args.dataset}", flush=True)
    print(f"task     = {args.task}", flush=True)
    print(f"n_seeds  = {good.sum()}/{len(seeds)}", flush=True)
    if good.sum() >= 2:
        print(f"accuracy = {acc[good].mean():+.4f} +/- {acc[good].std(ddof=1):.4f},"
              f" range [{acc[good].min():+.4f}, {acc[good].max():+.4f}]",
              flush=True)
        print(f"f1_score = {f1[good].mean():+.4f} +/- {f1[good].std(ddof=1):.4f},"
              f" range [{f1[good].min():+.4f}, {f1[good].max():+.4f}]",
              flush=True)

    out = {
        "dataset": args.dataset,
        "task": args.task,
        "epochs": args.epochs,
        "n_seeds_total": len(seeds),
        "n_seeds_success": int(good.sum()),
        "accuracy_mean": float(np.nanmean(acc)),
        "accuracy_std": float(np.nanstd(acc, ddof=1)) if good.sum() >= 2 else float("nan"),
        "accuracy_range": [float(np.nanmin(acc)) if good.sum() else float("nan"),
                            float(np.nanmax(acc)) if good.sum() else float("nan")],
        "f1_mean": float(np.nanmean(f1)),
        "f1_std": float(np.nanstd(f1, ddof=1)) if good.sum() >= 2 else float("nan"),
        "per_seed": [{k: v for k, v in r.items() if k not in ("stdout_tail",)}
                       for r in per_seed],
    }
    out_path = RESULTS_DIR / f"multi_seed_{args.dataset}_{args.task}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {out_path}", flush=True)


if __name__ == "__main__":
    main()
