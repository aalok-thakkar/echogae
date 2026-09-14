"""Rewiring under the paper's exact Louvain pipeline (v2 audit).

Two-panel figure reading v2 JSON results:
  Panel A: average local clustering coefficient, original vs.~rewired.
  Panel B: EchoGAE community-level ECS, original vs.~rewired, with the
           $0.5$ no-discrimination midpoint marked.

The point of the figure is to place the closure-collapse signal (Panel A)
and the estimator's near-invariance (Panel B) on the same axes per topic,
so the mismatch a closure-tracking estimator would not exhibit is visual.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).resolve().parents[1] / "results" / "echogae_audit_v2"
OUT_DIR = Path(__file__).resolve().parents[2] / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = ["sxsw", "super_bowl", "gun", "abortion"]
TOPIC_LABELS = {
    "sxsw": "SXSW", "super_bowl": "Super Bowl",
    "gun": "Gun control", "abortion": "Abortion",
}


def load(topic: str) -> dict:
    with (RESULTS / f"{topic}.json").open() as f:
        return json.load(f)


def main() -> None:
    data = {t: load(t) for t in TOPICS}

    labels = [TOPIC_LABELS[t] for t in TOPICS]
    x = np.arange(len(labels))
    width = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.5), dpi=140)

    clus_before = [data[t]["clus_before"] for t in TOPICS]
    clus_after = [data[t]["clus_after"] for t in TOPICS]
    ax1.bar(x - width / 2, clus_before, width, label="Original graph",
            color="#4a6fa5")
    ax1.bar(x + width / 2, clus_after, width, label="After rewiring",
            color="#c85a5a")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Average local clustering")
    ax1.set_title("(a) Closure collapses")
    ax1.legend(frameon=False, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ecs_before = [data[t]["graph_level_ecs"] for t in TOPICS]
    ecs_after = [data[t]["graph_level_ecs_rewired"] for t in TOPICS]
    ax2.bar(x - width / 2, ecs_before, width, label="Original graph",
            color="#4a6fa5")
    ax2.bar(x + width / 2, ecs_after, width, label="After rewiring",
            color="#c85a5a")
    ax2.axhline(0.5, color="grey", linestyle=":", linewidth=1,
                label="No-discrimination midpoint")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("EchoGAE community-level ECS")
    ymin = min(min(ecs_before), min(ecs_after), 0.45) - 0.02
    ymax = max(max(ecs_before), max(ecs_after), 0.55) + 0.02
    ax2.set_ylim(ymin, ymax)
    ax2.set_title("(b) EchoGAE score does not follow")
    ax2.legend(frameon=False, loc="upper right", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "rewire_v2_side_by_side.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
