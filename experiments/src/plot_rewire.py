"""Before/after plot: rewiring's effect on graph clustering vs. EchoGAE's ECS.

Reads the four per-topic JSON files under results/echogae_rewire/ and
produces a two-panel figure showing (i) collapse in the closure signal and
(ii) invariance of the estimator output.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).resolve().parents[1] / "results" / "echogae_rewire"
OUT_DIR = Path(__file__).resolve().parents[2] / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = ["sxsw", "super_bowl", "gun", "abortion"]
TOPIC_LABELS = {
    "sxsw": "SXSW",
    "super_bowl": "Super Bowl",
    "gun": "Gun control",
    "abortion": "Abortion",
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

    # Panel 1: clustering coefficient before/after.
    clus_before = [data[t]["before"]["avg_clustering"] for t in TOPICS]
    clus_after = [data[t]["after"]["avg_clustering"] for t in TOPICS]
    ax1.bar(x - width / 2, clus_before, width, label="Original graph",
            color="#4a6fa5")
    ax1.bar(x + width / 2, clus_after, width, label="After rewiring",
            color="#c85a5a")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Average local clustering coefficient")
    ax1.set_title("Rewiring collapses the closure signal")
    ax1.legend(frameon=False, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2: EchoGAE ECS before/after.
    ecs_before = [data[t]["before"]["overall_eci"] for t in TOPICS]
    ecs_after = [data[t]["after"]["overall_eci"] for t in TOPICS]
    ax2.bar(x - width / 2, ecs_before, width, label="Original graph",
            color="#4a6fa5")
    ax2.bar(x + width / 2, ecs_after, width, label="After rewiring",
            color="#c85a5a")
    ax2.axhline(0.5, color="grey", linestyle=":", linewidth=1,
                label="No-discrimination midpoint")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("EchoGAE community-level ECS")
    ax2.set_ylim(0.48, 0.56)
    ax2.set_title("EchoGAE's output does not move")
    ax2.legend(frameon=False, loc="upper right", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "rewire_before_after.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")

    # Delta bar chart as a companion PNG for the paper's discussion.
    fig2, ax = plt.subplots(1, 1, figsize=(6.5, 3.2), dpi=140)
    delta_clus = [c1 - c0 for c0, c1 in zip(clus_before, clus_after)]
    delta_ecs = [e1 - e0 for e0, e1 in zip(ecs_before, ecs_after)]
    ax.bar(x - width / 2, delta_clus, width,
           label=r"$\Delta$ clustering", color="#4a6fa5")
    ax.bar(x + width / 2, delta_ecs, width,
           label=r"$\Delta$ ECS", color="#c85a5a")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"$\Delta$ (after $-$ before)")
    ax.set_title("Closure collapses; the estimator does not")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out2 = OUT_DIR / "rewire_delta.pdf"
    plt.savefig(out2, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
