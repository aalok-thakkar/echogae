#!/usr/bin/env python3
"""One figure for the ReScience article: paired user-vs-community averages
for the four audited topics, with the paper's reported values and the
notebook-stored values marked for comparison. Minimal matplotlib style."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rescience-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/validation/results"
OUT = ROOT / "paper/rescience"

TOPICS = ["gun", "abortion", "super_bowl", "sxsw"]
NAMES = ["Gun control", "Abortion", "Super Bowl", "SXSW"]
PAPER = [0.714, 0.626, 0.485, 0.465]
NOTEBOOK = [0.628, 0.622, 0.562, 0.465]

INK = "#111111"
MUTED = "#8a8a8a"
BLUE = "#1f6fb2"
ORANGE = "#d06a2e"
FAINT = "#dddddd"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


def load_topic(t):
    p = DATA / f"{t}.json"
    with open(p) as f:
        return json.load(f)


def main():
    rows = [load_topic(t) for t in TOPICS]
    # Extract ten user-avg and community-avg at partition seed 42.
    user_all, comm_all = [], []
    for r in rows:
        at42 = [x for x in r["factorial"] if x["partition_seed"] == 42]
        user_all.append([x["user"] for x in at42])
        comm_all.append([x["community"] for x in at42])

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for i, (u, c) in enumerate(zip(user_all, comm_all)):
        um, cm = float(np.mean(u)), float(np.mean(c))
        us, cs = float(np.std(u, ddof=1)), float(np.std(c, ddof=1))
        # Thin paired connector between the two means.
        ax.plot([um, cm], [i, i], lw=0.8, color=MUTED, zorder=1)
        # Ten-fit scatter faintly behind the means.
        off = np.linspace(-0.1, 0.1, len(u))
        ax.scatter(u, i + off, s=9, color=BLUE, alpha=0.35, linewidths=0, zorder=2)
        ax.scatter(c, i + off, s=11, facecolor="none",
                   edgecolor=ORANGE, alpha=0.7, linewidths=0.6, zorder=2)
        # Means on top.
        ax.errorbar(um, i, xerr=us, fmt="o", color=BLUE,
                    markersize=5, linewidth=0.8, capsize=2, zorder=4)
        ax.errorbar(cm, i, xerr=cs, fmt="s", color=ORANGE,
                    markerfacecolor="none", markersize=5, linewidth=0.8,
                    capsize=2, zorder=4)
        # Published and notebook-stored reference marks.
        ax.scatter(PAPER[i], i - 0.22, marker="x", s=36, color=INK,
                   linewidths=1.0, zorder=5)
        ax.scatter(NOTEBOOK[i], i + 0.22, marker="+", s=40, color=INK,
                   linewidths=1.0, zorder=5)

    ax.set(yticks=range(4), yticklabels=NAMES,
           xlim=(0.40, 0.76), xlabel="Echo Chamber Score")
    ax.invert_yaxis()
    ax.grid(axis="x", color=FAINT, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)

    ax.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, ls="", markersize=5,
               label=r"User average $S_{\mathrm{user}}$ (10 fits)"),
        Line2D([], [], marker="s", color=ORANGE, markerfacecolor="none",
               ls="", markersize=5, markeredgewidth=0.8,
               label=r"Community average $S_{\mathrm{community}}$ (10 fits)"),
        Line2D([], [], marker="x", color=INK, ls="", markersize=6,
               label="Paper (Table II)"),
        Line2D([], [], marker="+", color=INK, ls="", markersize=6,
               label="Released notebook"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
        frameon=False, handletextpad=0.4, columnspacing=1.2)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.30)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure_averaging.{ext}",
                    bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close(fig)
    print(f"Wrote {OUT/'figure_averaging.pdf'}")


if __name__ == "__main__":
    main()
