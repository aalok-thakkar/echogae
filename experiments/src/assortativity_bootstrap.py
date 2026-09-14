"""Bootstrap CIs on labelled-subset assortativity for the four EchoGAE
topics and the seven audit datasets.

Assortativity is a deterministic single-graph statistic; it has no
seed variance. What it does have is finite-sample variance in the
choice of *which* vertices are labelled. This script resamples the
labelled subset with replacement 500 times per dataset and reports
the mean, std, and 95% CI of Newman signed attribute assortativity
under the resampling.

This gives assortativity its own error bar, comparable to the
across-seed std reported for EchoGAE's ECS, so a reader can compare
apples to apples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" /
                        "echo-chamber-score"))
from direct_estimators import (  # noqa: E402
    load_topic, load_reddit, load_timme, load_polblogs,
)

DATA = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score" / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "assort_bootstrap"
RESULTS.mkdir(parents=True, exist_ok=True)


def assortativity_on_labelled(G: nx.Graph, side: dict[int, int],
                                labelled_v: list[int]) -> float:
    S = set(labelled_v)
    H = G.subgraph(S).copy()
    if H.number_of_edges() == 0:
        return float("nan")
    for v in H.nodes:
        H.nodes[v]["block"] = int(side[v])
    try:
        return float(nx.attribute_assortativity_coefficient(H, "block"))
    except Exception:
        return float("nan")


def bootstrap(G: nx.Graph, side: dict[int, int], n_boot: int,
               seed: int = 42) -> dict:
    labelled = sorted(side.keys())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        picks = rng.choice(len(labelled), size=len(labelled), replace=True)
        subset = [labelled[i] for i in picks]
        # De-dup for the induced subgraph; multiplicity affects nothing on
        # the subgraph statistic.
        v = assortativity_on_labelled(G, side, list(set(subset)))
        if not np.isnan(v):
            boots.append(v)
    a = np.array(boots)
    return {"n_boot": int(len(a)),
            "point": float(assortativity_on_labelled(G, side, labelled)),
            "mean": float(a.mean()), "std": float(a.std(ddof=1)),
            "ci95": [float(np.percentile(a, 2.5)),
                      float(np.percentile(a, 97.5))]}


def load_echogae_topic_wrapper(topic: str):
    G = load_topic(topic)
    al = pd.read_feather(DATA / topic / "allsides.feather")
    side = {}
    for v in G.nodes:
        uid = str(G.nodes[v]["user_id"])
        r = al[al["user_id"] == uid]
        if not r.empty and pd.notna(r.iloc[0]["allsides_score"]):
            s = float(r.iloc[0]["allsides_score"])
            side[v] = 0 if s < 0 else 1
    return G, side


def main():
    out = {}
    for t in ("gun", "abortion", "super_bowl", "sxsw"):
        print(f"\n== {t}", flush=True)
        G, side = load_echogae_topic_wrapper(t)
        r = bootstrap(G, side, n_boot=500)
        out[t] = r
        print(json.dumps(r, indent=2), flush=True)
    print(f"\n== reddit", flush=True)
    G, side = load_reddit()
    out["reddit"] = bootstrap(G, side, n_boot=500)
    print(json.dumps(out["reddit"], indent=2), flush=True)
    for d in ("PureP", "P50"):
        print(f"\n== {d}", flush=True)
        G, side = load_timme(d)
        out[d] = bootstrap(G, side, n_boot=500)
        print(json.dumps(out[d], indent=2), flush=True)
    print(f"\n== polblogs", flush=True)
    G, side = load_polblogs()
    out["polblogs"] = bootstrap(G, side, n_boot=500)
    print(json.dumps(out["polblogs"], indent=2), flush=True)
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {RESULTS / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
