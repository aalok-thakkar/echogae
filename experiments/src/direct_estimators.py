"""Direct polarisation estimators on the audit datasets.

For each graph carried by the audit (EchoGAE's four Twitter topics; the
Reddit subreddit-subreddit component; TIMME's PureP and P50 unions),
compute:

  clustering    average local clustering (Watts-Strogatz)
  transitivity  global clustering / triangle fraction
  modularity    modularity of the seed=42 Louvain partition
  homophily     edge homophily on AllSides label sign (Twitter),
                Waller-Anderson score sign (Reddit),
                Democrat/Republican label (TIMME)
  assortativity Newman signed attribute assortativity coefficient
  rwc           random walk controversy score (Garimella et al. 2018)
                on the two-community split
  n_labelled    number of labelled vertices used

These are deterministic single-graph statistics: no seeds, no
training, no error bars. They stand as the counterfactual: what a
direct estimator says on the same data an MPNN score is claimed to
summarise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" /
                        "echo-chamber-score"))

from networkx.algorithms.community.louvain import louvain_communities  # noqa: E402
from networkx.algorithms.community.quality import modularity  # noqa: E402


def load_topic(topic: str) -> nx.Graph:
    """Local reimplementation of echogae_reproduction.load_topic without
    the transitive torch_geometric import."""
    gml = DATA / topic / "graph.gml"
    G = nx.read_gml(gml)
    G = nx.convert_node_labels_to_integers(G, label_attribute="_orig")
    for v in G.nodes:
        if "user_id" not in G.nodes[v]:
            G.nodes[v]["user_id"] = G.nodes[v].get("_orig", str(v))
    return G

DATA = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score" / "data"
TIMME_DATA = Path(__file__).resolve().parents[1] / "vendor" / "TIMME" / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "direct_estimators"
RESULTS.mkdir(parents=True, exist_ok=True)


def rwc_score(G: nx.Graph, side: np.ndarray, k: int = 20,
               n_walks: int = 500) -> float:
    """Approximate Garimella-style RWC on the two-side partition.
    Random walks from top-k degree vertices on each side; RWC = (a d - b c)
    with a, b, c, d probabilities of ending on each side."""
    L = np.where(side == 0)[0]
    R = np.where(side == 1)[0]
    if len(L) < k or len(R) < k:
        return float("nan")
    deg = dict(G.degree())
    top_L = sorted(L, key=lambda v: -deg.get(v, 0))[:k]
    top_R = sorted(R, key=lambda v: -deg.get(v, 0))[:k]
    top_L_set = set(top_L); top_R_set = set(top_R)
    rng = np.random.default_rng(42)

    def walk_end(start: int, targets_L: set, targets_R: set) -> int:
        v = start
        for _ in range(1000):
            if v in targets_L: return 0
            if v in targets_R: return 1
            nbrs = list(G.neighbors(v))
            if not nbrs: return -1
            v = int(rng.choice(nbrs))
        return -1

    # From side L, target authorities on both sides
    from_L = np.zeros(2, dtype=np.int64)
    from_R = np.zeros(2, dtype=np.int64)
    for _ in range(n_walks):
        s = int(rng.choice(L))
        e = walk_end(s, top_L_set, top_R_set)
        if e >= 0: from_L[e] += 1
        s = int(rng.choice(R))
        e = walk_end(s, top_L_set, top_R_set)
        if e >= 0: from_R[e] += 1
    if from_L.sum() < 20 or from_R.sum() < 20:
        return float("nan")
    a = from_L[0] / from_L.sum(); b = from_L[1] / from_L.sum()
    c = from_R[0] / from_R.sum(); d = from_R[1] / from_R.sum()
    return float(a * d - b * c)


def stats(G: nx.Graph, side: dict[int, int] | None, resolution: float,
           seed: int) -> dict:
    G_und = G.to_undirected() if G.is_directed() else G
    G_und = G_und.copy()
    G_und.remove_edges_from(list(nx.selfloop_edges(G_und)))
    n = G_und.number_of_nodes(); m = G_und.number_of_edges()
    cc = max(nx.connected_components(G_und), key=len)
    Gc = G_und.subgraph(cc).copy()

    parts = list(louvain_communities(Gc, resolution=resolution, seed=seed))
    mod = float(modularity(Gc, parts, resolution=resolution))
    n_comm = sum(1 for p in parts if len(p) > 1)

    avg_c = float(nx.average_clustering(G_und))
    trans = float(nx.transitivity(G_und))

    out = {"n": int(n), "m": int(m), "n_lcc": int(Gc.number_of_nodes()),
           "avg_clustering": avg_c, "transitivity": trans,
           "modularity_louvain": mod, "n_communities": int(n_comm)}

    if side is not None:
        labelled = [v for v in G_und.nodes if v in side]
        vals = np.array([side[v] for v in labelled])
        if len(np.unique(vals)) >= 2:
            attrs = {v: side[v] for v in labelled}
            for v in G_und.nodes:
                if v not in attrs: attrs[v] = -999
            nx.set_node_attributes(G_und, attrs, "block")
            H = G_und.subgraph(labelled).copy()
            same = sum(1 for u, v in H.edges if side[u] == side[v])
            homo = same / H.number_of_edges() if H.number_of_edges() else float("nan")
            try:
                assort = float(nx.attribute_assortativity_coefficient(H, "block"))
            except Exception:
                assort = float("nan")
            side_arr = np.full(n, -1, dtype=np.int8)
            for i, v in enumerate(sorted(G_und.nodes)):
                if v in side: side_arr[i] = side[v]
            # Relabel for rwc_score
            mapping = {v: i for i, v in enumerate(sorted(G_und.nodes))}
            G_rw = nx.relabel_nodes(G_und, mapping)
            rwc = rwc_score(G_rw, side_arr)
            out.update({"n_labelled": int(len(labelled)),
                        "homophily_edge": float(homo),
                        "assortativity_newman": assort,
                        "rwc": rwc})
    return out


def load_echogae_topic(topic: str):
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


def load_reddit():
    p = Path(__file__).resolve().parents[1] / "vendor" / "reddit"
    tsv = p / "hyperlinks-body.tsv"
    if not tsv.exists():
        return None, None
    df = pd.read_csv(tsv, sep="\t")
    edges = df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]]
    edges = edges[edges["SOURCE_SUBREDDIT"] != edges["TARGET_SUBREDDIT"]]
    from collections import Counter
    c = Counter(tuple(sorted(x)) for x in edges.itertuples(index=False))
    G = nx.Graph()
    for (a, b), n in c.items():
        if n >= 3:
            G.add_edge(a.lower(), b.lower())
    sc = pd.read_csv(p / "wa" / "data" / "scores.csv")
    m = {str(n).lower(): float(v) for n, v in
         zip(sc["community"], sc["partisan"])}
    nodes_with_score = [v for v in G.nodes if v in m and pd.notna(m[v])]
    G = G.subgraph(nodes_with_score).copy()
    if G.number_of_nodes() == 0:
        return None, None
    G0 = nx.convert_node_labels_to_integers(G, label_attribute="_orig")
    ccs = list(nx.connected_components(G0))
    biggest = max(ccs, key=len)
    Gc = G0.subgraph(biggest).copy()
    Gc = nx.convert_node_labels_to_integers(Gc, label_attribute="_orig2")
    side = {}
    for v in Gc.nodes:
        orig2 = Gc.nodes[v]["_orig2"]
        name = G0.nodes[orig2]["_orig"]
        if name in m and pd.notna(m[name]):
            side[v] = 0 if m[name] < 0 else 1
    return Gc, side


def load_polblogs():
    """Adamic-Glance 2005 political blogs. Directed hyperlink graph;
    'value' attribute is 0 = liberal, 1 = conservative. Take the
    undirected simple projection of the largest connected component
    (standard convention in polarisation studies)."""
    p = Path(__file__).resolve().parents[1] / "vendor" / "polblogs" / "polblogs.gml"
    # Manual GML parse: the release contains duplicate directed edges
    # that networkx 3.6's parser rejects.
    text = p.read_text(errors="ignore")
    G = nx.Graph()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "node [":
            block = {}
            i += 1
            while i < len(lines) and lines[i].strip() != "]":
                s = lines[i].strip()
                for k in ("id", "value"):
                    if s.startswith(k + " "):
                        block[k] = int(s.split()[1])
                if s.startswith("label "):
                    block["label"] = s.split(" ", 1)[1].strip().strip('"')
                i += 1
            if "id" in block:
                G.add_node(block["id"], value=block.get("value", -1),
                            label=block.get("label", ""))
        elif line == "edge [":
            block = {}
            i += 1
            while i < len(lines) and lines[i].strip() != "]":
                s = lines[i].strip()
                for k in ("source", "target"):
                    if s.startswith(k + " "):
                        block[k] = int(s.split()[1])
                i += 1
            if block.get("source") != block.get("target"):
                G.add_edge(block["source"], block["target"])
        i += 1
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    ccs = list(nx.connected_components(G))
    biggest = max(ccs, key=len)
    G = G.subgraph(biggest).copy()
    G = nx.convert_node_labels_to_integers(G, label_attribute="_orig")
    side = {v: int(G.nodes[v]["value"]) for v in G.nodes
            if "value" in G.nodes[v]}
    return G, side


def load_timme(dataset: str):
    ids = pd.read_csv(TIMME_DATA / dataset / "all_twitter_ids.csv",
                       sep="\t")["twitter_id"].values
    idx = {int(t): i for i, t in enumerate(ids)}
    G = nx.Graph()
    G.add_nodes_from(range(len(ids)))
    for rel in ["retweet_list.csv", "mention_list.csv", "friend_list.csv",
                 "reply_list.csv", "favorite_list.csv"]:
        df = pd.read_csv(TIMME_DATA / dataset / rel, sep="\t")
        for a, b in zip(df.iloc[:, 0].values, df.iloc[:, 1].values):
            i, j = idx.get(int(a)), idx.get(int(b))
            if i is not None and j is not None and i != j:
                G.add_edge(i, j)
    dict_df = pd.read_csv(TIMME_DATA / dataset / "dict.csv", sep="\t")
    add_df = pd.read_csv(TIMME_DATA / "additional_labels" /
                          "new_dict_cleaned.csv", sep="\t")
    side = {}
    for df in (dict_df, add_df):
        for t, p in zip(df["twitter_id"].values, df["party"].values):
            if p in ("D", "R"):
                i = idx.get(int(t))
                if i is not None:
                    side[i] = 0 if p == "D" else 1
    return G, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                     default=["gun", "abortion", "super_bowl", "sxsw",
                              "reddit", "PureP", "P50"])
    ap.add_argument("--resolution", type=float, default=0.05)
    args = ap.parse_args()

    out = {}
    for d in args.datasets:
        print(f"\n== {d}", flush=True)
        if d in ("gun", "abortion", "super_bowl", "sxsw"):
            G, side = load_echogae_topic(d)
            res = 0.05
        elif d == "reddit":
            G, side = load_reddit()
            res = 1.0
            if G is None:
                print("  reddit data not found, skipping"); continue
        elif d == "polblogs":
            G, side = load_polblogs()
            res = 1.0
        elif d in ("PureP", "P50"):
            G, side = load_timme(d)
            res = 1.0
        else:
            print(f"  unknown dataset {d}"); continue
        s = stats(G, side, resolution=res, seed=42)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in s.items()}), flush=True)
        out[d] = s
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {RESULTS / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
