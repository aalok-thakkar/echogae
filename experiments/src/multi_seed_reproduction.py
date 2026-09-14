"""Multi-seed EchoGAE reproduction under a pinned Sentence-BERT revision.

For each of the four EchoGAE datasets we run K seeds through the paper's
stated pipeline (Louvain resolution 0.05, seed varies; Sentence-BERT
all-MiniLM-L6-v2 at the pinned revision; EchoGAE 300 epochs, seed varies)
and report the mean, std, and range of the graph-level ECS across seeds.

The purpose is to distinguish the 0.06--0.31 gap against the paper's
Table 2 from seed-level pipeline noise.

The SBERT features are cached and revision-pinned, so cross-seed
variation comes from Louvain community detection and EchoGAE
training initialisation. torch.manual_seed(seed) is threaded into
EchoGAE_algorithm and GAE.run; networkx Louvain uses seed=seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ECHOGAE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "echo-chamber-score"
sys.path.insert(0, str(ECHOGAE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.EchoGAE import EchoGAE_algorithm  # noqa: E402
from src.echo_chamber_measure import EchoChamberMeasure  # noqa: E402
from echogae_reproduction import (  # noqa: E402
    load_topic, compute_sbert_features, louvain_communities,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "multi_seed_reproduction"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SBERT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

PUBLISHED = {"gun": 0.714, "abortion": 0.626,
             "super_bowl": 0.485, "sxsw": 0.465}


def prepare_topic(topic: str):
    """Return (G with SBERT filter applied, user_embeddings dict)."""
    G = load_topic(topic)
    sbert = compute_sbert_features(topic)
    keep = [n for n in G.nodes if str(G.nodes[n]["user_id"]) in sbert]
    if len(keep) < G.number_of_nodes():
        Gs = G.subgraph(keep).copy()
        for n in Gs.nodes:
            Gs.nodes[n]["twitter_user_id"] = str(Gs.nodes[n]["user_id"])
        G = nx.convert_node_labels_to_integers(Gs, label_attribute="_orig")
    else:
        for n in G.nodes:
            G.nodes[n]["twitter_user_id"] = str(G.nodes[n]["user_id"])
    user_embeddings = {n: sbert[G.nodes[n]["twitter_user_id"]] for n in G.nodes}
    return G, user_embeddings


def one_seed(G, user_embeddings, seed: int, epochs: int = 300,
             resolution: float = 0.05) -> dict:
    comm_ids = louvain_communities(G, resolution=resolution, seed=seed)
    unique, counts = np.unique(comm_ids, return_counts=True)
    non_singleton_mask = counts > 1
    ns_count = int(non_singleton_mask.sum())

    emb = EchoGAE_algorithm(G, user_embeddings=user_embeddings,
                            epochs=epochs, show_progress=False, seed=seed)
    ec = EchoChamberMeasure(emb, comm_ids, metric="euclidean")

    per_comm = []
    for c in unique[non_singleton_mask]:
        member_idx = np.where(comm_ids == c)[0].tolist()
        scores = [ec.metric(int(i)) for i in member_idx]
        per_comm.append(float(np.mean(scores)))
    graph_ecs = float(np.mean(per_comm)) if per_comm else float("nan")
    return {"seed": int(seed), "n_communities_non_singleton": ns_count,
            "graph_ecs": graph_ecs}


def run_topic_multi(topic: str, seeds: list, epochs: int = 300) -> dict:
    print(f"\n=== {topic}: {len(seeds)} seeds ===", flush=True)
    G, user_embeddings = prepare_topic(topic)
    print(f"  Graph after SBERT filter: n={G.number_of_nodes()}, "
          f"m={G.number_of_edges()}", flush=True)
    seed_results = []
    for i, seed in enumerate(seeds):
        r = one_seed(G, user_embeddings, seed=seed, epochs=epochs)
        seed_results.append(r)
        print(f"  seed {seed:>3}: ECS = {r['graph_ecs']:.4f}, "
              f"{r['n_communities_non_singleton']} communities", flush=True)

    ecs_vals = np.array([r["graph_ecs"] for r in seed_results])
    published = PUBLISHED[topic]
    summary = {
        "topic": topic,
        "sbert_revision": SBERT_REVISION,
        "n_seeds": len(seeds),
        "published": published,
        "ecs_mean": float(ecs_vals.mean()),
        "ecs_std": float(ecs_vals.std(ddof=1)),
        "ecs_min": float(ecs_vals.min()),
        "ecs_max": float(ecs_vals.max()),
        "gap_mean_to_published": float(ecs_vals.mean() - published),
        "gap_min_to_published": float(np.min(np.abs(ecs_vals - published))),
        "seed_results": seed_results,
    }
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+",
                    default=["gun", "abortion", "super_bowl", "sxsw"])
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix for the output file, e.g. _nx31.")
    args = ap.parse_args()

    seeds = list(range(42, 42 + args.n_seeds))
    print(f"Seeds: {seeds}", flush=True)
    print(f"networkx {nx.__version__}", flush=True)

    all_out = {}
    for t in args.topics:
        try:
            all_out[t] = run_topic_multi(t, seeds=seeds, epochs=args.epochs)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"ERROR on {t}: {e}", flush=True)

    print("\n=== Multi-Seed Summary ===", flush=True)
    print(f"{'Topic':<12} {'Paper':<8} {'Ours (mean +/- std)':<24} "
          f"{'Range':<20}", flush=True)
    for t, s in all_out.items():
        print(f"  {t:<10} {s['published']:<8.3f} "
              f"{s['ecs_mean']:.3f} +/- {s['ecs_std']:.3f}            "
              f"[{s['ecs_min']:.3f}, {s['ecs_max']:.3f}]", flush=True)

    out_path = RESULTS_DIR / f"multi_seed{args.tag}.json"
    with out_path.open("w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nWritten to {out_path}", flush=True)
