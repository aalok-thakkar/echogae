"""Graph-blind baseline against TIMME on PureP.

The audit question: how much of TIMME's ideology-classification accuracy
survives when we strip the graph structure entirely? A per-user linear
classifier on the GloVe features TIMME itself accepts as node inputs is
already available to a practitioner who has no graph at all.

Setup. Use TIMME's own PureP loader (multi_relation_load) to get the
label map and the 80/10/10 index split under the same seed schedule as
timme_multi_seed.py. For each seed, fit logistic regression on
tweet_features['tweets_average'] (all-user GloVe tweet embeddings) or
features['description'] (profile-description GloVe) for the labeled
users, then predict on the held-out test users.

The reported metric is the fraction of TIMME's own test accuracy that
this graph-blind ceiling already reaches.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


TIMME_CODE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "TIMME" / "code"
TIMME_DATA_DIR = Path(__file__).resolve().parents[1] / "vendor" / "TIMME" / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "timme_graph_blind"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_purep_labels(dataset: str = "PureP", seed: int = 42,
                       add_additional: bool = True):
    """Reproduce the labeled-index / label-split logic in
    utils.multi_relation_load without loading the adjacency."""
    import math

    data_dir = TIMME_DATA_DIR / dataset
    label_df = pd.read_csv(data_dir / "dict.csv", sep="\t")
    labeled_ids = label_df["twitter_id"].values
    label_list = label_df["party"].values
    if add_additional:
        add_df = pd.read_csv(
            TIMME_DATA_DIR / "additional_labels" / "new_dict_cleaned.csv",
            sep="\t")
        labeled_ids = np.append(labeled_ids, add_df["twitter_id"].values)
        label_list = np.append(label_list, add_df["party"].values)

    # Same "valid_ids" filter TIMME applies: keep only ids that appear in
    # at least one relation. Match TIMME by loading the five relations.
    files = ["retweet_list.csv", "mention_list.csv", "friend_list.csv",
              "reply_list.csv", "favorite_list.csv"]
    all_ids = set()
    for f in files:
        df = pd.read_csv(data_dir / f, sep="\t")
        all_ids |= set(df[df.columns[0]])
        all_ids |= set(df[df.columns[1]])
    ignored = [i for i, (t, v) in enumerate(zip(labeled_ids, label_list))
                if v == "I" or t not in all_ids]
    labeled_ids = np.delete(labeled_ids, ignored)
    label_list = np.delete(label_list, ignored)

    label_categories = sorted(set(label_list))
    label_map = {c: i for i, c in enumerate(label_categories)}
    labels = np.array([label_map[v] for v in label_list], dtype=np.int64)

    n = len(labels)
    n_train = math.ceil(n * .8)
    n_valid = math.ceil(n * .1)

    # TIMME does random.shuffle(idx_all) after seeding Python's random.
    idx_all = list(range(n))
    random.seed(seed)
    random.shuffle(idx_all)
    idx_train = idx_all[:n_train]
    idx_val = idx_all[n_train: n_train + n_valid]
    idx_test = idx_all[n_train + n_valid:]

    return {"labeled_ids": labeled_ids, "labels": labels,
             "label_map": label_map,
             "idx_train": idx_train, "idx_val": idx_val, "idx_test": idx_test}


def load_features_for(labeled_ids, dataset: str, feature: str):
    """Return a matrix (n_labeled, d) of GloVe features aligned to labeled_ids."""
    data_dir = TIMME_DATA_DIR / dataset
    all_ids = pd.read_csv(data_dir / "all_twitter_ids.csv",
                           sep="\t")["twitter_id"].values
    id_to_row = {int(t): i for i, t in enumerate(all_ids)}
    if feature == "tweets_average":
        arr = np.load(data_dir / "tweet_features.npz")["tweets_average"]
    elif feature == "description":
        arr = np.load(data_dir / "features.npz")["description"]
    elif feature == "status":
        arr = np.load(data_dir / "features.npz")["status"]
    else:
        raise ValueError(feature)
    rows = np.array([id_to_row[int(t)] for t in labeled_ids])
    return arr[rows]


def one_seed(dataset: str, feature: str, seed: int) -> dict:
    info = load_purep_labels(dataset=dataset, seed=seed)
    X = load_features_for(info["labeled_ids"], dataset, feature)
    y = info["labels"]
    idx_train = info["idx_train"]
    idx_test = info["idx_test"]
    scaler = StandardScaler().fit(X[idx_train])
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(Xs[idx_train], y[idx_train])
    pred = clf.predict(Xs[idx_test])
    return {"seed": seed,
             "feature": feature,
             "n_labeled": int(len(y)),
             "n_train": len(idx_train),
             "n_test": len(idx_test),
             "accuracy": float(accuracy_score(y[idx_test], pred)),
             "f1": float(f1_score(y[idx_test], pred, average="macro"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="PureP")
    ap.add_argument("--feature", type=str, default="tweets_average",
                     choices=["tweets_average", "description", "status"])
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--start_seed", type=int, default=42)
    args = ap.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    per_seed = []
    for s in seeds:
        r = one_seed(args.dataset, args.feature, s)
        print(json.dumps({k: v for k, v in r.items() if k != "feature"}),
              flush=True)
        per_seed.append(r)

    acc = np.array([r["accuracy"] for r in per_seed])
    f1_ = np.array([r["f1"] for r in per_seed])
    print("\n=== Graph-blind logistic regression on TIMME features ===",
          flush=True)
    print(f"dataset  = {args.dataset}", flush=True)
    print(f"feature  = {args.feature}", flush=True)
    print(f"n_seeds  = {len(seeds)}", flush=True)
    print(f"accuracy = {acc.mean():+.4f} +/- {acc.std(ddof=1):.4f},"
          f" range [{acc.min():+.4f}, {acc.max():+.4f}]", flush=True)
    print(f"f1_macro = {f1_.mean():+.4f} +/- {f1_.std(ddof=1):.4f},"
          f" range [{f1_.min():+.4f}, {f1_.max():+.4f}]", flush=True)

    out = {"dataset": args.dataset, "feature": args.feature,
             "n_seeds": len(seeds),
             "accuracy_mean": float(acc.mean()),
             "accuracy_std": float(acc.std(ddof=1)),
             "accuracy_range": [float(acc.min()), float(acc.max())],
             "f1_mean": float(f1_.mean()),
             "f1_std": float(f1_.std(ddof=1)),
             "per_seed": per_seed}
    out_path = RESULTS_DIR / f"graph_blind_{args.dataset}_{args.feature}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {out_path}", flush=True)


if __name__ == "__main__":
    main()
