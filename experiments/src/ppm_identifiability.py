"""Numerical identifiability check for the entanglement claim.

Reframed question. Instead of asking "do two matched theta produce identical MPNN outputs?"
(which requires matching high-order WL joint distributions, out of reach with three params),
ask: "how much does closure vary along the (degree = D, homophily = h) level set of PPM
parameter space?"

If closure varies by a constant while (degree, homophily) are pinned, then MPNN outputs
(which have limited additional discrimination power beyond the depth-1 WL joint statistics
that are themselves close to determined by (degree, homophily)) cannot uniformly recover
closure. That is the operational entanglement / non-identifiability claim.

Method.
  Sweep (p, q, sigma) over a grid. Reject samples that don't satisfy |degree - D| < eps
  and |homophily - h| < eps. On the surviving parameter set, plot the range of
  expected clustering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "ppm_check"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Theta:
    p: float
    q: float
    sigma: float

    def expected_degree(self, n: int) -> float:
        return (n - 1) / 2 * (self.p + self.q)

    def expected_homophily(self) -> float:
        rho = (1 - self.sigma) ** 2 + self.sigma**2
        mu = 2 * self.sigma * (1 - self.sigma)
        return (self.p * rho + self.q * mu) / (self.p + self.q)

    def expected_clustering(self, n: int) -> float:
        """Leading-order average clustering coefficient of two-block PPM as n -> infinity.
        For a vertex in block B_+ of size n/2:
          P(neighbour u in B_+) = p / (p + q)  (given a neighbour exists)
          P(neighbour u in B_-) = q / (p + q)
        Pair of neighbours (u, w). Probability an edge between them:
          same-block: p^2 * (p / (p+q))^2 + p^2 * (q / (p+q))^2  ...
        Cleaner: E[clustering] = P(edge between two random neighbours of v).
        For a random neighbour u of v in B_+: P(u in B_+) = p / (p+q).
        Two independent neighbours u, w: joint (block(u), block(w)) has probs
          (++): (p/(p+q))^2   edge prob = p
          (+-): (p/(p+q)) * (q/(p+q))   edge prob = q
          (-+): symmetric      edge prob = q
          (--): (q/(p+q))^2   edge prob = p
        Total: (p*(p^2 + q^2) + 2*q*p*q) / (p+q)^2 = (p^3 + p*q^2 + 2*p*q^2)/(p+q)^2
                                                    = (p^3 + 3*p*q^2) / (p+q)^2   ← wait, algebra
        Let me redo:
          E[clus] = [(p/(p+q))^2 * p + 2*(p/(p+q))*(q/(p+q)) * q + (q/(p+q))^2 * p]
                  = (p^3 + 2*p*q^2 + p*q^2) / (p+q)^2
                  = (p^3 + p*q^2 + 2*q*p*q) / (p+q)^2   -- typo
        Cleanly: p^3 / (p+q)^2 + 2*p*q^2/(p+q)^2 + p*q^2/(p+q)^2  ← wait
        Redo step by step:
          (p/(p+q))^2 * p     = p^3 / (p+q)^2
          2 * (p*q/(p+q)^2) * q = 2 * p*q^2 / (p+q)^2
          (q/(p+q))^2 * p     = p*q^2 / (p+q)^2
        Sum = (p^3 + 2*p*q^2 + p*q^2) / (p+q)^2 = (p^3 + 3*p*q^2) / (p+q)^2.
        By symmetry the +/- reversed vertex has the same expression with p, q swapped
        outside... actually the vertex is in B_+ so within/across probabilities are as
        above; the formula holds. Symmetric total over both blocks: identical.
        """
        num = self.p**3 + 3 * self.p * self.q**2
        return num / (self.p + self.q) ** 2


def sweep(n: int = 1000, grid: int = 41) -> dict:
    ps = np.linspace(0.01, 0.30, grid)
    qs = np.linspace(0.01, 0.30, grid)
    sigmas = np.linspace(0.0, 0.5, grid)

    results = []
    for p in ps:
        for q in qs:
            for sigma in sigmas:
                t = Theta(p=p, q=q, sigma=sigma)
                results.append(
                    {
                        "p": float(p),
                        "q": float(q),
                        "sigma": float(sigma),
                        "degree": t.expected_degree(n),
                        "homophily": t.expected_homophily(),
                        "clustering": t.expected_clustering(n),
                    }
                )
    return {"n": n, "grid": grid, "rows": results}


def constant_level_set(rows: list[dict], d_target: float, h_target: float, tol_d: float, tol_h: float) -> list[dict]:
    return [
        r
        for r in rows
        if abs(r["degree"] - d_target) < tol_d and abs(r["homophily"] - h_target) < tol_h
    ]


def report_level_set(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "clustering_min": None, "clustering_max": None, "range": None}
    cs = [r["clustering"] for r in rows]
    return {
        "count": len(rows),
        "clustering_min": float(min(cs)),
        "clustering_max": float(max(cs)),
        "range": float(max(cs) - min(cs)),
        "example_min": min(rows, key=lambda r: r["clustering"]),
        "example_max": max(rows, key=lambda r: r["clustering"]),
    }


if __name__ == "__main__":
    n = 1000
    sw = sweep(n=n, grid=41)
    print(f"Swept {len(sw['rows'])} parameter points at n = {n}.")

    scenarios = [
        {"d_target": 30.0, "h_target": 0.60, "tol_d": 1.5, "tol_h": 0.02},
        {"d_target": 60.0, "h_target": 0.70, "tol_d": 3.0, "tol_h": 0.02},
        {"d_target": 100.0, "h_target": 0.55, "tol_d": 3.0, "tol_h": 0.02},
    ]
    all_reports = []
    for sc in scenarios:
        subset = constant_level_set(
            sw["rows"], sc["d_target"], sc["h_target"], sc["tol_d"], sc["tol_h"]
        )
        rep = report_level_set(subset)
        print(f"\n(degree ~ {sc['d_target']}, homophily ~ {sc['h_target']}):")
        print(f"  {rep['count']} parameter points on the level set")
        if rep["count"] > 0:
            print(
                f"  clustering: min={rep['clustering_min']:.4f}, max={rep['clustering_max']:.4f}, "
                f"range={rep['range']:.4f}"
            )
            emn = rep["example_min"]
            emx = rep["example_max"]
            print(
                f"  min at (p={emn['p']:.3f}, q={emn['q']:.3f}, sigma={emn['sigma']:.3f}) "
                f"→ clus={emn['clustering']:.4f}, hom={emn['homophily']:.4f}, deg={emn['degree']:.2f}"
            )
            print(
                f"  max at (p={emx['p']:.3f}, q={emx['q']:.3f}, sigma={emx['sigma']:.3f}) "
                f"→ clus={emx['clustering']:.4f}, hom={emx['homophily']:.4f}, deg={emx['degree']:.2f}"
            )
        all_reports.append({"scenario": sc, "report": rep})

    with (RESULTS_DIR / "identifiability.json").open("w") as fh:
        json.dump({"n": n, "grid": sw["grid"], "reports": all_reports}, fh, indent=2)
