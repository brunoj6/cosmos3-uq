"""Estimator fidelity against exact diagonal cells sampled from each record.

Each record stores its per-probe products at the largest budget and exact J_ii on two strata: cells drawn uniformly
over the active latent and salient cells (the top 15 percent by estimated column energy). Every budget S' is
evaluated on the first S' stored probes, so budgets are nested. Metrics per record and stratum: relative l2 error
||d_hat - d|| / ||d||, Spearman rho, and the z-gate of the raw estimate z = (d_hat - d) / sqrt(v), which should be
about N(0, 1) when the probe noise variance is estimated correctly. Records get equal weight; win counts compare Hier with raw per record.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .hier import ESTIMATORS, hier_from_products

BUDGETS = (4, 8, 16, 32, 64)


def evaluate_record(npz_path, budgets=BUDGETS):
    z = np.load(npz_path)
    q = z["probes"].astype(np.float64)                      # (S, C, F, H, W), active frames only.
    idx = z["exact_index_active"]                           # Flat indices into (C, F, H, W).
    kinds = z["exact_kind"]
    d_true = z["exact_value"].astype(np.float64)
    S = q.shape[0]
    rows = []
    for budget in [b for b in budgets if b <= S]:
        qb = q[:budget]
        ests = {name: f(qb).reshape(-1) for name, f in ESTIMATORS.items()}
        v = hier_from_products(qb)["v"].reshape(-1)
        for stratum in np.unique(kinds):
            sel = kinds == stratum
            t = d_true[sel]
            for name, dhat in ests.items():
                e = dhat[idx[sel]]
                rel = float(np.linalg.norm(e - t) / max(np.linalg.norm(t), 1e-12))
                rho = float(spearmanr(e, t).correlation) if sel.sum() > 2 else float("nan")
                row = {"S": budget, "stratum": str(stratum), "estimator": name, "rel": rel, "rho": rho,
                       "median_abs": float(np.median(np.abs(e - t)))}
                if name == "raw":
                    zz = (e - t) / np.sqrt(v[idx[sel]])
                    row.update({"z_mean": float(zz.mean()), "z_std": float(zz.std(ddof=1))})
                rows.append(row)
    return rows


def summarize(out_root, budgets=BUDGETS):
    """Aggregate every record under out_root/*/u_divergence.npz into fidelity.json and a printed table."""
    out_root = Path(out_root)
    per_record = {}
    for npz in sorted(out_root.glob("*/u_divergence.npz")):
        if "probes" in np.load(npz).files and np.load(npz)["exact_index_active"].size:
            per_record[npz.parent.name] = evaluate_record(npz, budgets)
    keys = sorted({(r["S"], r["stratum"], r["estimator"]) for rows in per_record.values() for r in rows})
    table = []
    for S, stratum, est in keys:
        vals = [r for rows in per_record.values() for r in rows if (r["S"], r["stratum"], r["estimator"]) == (S, stratum, est)]
        entry = {"S": S, "stratum": stratum, "estimator": est, "n_records": len(vals),
                 "rel": float(np.mean([r["rel"] for r in vals])), "rho": float(np.nanmean([r["rho"] for r in vals]))}
        if est == "raw":
            entry["z_mean"] = float(np.mean([r["z_mean"] for r in vals]))
            entry["z_std"] = float(np.mean([r["z_std"] for r in vals]))
        if est == "hier":
            raws = {rid: next(r for r in rows if (r["S"], r["stratum"], r["estimator"]) == (S, stratum, "raw")) for rid, rows in per_record.items()}
            entry["wins_vs_raw"] = int(sum(v["rel"] < raws[rid]["rel"] for rid, rows in per_record.items()
                                          for v in rows if (v["S"], v["stratum"], v["estimator"]) == (S, stratum, "hier")))
        table.append(entry)
    (out_root / "fidelity.json").write_text(json.dumps({"per_record": per_record, "table": table}, indent=1))
    print(f"{'S':>3} {'stratum':>8} {'estimator':>9} {'rel':>7} {'rho':>6}  extra")
    for e in table:
        extra = " ".join(f"{k}={e[k]:.3f}" if isinstance(e[k], float) else f"{k}={e[k]}" for k in ("z_mean", "z_std", "wins_vs_raw") if k in e)
        print(f"{e['S']:>3} {e['stratum']:>8} {e['estimator']:>9} {e['rel']:7.4f} {e['rho']:6.3f}  {extra} (n={e['n_records']})")
    return table
