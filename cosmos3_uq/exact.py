"""Exact Jacobian diagonal on every active cell of a small Cosmos3 configuration, scored against raw Hutchinson and Hier.

fidelity.py scores the estimators on sampled exact cells. Here the configuration is small enough (Cosmos3-Edge, 256
resolution tier, 21 frames = 1 conditioning + 5 denoised latent frames) that J_ii is computed for every active cell,
one adjoint row J^T e_i each, so the errors cover the whole latent. Each row is also squared and summed into the
column energies sum_{i != j} J_ij^2, which are the exact per-probe noise variances of the vjp estimate; the salient
stratum is the top 15 percent of cells by exact column energy. A record takes about 16 minutes on one A100.

Budgets are scored on disjoint blocks of the stored probes (128 probes give 32 blocks of 4, ..., 2 blocks of 64), and
the block errors are averaged within a record before records get equal weight.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .hier import ESTIMATORS, box_mean, hier_from_products
from .maps import channel_mean, clip_scores, u_map

SALIENT_FRACTION = 0.15
BUDGETS = (4, 8, 16, 32, 64)
N_UNIFORM = 512
ESTIMATORS_ALL = {**ESTIMATORS, "raw3x3": lambda q: box_mean(q.mean(0))}


def process_record(pipe, rec, out_dir, args):
    from diffusers.utils import export_to_video

    from .divergence import exact_rows, hutchinson_vjp
    from .rollout import decode_latents, make_closure, rollout

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    ro = rollout(pipe, rec, sigma_q=args.sigma_q, seed=int(rec.get("seed", 0)))
    export_to_video(ro["frames"], str(out_dir / "rollout.mp4"), fps=int(rec.get("fps", 10)), macro_block_size=1)
    cap = ro.pop("cap")
    x = cap.x
    v_fn, mask, noisy = make_closure(pipe, cap)
    sigma = ro["sigma"]

    # Convention gate: x0 = x_sigma - sigma v must decode to a clean scene.
    with torch.no_grad():
        x0 = x - sigma * v_fn(x)
    frames0 = decode_latents(pipe, x0)
    frames0[len(frames0) // 2].save(out_dir / "x0_pred.png")

    # Stored adjoint Hutchinson products on the active cells.
    active = mask.expand_as(x).contiguous()
    q, probe_seconds = hutchinson_vjp(v_fn, x, args.probes, seed=args.probe_seed, active=active)
    q = q[:, 0].numpy()                                   # (S, C, T, H, W).
    C, T, H, W = q.shape[1:]
    qa = q[:, :, noisy]                                   # Active frames only, (S, C, F, H, W).
    shape_a = qa.shape[1:]

    # Exact rows for every active cell, or the first exact_limit of them in a smoke test.
    c_i, f_i, h_i, w_i = np.unravel_index(np.arange(int(np.prod(shape_a))), shape_a)
    flat_full = np.ravel_multi_index((c_i, np.asarray(noisy)[f_i], h_i, w_i), (C, T, H, W))
    if args.exact_limit:
        flat_full = flat_full[: args.exact_limit]
    diag, col, exact_seconds, rows_per_pass = exact_rows(v_fn, x, flat_full, args.exact_batch)
    n_done = diag.size
    complete = n_done == flat_full.size and not args.exact_limit
    col_a = col.reshape(C, T, H, W)[:, noisy]             # Column energies of the active inputs.
    exact = np.full(int(np.prod(shape_a)), np.nan)
    exact[:n_done] = diag
    exact = exact.reshape(shape_a)
    col_off = col_a - np.nan_to_num(exact) ** 2           # sum_{i != j} J_ij^2, valid when complete.

    np.savez_compressed(
        out_dir / "exact_full.npz",
        x_sigma=x[0].cpu().numpy().astype(np.float32), probes=qa.astype(np.float32), active_frames=np.asarray(noisy),
        exact_diag=exact.astype(np.float64), col_off=col_off.astype(np.float64), complete=np.asarray(complete),
    )
    trace_exact = float(np.nansum(exact))
    trace_hutch = float(qa.reshape(qa.shape[0], -1).sum(1).mean())
    info = {
        **{k: v for k, v in ro.items() if k != "frames"}, "record": rec, "model": args.model,
        "latent_shape": [C, T, H, W], "active_frames": noisy, "n_active_cells": int(np.prod(shape_a)),
        "n_exact_rows": int(n_done), "complete": bool(complete), "rows_per_pass": rows_per_pass,
        "product": "vjp", "probe_dist": "rademacher", "num_probes": args.probes, "probe_seed": args.probe_seed,
        "trace_exact": trace_exact, "trace_hutchinson": trace_hutch,
        "trace_hutchinson_se": float(qa.reshape(qa.shape[0], -1).sum(1).std(ddof=1) / np.sqrt(qa.shape[0])),
        "probe_seconds": probe_seconds, "exact_seconds": exact_seconds,
        "gpu": torch.cuda.get_device_name(0), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "diffusers": __import__("diffusers").__version__, "torch": torch.__version__,
    }
    (out_dir / "exact_full.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"[{rec['id']}] sigma {sigma:.4f} latent {C}x{T}x{H}x{W} active {info['n_active_cells']} rollout "
          f"{ro['rollout_seconds']:.0f}s probes {probe_seconds:.0f}s exact {n_done} rows {exact_seconds:.0f}s "
          f"({rows_per_pass}/pass) trace exact {trace_exact:.1f} hutch {trace_hutch:.1f} peak {info['peak_mem_gb']:.1f} GB",
          flush=True)
    torch.cuda.empty_cache()


def cmd_record(args):
    from .rollout import load_pipe
    from .run import load_manifest

    records = load_manifest(args.manifest)
    if args.ids:
        records = [r for r in records if r["id"] in args.ids]
    t0 = time.time()
    pipe = load_pipe(args.model)
    print(f"loaded {args.model} in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}", flush=True)
    for rec in records:
        out_dir = Path(args.out) / rec["id"]
        if (out_dir / "exact_full.json").exists() and not args.force:
            print(f"[{rec['id']}] exists, skipping")
            continue
        process_record(pipe, rec, out_dir, args)


def _metrics(est, true, sel):
    from scipy.stats import spearmanr

    e, t = est[sel], true[sel]
    return {"rel": float(np.linalg.norm(e - t) / max(np.linalg.norm(t), 1e-12)),
            "rho": float(spearmanr(e, t).correlation)}


def evaluate_record(npz_path, sigma, budgets=BUDGETS):
    """Per-budget, per-estimator errors on every active cell, the salient stratum (top 15 percent by exact column
    energy), a 512-uniform-cell sample, and the channel-mean U map and clip score."""
    z = np.load(npz_path)
    q = z["probes"].astype(np.float64)
    true = z["exact_diag"]
    col_off = z["col_off"]
    S_max = q.shape[0]
    flat_true = true.ravel()
    n = flat_true.size
    order = np.argsort(col_off.ravel())[::-1]
    strata = {"all": np.ones(n, bool), "salient": np.zeros(n, bool), "uniform512": np.zeros(n, bool)}
    strata["salient"][order[: int(SALIENT_FRACTION * n)]] = True
    strata["quiet"] = ~strata["salient"]
    strata["uniform512"][np.random.default_rng(0).choice(n, N_UNIFORM, replace=False)] = True

    u_true, _ = u_map(true, sigma)
    m_true = channel_mean(u_true)
    a_true = clip_scores(m_true)["u_smooth3_q99"]
    rows = []
    for S in [b for b in budgets if b <= S_max]:
        acc = {}
        for rep in range(S_max // S):
            qb = q[rep * S: (rep + 1) * S]
            est_all = {name: f(qb) for name, f in ESTIMATORS_ALL.items()}
            v_hat = hier_from_products(qb)["v"].ravel()
            for name, d in est_all.items():
                flat = d.ravel()
                for stratum, sel in strata.items():
                    acc.setdefault((name, stratum), []).append(_metrics(flat, flat_true, sel))
                m = channel_mean(u_map(d, sigma)[0])
                a = clip_scores(m)["u_smooth3_q99"]
                acc.setdefault((name, "map"), []).append({
                    "rel": float(np.linalg.norm(m - m_true) / np.linalg.norm(m_true)),
                    "rho": float(np.corrcoef(m.ravel(), m_true.ravel())[0, 1]),
                    "clip_rel": float(abs(a - a_true) / a_true)})
            # z-gates of the raw estimate, with the estimated noise variance (what Hier uses) and with the exact
            # variance col_off / S.
            err = qb.mean(0).ravel() - flat_true
            z_hat = err / np.sqrt(np.maximum(v_hat, 1e-12))
            z_ex = err / np.sqrt(np.maximum(col_off.ravel() / S, 1e-12))
            acc.setdefault(("raw", "zgate"), []).append({"z_hat_mean": float(z_hat.mean()), "z_hat_std": float(z_hat.std()),
                                                         "z_exact_mean": float(z_ex.mean()), "z_exact_std": float(z_ex.std())})
        for (name, stratum), vals in acc.items():
            row = {"S": S, "estimator": name, "stratum": stratum, "reps": len(vals)}
            row.update({k: float(np.mean([v[k] for v in vals])) for k in vals[0]})
            rows.append(row)
    stats = {"n_cells": int(n), "exact_median": float(np.median(flat_true)), "exact_max": float(flat_true.max()),
             "exact_min": float(flat_true.min()), "col_off_median": float(np.median(col_off)),
             "noise_to_signal_salient": float(np.median(col_off.ravel()[strata["salient"]] / flat_true[strata["salient"]] ** 2)),
             "noise_to_signal_quiet": float(np.median(col_off.ravel()[strata["quiet"]] / flat_true[strata["quiet"]] ** 2)),
             "clip_score_exact": a_true}
    return rows, stats


def cmd_summary(args):
    out_root = Path(args.out)
    per_record, stats = {}, {}
    for js in sorted(out_root.glob("*/exact_full.json")):
        info = json.loads(js.read_text())
        if not info["complete"]:
            print(f"[{js.parent.name}] incomplete exact diagonal, skipped")
            continue
        rows, st = evaluate_record(js.parent / "exact_full.npz", info["sigma"])
        st.update({k: info[k] for k in ("trace_exact", "trace_hutchinson", "trace_hutchinson_se", "sigma",
                                        "exact_seconds", "probe_seconds", "rollout_seconds", "gpu", "rows_per_pass")})
        per_record[js.parent.name], stats[js.parent.name] = rows, st
        print(f"[{js.parent.name}] scored", flush=True)
    keys = sorted({(r["S"], r["stratum"], r["estimator"]) for rows in per_record.values() for r in rows})
    table = []
    for S, stratum, est in keys:
        vals = {rid: next(r for r in rows if (r["S"], r["stratum"], r["estimator"]) == (S, stratum, est))
                for rid, rows in per_record.items()}
        entry = {"S": S, "stratum": stratum, "estimator": est, "n_records": len(vals)}
        for k in [k for k in next(iter(vals.values())) if k not in ("S", "stratum", "estimator", "reps")]:
            entry[k] = float(np.mean([v[k] for v in vals.values()]))
        if est != "raw" and stratum != "zgate":
            raw = {rid: next(r for r in rows if (r["S"], r["stratum"], r["estimator"]) == (S, stratum, "raw"))
                   for rid, rows in per_record.items()}
            entry["wins_vs_raw"] = int(sum(vals[rid]["rel"] < raw[rid]["rel"] for rid in vals))
        table.append(entry)
    (out_root / "summary.json").write_text(json.dumps({"table": table, "per_record": per_record, "stats": stats}, indent=1))
    print(f"{'S':>3} {'stratum':>10} {'estimator':>9} {'rel':>7} {'rho':>6}  extra")
    for e in table:
        extra = " ".join(f"{k}={e[k]:.3f}" if isinstance(e[k], float) else f"{k}={e[k]}"
                         for k in e if k not in ("S", "stratum", "estimator", "rel", "rho", "n_records"))
        print(f"{e['S']:>3} {e['stratum']:>10} {e['estimator']:>9} {e.get('rel', float('nan')):7.4f} "
              f"{e.get('rho', float('nan')):6.3f}  {extra} (n={e['n_records']})")
