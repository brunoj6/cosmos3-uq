"""Command line entry point.

  python -m cosmos3_uq.run selftest
      CPU check of the estimators and the calibration code on a synthetic Jacobian.
  python -m cosmos3_uq.run record --manifest examples/demo.json --out outputs/nano [--model nano|edge]
                                  [--ids a b] [--probes 16] [--exact 512] [--salient 128] [--sigma-q 0.3]
      Rollout, raw and Hier diagonal, U maps and clip scores per manifest record (GPU).
  python -m cosmos3_uq.run fidelity --out outputs/nano
      Estimator error against the sampled exact cells of every record.
  python -m cosmos3_uq.run exact --manifest my_edge_manifest.json --out outputs/edge_exact [--probes 128]
  python -m cosmos3_uq.run exact-summary --out outputs/edge_exact
      Exact diagonal on every active cell (Edge, 256 tier) and the raw-vs-Hier comparison on it.
  python -m cosmos3_uq.run likelihood --manifest examples/demo.json --rollouts outputs/edge --out outputs/edge_lik
                                      [--model edge] [--steps 32] [--probes 16] [--ref 48] [--gauss 4]
  python -m cosmos3_uq.run likelihood-summary --out outputs/edge_lik
      Log-likelihood of each record's video (a rollout or a real recording) with per-cell maps, and raw vs Hier
      against the reference probes.
  python -m cosmos3_uq.run heatmap outputs/nano/av0_forward [more record dirs] [--estimator hier|raw] [--no-video]
      Clip-level heatmap strip (PNG) and overlay video (MP4) per record (U maps, or NLL maps for likelihood records).
  python -m cosmos3_uq.run calibrate --fit <record dirs> --test <record dirs> --eps 0.3 0.5 --out calib.json
      Isotonic recalibration of U against rollout error (records need gt_video in their manifest entry).

`record` writes, per manifest record, <out>/<id>/rollout.mp4, x0_pred.png (convention gate: x0 = x_sigma - sigma v
must decode to a clean scene), u_divergence.npz and u_divergence.json (settings, timings, clip scores, diagnostics).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .hier import hier_from_products
from .maps import channel_mean, clip_scores, trace_uncertainty, u_map

SALIENT_FRACTION = 0.15


def _resolve(path, base):
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def load_manifest(path):
    """A JSON list of records; relative paths resolve against the manifest's folder."""
    path = Path(path)
    records = json.load(open(path))
    for r in records:
        for k in ("image", "actions", "gt_video"):
            if k in r:
                r[k] = str(_resolve(r[k], path.parent))
    return records


def choose_exact_cells(v_active, n_uniform, n_salient, rng):
    """Uniform cells over the active latent plus salient cells, the top 15 percent by estimated column energy.

    The probe noise variance is (1/S) sum_{j != i} J_ji^2, so v ranks cells by column energy. Ranking and scoring
    with the same probes biases the salient stratum toward cells where the raw estimate happened to be noisy.
    """
    n = v_active.size
    idx_u = rng.choice(n, size=min(n_uniform, n), replace=False)
    order = np.argsort(v_active.ravel())[::-1][: max(1, int(SALIENT_FRACTION * n))]
    idx_s = rng.choice(order, size=min(n_salient, order.size), replace=False)
    return np.concatenate([idx_u, idx_s]), np.array(["uniform"] * idx_u.size + ["salient"] * idx_s.size)


def process_record(pipe, rec, out_dir, args):
    import torch
    from diffusers.utils import export_to_video

    from .divergence import exact_diagonal, hutchinson_vjp, trace_estimate
    from .heatmap import read_video
    from .rollout import decode_latents, encode_video_latents, make_closure, rollout

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    ro = rollout(pipe, rec, sigma_q=args.sigma_q, num_steps=args.steps, guidance=args.guidance,
                 seed=int(rec.get("seed", args.seed)))
    export_to_video(ro["frames"], str(out_dir / "rollout.mp4"), fps=int(rec.get("fps", 10)), macro_block_size=1)
    cap = ro.pop("cap")
    x = cap.x
    v_fn, mask, noisy = make_closure(pipe, cap)
    sigma = ro["sigma"]

    # Convention gate: the endpoint prediction at the captured step must decode to a clean scene.
    with torch.no_grad():
        v = v_fn(x)
        x0 = x - sigma * v
    frames0 = decode_latents(pipe, x0)
    frames0[len(frames0) // 2].save(out_dir / "x0_pred.png")

    # Adjoint Hutchinson probes on the active cells, one forward and S backward passes.
    active = mask.expand_as(x).contiguous()
    q, probe_seconds = hutchinson_vjp(v_fn, x, args.probes, seed=args.probe_seed, active=active)
    q = q[:, 0].numpy()                                   # (S, C, T, H, W).
    C, T, H, W = q.shape[1:]
    qa = q[:, :, noisy]                                   # Active frames only.
    # The map uses the first map_probes probes; every probe is stored so that fidelity can read nested budgets.
    est = hier_from_products(qa[: args.map_probes], pooled="covariance")
    d_raw, v_noise, d_hier = est["d_raw"], est["v"], est["d_hier"]

    # Exact diagonal on sampled cells (one backward per cell).
    rng = np.random.default_rng(args.probe_seed)
    if args.exact > 0:
        exact_idx, exact_kind = choose_exact_cells(v_noise, args.exact, args.salient, rng)
    else:
        exact_idx, exact_kind = np.zeros(0, int), np.zeros(0, str)
    exact_val, exact_seconds = torch.zeros(0), 0.0
    if exact_idx.size:
        c_i, f_i, h_i, w_i = np.unravel_index(exact_idx, qa.shape[1:])
        flat_full = np.ravel_multi_index((c_i, np.asarray(noisy)[f_i], h_i, w_i), (C, T, H, W))
        exact_val, exact_seconds = exact_diagonal(v_fn, x, flat_full.tolist())

    # Maps and clip scores at the captured sigma, for Hier and for the raw baseline.
    u_hier, clamp_hier = u_map(d_hier, sigma)
    u_raw, clamp_raw = u_map(d_raw, sigma)
    m_hier, m_raw = channel_mean(u_hier), channel_mean(u_raw)
    scores = {"hier": clip_scores(m_hier), "raw": clip_scores(m_raw)}
    n_active = qa[0].size
    trace = trace_estimate(torch.from_numpy(qa))
    u_trace = trace_uncertainty(trace, n_active, sigma)

    extra = {"latents_final": cap.final[0][:, noisy].cpu().numpy().astype(np.float32)}
    if rec.get("gt_video"):
        frames = read_video(rec["gt_video"])
        z = encode_video_latents(pipe, frames, ro["chunk_size"], int(rec.get("resolution_tier", 480)))
        if tuple(z.shape) != tuple(cap.final.shape):
            raise ValueError(f"[{rec['id']}] real-future latent {tuple(z.shape)} != rollout latent {tuple(cap.final.shape)}")
        extra["latents_gt"] = z[0][:, noisy].cpu().numpy().astype(np.float32)

    np.savez_compressed(
        out_dir / "u_divergence.npz",
        x_sigma=x[0].cpu().numpy().astype(np.float32), sigma=np.float64(sigma), active_frames=np.asarray(noisy),
        d_raw=d_raw.astype(np.float32), d_hier=d_hier.astype(np.float32),
        noise_var=v_noise.astype(np.float32), prior=est["p"].astype(np.float32),
        u_hier=u_hier.astype(np.float32), u_raw=u_raw.astype(np.float32),
        map_hier=m_hier.astype(np.float32), map_raw=m_raw.astype(np.float32), probes=qa.astype(np.float32),
        exact_index_active=exact_idx, exact_kind=exact_kind, exact_value=exact_val.numpy(), **extra,
    )
    info = {
        **{k: v for k, v in ro.items() if k != "frames"},
        "record": rec, "model": args.model,
        "latent_shape": [C, T, H, W], "active_frames": noisy, "n_active_cells": int(n_active),
        "product": "vjp", "probe_dist": "rademacher", "num_probes": args.probes, "map_probes": args.map_probes,
        "probe_seed": args.probe_seed, "pooled_var": "covariance",
        "clamp_fraction": {"hier": clamp_hier, "raw": clamp_raw}, "clip_scores": scores,
        "trace_diag_sum": trace, "u_trace": u_trace, "hier_diag": est["diag"],
        "n_exact": int(exact_idx.size), "probe_seconds": probe_seconds, "exact_seconds": exact_seconds,
        "has_gt": "latents_gt" in extra,
        "gpu": torch.cuda.get_device_name(0), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "diffusers": __import__("diffusers").__version__, "torch": torch.__version__,
    }
    (out_dir / "u_divergence.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"[{rec['id']}] sigma {sigma:.4f} (step {ro['step_index']}) latent {C}x{T}x{H}x{W} rollout "
          f"{ro['rollout_seconds']:.0f}s probes {probe_seconds:.0f}s exact {exact_seconds:.0f}s | "
          f"A_hier {scores['hier']['u_smooth3_q99']:.4f} A_raw {scores['raw']['u_smooth3_q99']:.4f} "
          f"clamp {clamp_hier:.3f}/{clamp_raw:.3f} peak {info['peak_mem_gb']:.1f} GB", flush=True)
    del q, v, x0
    torch.cuda.empty_cache()


def cmd_record(args):
    import torch

    from .rollout import load_pipe

    records = load_manifest(args.manifest)
    if args.ids:
        records = [r for r in records if r["id"] in args.ids]
    t0 = time.time()
    pipe = load_pipe(args.model)
    print(f"loaded {args.model} in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}", flush=True)
    out_root = Path(args.out)
    for rec in records:
        if (out_root / rec["id"] / "u_divergence.json").exists() and not args.force:
            print(f"[{rec['id']}] exists, skipping")
            continue
        process_record(pipe, rec, out_root / rec["id"], args)


def cmd_selftest(args):
    """Synthetic Jacobian with channel/frame/location structure and heavy-tailed off-diagonal mass, probed with exact
    adjoint products. Checks the z-gate, the estimator ordering, and the calibration code."""
    from scipy.stats import spearmanr

    from .calibration import Isotonic, auroc, reliability
    from .hier import ESTIMATORS

    rng = np.random.default_rng(0)
    C, F, H, W, S = 8, 3, 10, 12, 16
    n = C * F * H * W
    chan = rng.normal(0, 1.0, (C, 1, 1, 1))
    loc = np.cumsum(rng.normal(0, 0.3, (1, F, H, W)), axis=2)
    true = -3 + chan + loc + 0.2 * rng.normal(size=(C, F, H, W))
    col_scale = np.exp(1.2 * rng.normal(size=n)) * 0.025
    J = rng.normal(size=(n, n)) * col_scale[None, :]
    J[np.arange(n), np.arange(n)] = true.ravel()
    eps = rng.choice([-1.0, 1.0], size=(S, n))
    q = (eps * (eps @ J)).reshape(S, C, F, H, W)                    # eps (x) (J^T eps).
    est = hier_from_products(q)
    z = (est["d_raw"] - true) / np.sqrt(est["v"])
    print(f"z-gate raw: mean {z.mean():.3f} std {z.std():.3f} (expect ~0, ~1)")
    for name, f in ESTIMATORS.items():
        d = f(q)
        rel = np.linalg.norm(d - true) / np.linalg.norm(true)
        rho = spearmanr(d.ravel(), true.ravel()).correlation
        print(f"{name:8s} rel {rel:.4f} rho {rho:.4f}")
    u, clamp = u_map(est["d_hier"], 0.26)
    print("clip scores at sigma 0.26:", {k: round(v, 4) for k, v in clip_scores(channel_mean(u)).items()},
          "clamp", round(clamp, 3))

    # Calibration: Gaussian errors with known per-cell variance; isotonic on -U should be calibrated.
    u_cal = np.exp(rng.normal(-2, 1, 200_000))
    err = np.abs(rng.normal(size=u_cal.size) * np.sqrt(2 * u_cal))
    acc = (err <= 0.3).astype(float)
    iso = Isotonic().fit(-u_cal[:100_000], acc[:100_000])
    rel = reliability(iso(-u_cal[100_000:]), acc[100_000:])
    print(f"isotonic on held-out cells: ECE {rel['ece']:.4f} (expect < 0.01), "
          f"AUROC {auroc(-u_cal[100_000:], acc[100_000:]):.3f}")

    # Likelihood: probability-flow ODE + trace integral on Gaussian data must match the exact log density.
    from .likelihood import analytic_check

    print("likelihood integrator on Gaussian data:")
    worst = analytic_check()
    print(f"worst per-dim error on the finest grid {worst:.2e} (expect < 1e-2; coarse grids err more on low-variance data)")


def cmd_heatmap(args):
    from .heatmap import render_record

    for d in args.records:
        for p in render_record(d, estimator=args.estimator, fps=args.fps, video=not args.no_video):
            print("wrote", p)


def cmd_calibrate(args):
    from .calibration import evaluate, save_fits

    res = evaluate(args.fit, args.test, args.eps, estimator=args.estimator, rule=args.rule, num_bins=args.bins)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    save_fits(res, Path(args.out).with_suffix(".fits.json"))
    print(f"{'eps':>5} {'acc':>6} {'AUROC':>6} | {'raw conf':>8} {'ECE':>6} {'MCE':>6} | {'iso conf':>8} {'ECE':>6} {'MCE':>6}")
    for eps, e in res["test"].items():
        print(f"{eps:>5} {e['mean_accuracy']:6.3f} {e['auroc'] or float('nan'):6.3f} | "
              f"{e['raw']['mean_confidence']:8.3f} {e['raw']['ece_pooled']:6.4f} {e['raw']['mce_pooled']:6.3f} | "
              f"{e['iso']['mean_confidence']:8.3f} {e['iso']['ece_pooled']:6.4f} {e['iso']['mce_pooled']:6.3f}")
    print("wrote", args.out)


def main():
    from . import exact

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", default="outputs/nano")
    r.add_argument("--ids", nargs="*")
    r.add_argument("--model", default="nano", help="nano, edge, or a Hugging Face model id")
    r.add_argument("--probes", type=int, default=16, help="probes to draw and store")
    r.add_argument("--map-probes", type=int, default=16, help="probes used for the maps and clip scores")
    r.add_argument("--exact", type=int, default=512, help="uniform exact cells per record (0 to skip)")
    r.add_argument("--salient", type=int, default=128, help="salient exact cells per record")
    r.add_argument("--sigma-q", type=float, default=0.3)
    r.add_argument("--steps", type=int, default=30)
    r.add_argument("--guidance", type=float, default=1.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--probe-seed", type=int, default=0)
    r.add_argument("--force", action="store_true")
    f = sub.add_parser("fidelity")
    f.add_argument("--out", default="outputs/nano")
    e = sub.add_parser("exact")
    e.add_argument("--manifest", required=True)
    e.add_argument("--out", default="outputs/edge_exact")
    e.add_argument("--ids", nargs="*")
    e.add_argument("--model", default="edge")
    e.add_argument("--probes", type=int, default=128)
    e.add_argument("--probe-seed", type=int, default=0)
    e.add_argument("--sigma-q", type=float, default=0.3)
    e.add_argument("--exact-batch", type=int, default=16)
    e.add_argument("--exact-limit", type=int, default=0, help="smoke test: only the first N active cells")
    e.add_argument("--force", action="store_true")
    es = sub.add_parser("exact-summary")
    es.add_argument("--out", default="outputs/edge_exact")
    h = sub.add_parser("heatmap")
    h.add_argument("records", nargs="+")
    h.add_argument("--estimator", default="hier", choices=["hier", "raw"])
    h.add_argument("--fps", type=int, default=10)
    h.add_argument("--no-video", action="store_true")
    c = sub.add_parser("calibrate")
    c.add_argument("--fit", nargs="+", required=True, help="record dirs of the calibration split")
    c.add_argument("--test", nargs="+", required=True, help="record dirs to score")
    c.add_argument("--eps", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.8])
    c.add_argument("--estimator", default="hier", choices=["hier", "raw"])
    c.add_argument("--rule", default="two_draw", choices=["two_draw", "velocity"])
    c.add_argument("--bins", type=int, default=20)
    c.add_argument("--out", default="calibration.json")
    lk = sub.add_parser("likelihood")
    lk.add_argument("--manifest", required=True, help="records with image, actions and video (or use --rollouts)")
    lk.add_argument("--rollouts", default=None, help="score <rollouts>/<id>/rollout.mp4 for records without video")
    lk.add_argument("--out", default="outputs/likelihood")
    lk.add_argument("--ids", nargs="*")
    lk.add_argument("--model", default="edge", help="nano, edge, or a Hugging Face model id")
    lk.add_argument("--steps", type=int, help="Heun steps on the logit-uniform sigma grid (default 32)")
    lk.add_argument("--sigma-min", dest="sigma_min", type=float, help="start of the grid (default 1e-3)")
    lk.add_argument("--probes", type=int, help="Rademacher probes per step for raw and Hier (default 16)")
    lk.add_argument("--ref", type=int, help="extra reference probes per step, 0 to skip (default 48)")
    lk.add_argument("--gauss", type=int, help="FFJORD Gaussian probes, one estimate each (default 4)")
    lk.add_argument("--batch", type=int, help="probes per batched backward pass (default 8)")
    lk.add_argument("--seed", type=int)
    lk.add_argument("--force", action="store_true")
    ls = sub.add_parser("likelihood-summary")
    ls.add_argument("--out", default="outputs/likelihood")
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "fidelity":
        from .fidelity import summarize

        summarize(args.out)
        return
    if args.cmd == "likelihood-summary":
        from .likelihood import summarize

        summarize(args.out)
        return
    if args.cmd == "likelihood":
        from .likelihood import cmd_run

        cmd_run(args)
        return
    if args.cmd == "exact":
        from .rollout import resolve_model

        args.model = resolve_model(args.model)
    {"record": cmd_record, "exact": exact.cmd_record, "exact-summary": exact.cmd_summary, "heatmap": cmd_heatmap,
     "calibrate": cmd_calibrate, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
