"""Log-likelihood of a video under a Cosmos3 model, with per-cell maps and several trace estimators.

A flow-matching model is a continuous normalizing flow. In the Cosmos convention x_sigma = (1 - sigma) x0 + sigma eps,
v = eps - x0 and dx/dsigma = v(x, sigma), so the instantaneous change of variables gives

    log p(x0) = log N(x1; 0, I) + int_0^1 tr(dv/dx)(x_sigma, sigma) dsigma,

with x_sigma the probability-flow ODE path from the data point x0 to the noise point x1. The trace is estimated at every
grid point along one shared path, in up to four ways:

  ffjord   one Gaussian probe eps ~ N(0, I) per solve, held fixed over the whole path, tr ~ eps^T J eps (the standard
           FFJORD estimator). G independent probes give G independent likelihood estimates.
  raw      vjp Hutchinson diagonal with S Rademacher probes, fresh at every step, summed.
  hier     the same S probes shrunk per cell by Hier, summed.
  ref      R further disjoint Rademacher probes, a reference for the trace and the per-cell diagonal.

The prior term and the trace both split over latent cells, so prior_i + int J_ii dsigma is an additive decomposition of
log p(x0): a per-cell log-likelihood map. Only the diagonal estimators (raw, Hier) give it.

What is scored: the video is encoded with the pipeline's own VAE path into x0 (the predicted latent frames). Latent frame
0 is the clean conditioning frame from the record's image, and the record's actions condition the transformer, so p is
the model's conditional density of the video given its first frame and actions. The video can be a rollout of the model
(its own samples) or a real recording.

Integration runs on a logit-uniform sigma grid from sigma_min to 1 (dense at both ends, where the trace changes fastest)
with Heun steps; the trace integral uses the trapezoid rule over the grid points. Probes are taken at the grid points
only, not at the Heun predictor points.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

LOG2PI = math.log(2 * math.pi)
DEFAULTS = {"steps": 32, "sigma_min": 1e-3, "sigma_eval_max": 0.999, "gauss": 4, "probes": 16, "ref": 48,
            "batch": 8, "seed": 0}


class _Abort(Exception):
    pass


def logit_sigma_grid(steps, sigma_min):
    """Logit-uniform grid from sigma_min to 1 - sigma_min, with the last point moved to 1 (the prior's sigma)."""
    lo = math.log(sigma_min / (1 - sigma_min))
    s = 1.0 / (1.0 + np.exp(-np.linspace(lo, -lo, steps + 1)))
    s[0], s[-1] = sigma_min, 1.0
    return s


def trapezoid_weights(s):
    w = np.zeros_like(s)
    ds = np.diff(s)
    w[:-1] += ds / 2
    w[1:] += ds / 2
    return w


def analytic_check(scales=(0.1, 0.5, 2.0), steps_list=(16, 32, 64)):
    """For Gaussian data N(0, s^2 I) the exact velocity is linear, J = (sigma - (1 - sigma) s^2) / a^2 with
    a^2 = (1 - sigma)^2 s^2 + sigma^2. Integrating the ODE and the trace on this module's grid must return the exact
    log N(x0; 0, s^2 I) up to grid error, which falls as steps grow. Returns the worst per-dimension error on the
    finest grid."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for scale in scales:
        for steps in steps_list:
            x = rng.normal(size=1000) * scale
            s = logit_sigma_grid(steps, 1e-3)
            w = trapezoid_weights(s)

            def jac(sig):
                a2 = (1 - sig) ** 2 * scale ** 2 + sig ** 2
                return (sig - (1 - sig) * scale ** 2) / a2

            x_sig = x * ((1 - s[0]) ** 2 * scale ** 2 + s[0] ** 2) ** 0.5 / scale
            integ = 0.0
            for k in range(len(s)):
                integ += w[k] * jac(s[k]) * x.size
                if k < len(s) - 1:
                    v0 = jac(s[k]) * x_sig
                    xp = x_sig + (s[k + 1] - s[k]) * v0
                    x_sig = x_sig + (s[k + 1] - s[k]) / 2 * (v0 + jac(s[k + 1]) * xp)
            logp = (-0.5 * x_sig ** 2 - 0.5 * LOG2PI).sum() + integ
            exact = (-0.5 * (x / scale) ** 2 - 0.5 * LOG2PI - math.log(scale)).sum()
            err = (logp - exact) / x.size
            if steps == max(steps_list):
                worst = max(worst, abs(err))
            print(f"  data sd {scale} steps {steps}: ODE log p {logp:.3f} exact {exact:.3f} (per dim {err:+.2e})")
    return worst


def pack(pipe, record, num_steps=30):
    """Static transformer inputs for a record (text, action and position packing), the pipeline's initial latents
    (clean conditioning frame + noise) and the noisy latent frame indices, from a pipeline call aborted at its first
    transformer call. Also returns the transformer-timestep per unit sigma."""
    import torch

    from .rollout import AV_PROMPT, action_condition, sigma_grid

    sigmas, timesteps = sigma_grid(pipe, num_steps)
    t_scale = float(timesteps[0] / sigmas[0])
    store = {}

    def grab(*a, **kw):
        store["kwargs"] = {k: v for k, v in kw.items() if k not in ("vision_tokens", "vision_timesteps", "return_dict")}
        store["x_init"] = kw["vision_tokens"][0].detach().float().clone()
        store["timestep0"] = float(kw["vision_timesteps"].reshape(-1)[0])
        raise _Abort

    action, _, chunk = action_condition(record)
    pipe.transformer.forward = grab
    try:
        pipe(prompt=record.get("prompt", AV_PROMPT), action=action, fps=float(record.get("fps", 10)),
             num_inference_steps=num_steps, guidance_scale=1.0, use_system_prompt=False,
             generator=torch.Generator(device=pipe._get_execution_device()).manual_seed(0), enable_safety_check=False)
    except _Abort:
        pass
    finally:
        del pipe.transformer.forward
    if abs(store["timestep0"] - sigmas[0] * t_scale) > 1e-3 * store["timestep0"]:
        raise RuntimeError(f"timestep convention: step-0 timestep {store['timestep0']} vs sigma {sigmas[0]}")
    kw = store["kwargs"]
    return {"kwargs": kw, "x_init": store["x_init"], "noisy": kw["vision_noisy_frame_indexes"][0].tolist(),
            "shape": tuple(store["x_init"].shape), "t_scale": t_scale, "chunk": chunk}


def batched_products(v, x, probes, batch):
    """eps * (J^T eps) for a stack of probes [P, *x.shape] through one retained graph, `batch` cotangents per pass."""
    import torch

    out = []
    for i in range(0, probes.shape[0], batch):
        e = probes[i: i + batch]
        if e.shape[0] == 1:
            (g,) = torch.autograd.grad(v, x, grad_outputs=e[0], retain_graph=True)
            g = g[None]
        else:
            (g,) = torch.autograd.grad(v, x, grad_outputs=e, retain_graph=True, is_grads_batched=True)
        out.append(e * g.float())
    return torch.cat(out)


def log_likelihood(pipe, pk, z0, cfg, log_every=8):
    """Integrate log p(z0) for latents z0 [1, C, T, h, w] (frame 0 is replaced by the clean conditioning frame).

    Returns (meta dict, maps dict of per-cell arrays (C, F, h, w) over the predicted frames).
    """
    import torch

    from .hier import hier_from_products

    t_start = time.time()
    tf = pipe.transformer
    device = z0.device
    kw = pk["kwargs"]
    noisy = pk["noisy"]
    n_tok = int(kw["vision_mse_loss_indexes"].numel())
    x = z0.clone().float()
    x[:, :, 0] = pk["x_init"][:, :, 0]
    mask = torch.zeros(x.shape[2], device=device)
    mask[noisy] = 1.0
    mask = mask.view(1, 1, -1, 1, 1)
    shape_a = (x.shape[1], len(noisy)) + tuple(x.shape[3:])
    D = int(np.prod(shape_a))

    def v_fn(xx, sig):
        ts = torch.full((n_tok,), float(sig) * pk["t_scale"], device=device)
        preds, _, _ = tf(vision_tokens=[xx.to(tf.dtype)], vision_timesteps=ts, return_dict=False, **kw)
        return preds[0].float().reshape(x.shape) * mask

    s = logit_sigma_grid(int(cfg["steps"]), float(cfg["sigma_min"]))
    w = trapezoid_weights(s)
    G, S, R, B = int(cfg["gauss"]), int(cfg["probes"]), int(cfg["ref"]), int(cfg["batch"])
    if S < 2:
        raise ValueError("Hier needs at least two probes per step")
    gen = torch.Generator(device=device).manual_seed(int(cfg["seed"]))
    gauss = torch.randn((G,) + tuple(x.shape), device=device, generator=gen) * mask      # Fixed over the path.
    x = (1 - s[0]) * x * mask + x * (1 - mask)                                             # Start point, x0 to O(sigma_min).

    tr = {"ffjord": np.zeros((len(s), G)), "rad": np.zeros((len(s), S + R)), "raw": np.zeros(len(s)),
          "hier": np.zeros(len(s))}
    integ = {k: np.zeros(shape_a, np.float64) for k in (("raw", "hier", "ref") if R else ("raw", "hier"))}
    hier_diag = []
    for k, sig in enumerate(s):
        sig_eval = min(float(sig), float(cfg["sigma_eval_max"]))
        xg = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            v = v_fn(xg, sig_eval)
            pg = batched_products(v, xg, gauss, B) if G else None
            gen_r = torch.Generator(device=device).manual_seed(int(cfg["seed"]) * 100003 + 1000 + k)
            rad = (torch.randint(0, 2, (S + R,) + tuple(x.shape), device=device, generator=gen_r).float() * 2 - 1) * mask
            pr = batched_products(v, xg, rad, B)
        v = v.detach()
        del xg, rad
        if G:
            tr["ffjord"][k] = pg.sum(dim=tuple(range(1, pg.ndim))).double().cpu().numpy()
        q = pr[:, 0][:, :, noisy].double().cpu().numpy()                                  # (S + R, C, F, h, w).
        del pr, pg
        tr["rad"][k] = q.reshape(S + R, -1).sum(1)
        est = hier_from_products(q[:S], pooled="covariance")
        tr["raw"][k] = est["d_raw"].sum()
        tr["hier"][k] = est["d_hier"].sum()
        integ["raw"] += w[k] * est["d_raw"]
        integ["hier"] += w[k] * est["d_hier"]
        if R:
            integ["ref"] += w[k] * q[S:].mean(0)
        hier_diag.append({k2: est["diag"][k2] for k2 in ("s2", "s2w", "w_mean")})
        # Heun step along the probability-flow ODE.
        if k < len(s) - 1:
            ds = float(s[k + 1] - sig)
            xp = x + ds * v
            with torch.no_grad():
                vp = v_fn(xp, min(float(s[k + 1]), float(cfg["sigma_eval_max"])))
            x = x + ds / 2 * (v + vp)
        if log_every and k % log_every == 0:
            print(f"  [{k + 1}/{len(s)}] sigma {sig:.4f}: tr raw {tr['raw'][k]:.1f} hier {tr['hier'][k]:.1f} "
                  f"ffjord {tr['ffjord'][k].round(0).tolist()} | {time.time() - t_start:.0f} s", flush=True)

    x1 = x[:, :, noisy][0].double().cpu().numpy()
    prior_cell = -0.5 * x1 ** 2 - 0.5 * LOG2PI
    prior = float(prior_cell.sum())
    trace_integral = {"ffjord": (w[:, None] * tr["ffjord"]).sum(0).tolist(), "raw": float(w @ tr["raw"]),
                      "hier": float(w @ tr["hier"])}
    trace = {"ffjord": tr["ffjord"].tolist(), "raw": tr["raw"].tolist(), "hier": tr["hier"].tolist()}
    if R:
        tr_ref = tr["rad"][:, S:].mean(1)
        se_ref = tr["rad"][:, S:].std(1, ddof=1) / math.sqrt(R) if R > 1 else np.zeros(len(s))
        trace_integral.update({"ref": float(w @ tr_ref), "ref_se": float(math.sqrt(((w * se_ref) ** 2).sum()))})
        trace.update({"ref": tr_ref.tolist(), "ref_se": se_ref.tolist()})
    logp = {"ffjord": [prior + t for t in trace_integral["ffjord"]],
            **{k: prior + trace_integral[k] for k in integ}}
    meta = {"D": D, "latent_shape": list(shape_a), "sigmas": s.tolist(), "weights": w.tolist(), "prior": prior,
            "trace": trace, "trace_integral": trace_integral, "logp": logp,
            "nll_per_dim": {k: (-np.asarray(v) / D).tolist() for k, v in logp.items()},
            "hier_diag": hier_diag, "cfg": dict(cfg), "x1_std": float(x1.std()), "x1_mean": float(x1.mean()),
            "seconds": time.time() - t_start}
    if R:
        meta["cell_corr_with_ref"] = {k: float(np.corrcoef(integ[k].ravel(), integ["ref"].ravel())[0, 1])
                                      for k in ("raw", "hier")}
    maps = {"prior": prior_cell, **{f"int_{k}": v for k, v in integ.items()}}
    return meta, maps


def process_record(pipe, rec, out_dir, cfg):
    """Score the record's video and write lik.json and lik_maps.npz (per-cell prior and trace integrals, plus
    channel-mean per-cell NLL maps nll_map_<estimator> (F, h, w) for heatmaps)."""
    import torch

    from .heatmap import read_video
    from .rollout import encode_video_latents

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    pk = pack(pipe, rec)
    frames = read_video(rec["video"])
    z0 = encode_video_latents(pipe, frames, pk["chunk"], int(rec.get("resolution_tier", 480)))
    if tuple(z0.shape) != pk["shape"]:
        raise ValueError(f"[{rec['id']}] video latent {tuple(z0.shape)} != pipeline latent {pk['shape']}")
    meta, maps = log_likelihood(pipe, pk, z0, cfg)
    meta.update({"record": rec, "video": rec["video"], "active_frames": pk["noisy"],
                 "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "gpu": torch.cuda.get_device_name(0)})
    (out_dir / "lik.json").write_text(json.dumps(meta, indent=1, default=float))
    nll_maps = {f"nll_map_{k[4:]}": -(maps["prior"] + v).mean(0) for k, v in maps.items() if k.startswith("int_")}
    np.savez_compressed(out_dir / "lik_maps.npz", active_frames=np.asarray(pk["noisy"]),
                        **{k: v.astype(np.float32) for k, v in {**maps, **nll_maps}.items()})
    nll = meta["nll_per_dim"]
    print(f"[{rec['id']}] D {meta['D']} nll/dim raw {nll['raw']:.4f} hier {nll['hier']:.4f}"
          + (f" ref {nll['ref']:.4f}" if "ref" in nll else "")
          + f" ffjord {np.round(nll['ffjord'], 4).tolist()} | {meta['seconds']:.0f} s, peak {meta['peak_mem_gb']:.1f} GB",
          flush=True)
    torch.cuda.empty_cache()
    return meta


def cmd_run(args):
    import torch

    from .rollout import load_pipe
    from .run import load_manifest

    cfg = dict(DEFAULTS)
    for k in DEFAULTS:
        if getattr(args, k, None) is not None:
            cfg[k] = getattr(args, k)
    records = load_manifest(args.manifest)
    if args.ids:
        records = [r for r in records if r["id"] in args.ids]
    for r in records:
        if "video" not in r:
            if not args.rollouts:
                raise ValueError(f"[{r['id']}] no video in the record; pass --rollouts to score `record` outputs")
            r["video"] = str(Path(args.rollouts) / r["id"] / "rollout.mp4")
    t0 = time.time()
    pipe = load_pipe(args.model)
    print(f"loaded {args.model} in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}; cfg {cfg}", flush=True)
    for rec in records:
        od = Path(args.out) / rec["id"]
        if (od / "lik.json").exists() and not args.force:
            print(f"[{rec['id']}] exists, skipping")
            continue
        process_record(pipe, rec, od, cfg)


def _noise_corrected(est, ref, var_ref):
    """Relative l2 error of est against ref with the reference's own probe noise removed, plus bias and correlation."""
    d = (est - ref).ravel()
    mse = float((d ** 2).mean())
    ref_sq = float((ref.ravel() ** 2).mean()) - var_ref
    return {"rel": float(np.sqrt(max(mse - var_ref, 0.0) / max(ref_sq, 1e-30))), "mse": mse, "bias": float(d.mean()),
            "corr": float(np.corrcoef(est.ravel(), ref.ravel())[0, 1])}


def summarize(out_root):
    """Raw vs Hier per-cell and per-location (channel sum) log-likelihood against the reference, for every record.

    Raw (S probes per step) and the reference (R disjoint probes) are both plain probe averages, so
    E[(raw - ref)^2] = var_raw + var_ref = (R / S + 1) var_ref; var_ref is estimated from that and removed from every
    error against the reference. Writes <out>/lik_summary.json.
    """
    out_root = Path(out_root)
    rows = []
    for js in sorted(out_root.glob("*/lik.json")):
        meta = json.loads(js.read_text())
        m = np.load(js.parent / "lik_maps.npz")
        if "int_ref" not in m.files:
            print(f"[{js.parent.name}] no reference probes (--ref 0), skipped")
            continue
        S, R = int(meta["cfg"]["probes"]), int(meta["cfg"]["ref"])
        ints = {k: m[f"int_{k}"].astype(np.float64) for k in ("raw", "hier", "ref")}
        row = {"id": js.parent.name, "D": meta["D"], "nll_per_dim": meta["nll_per_dim"],
               "ref_se_per_dim": meta["trace_integral"]["ref_se"] / meta["D"]}
        for level, red in (("cell", lambda a: a), ("location", lambda a: a.sum(0))):
            r = {k: red(v) for k, v in ints.items()}
            var_ref = float(((r["raw"] - r["ref"]) ** 2).mean()) / (R / S + 1.0)
            row[level] = {"var_ref": var_ref, "raw": _noise_corrected(r["raw"], r["ref"], var_ref),
                          "hier": _noise_corrected(r["hier"], r["ref"], var_ref)}
        rows.append(row)
    table = {}
    for level in ("cell", "location"):
        raw = np.array([r[level]["raw"]["rel"] for r in rows])
        hier = np.array([r[level]["hier"]["rel"] for r in rows])
        table[level] = {"median_rel_raw": float(np.median(raw)), "median_rel_hier": float(np.median(hier)),
                        "hier_wins": int((hier < raw).sum()), "n": len(rows)}
        print(f"{level:9s} median rel. error vs reference: raw {table[level]['median_rel_raw']:.4f} "
              f"hier {table[level]['median_rel_hier']:.4f} (Hier better on {table[level]['hier_wins']}/{len(rows)})")
    for r in rows:
        n = r["nll_per_dim"]
        print(f"  {r['id']}: nll/dim ref {n['ref']:.4f} (se {r['ref_se_per_dim']:.5f}) raw {n['raw']:.4f} "
              f"hier {n['hier']:.4f} ffjord {np.round(n['ffjord'], 4).tolist()}")
    (out_root / "lik_summary.json").write_text(json.dumps({"table": table, "rows": rows}, indent=1))
    return table
