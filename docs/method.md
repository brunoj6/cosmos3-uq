# Method: divergence uncertainty with Hier shrinkage

This note describes what `cosmos3_uq` computes and the settings and pitfalls that matter. Results are in
[findings.md](findings.md).

## 1. The quantity

Cosmos3 uses the diffusers flow convention: time runs from data to noise,

    x_sigma = sigma * noise + (1 - sigma) * x0,    v = noise - x0,    x0_hat = x_sigma - sigma * v.

For a flow-matching model the Jacobian J = dv/dx_sigma of the velocity determines the posterior covariance of the
clean latent given the noisy one:

    J = I / sigma - ((1 - sigma) / sigma^3) * Cov(x0 | x_sigma)
    Cov(x0 | x_sigma) = sigma^2 / (1 - sigma) * (I - sigma * J).

The per-cell uncertainty is the diagonal of that covariance,

    U_i = max(0, sigma^2 / (1 - sigma) * (1 - sigma * J_ii)),

and the scalar uncertainty is its trace, `sigma^2 / (1 - sigma) * (N - sigma * sum_i J_ii)`. A cell has high U when
the model, at that noise level, has not yet decided what goes there. No training is involved: everything is read from
the frozen model.

## 2. Pipeline

1. **Rollout and capture** (`rollout.py`). Run the normal forward-dynamics rollout (UniPC, 30 steps, flow shift 10,
   guidance 1.0). At the sampler step whose sigma is closest to `sigma_q = 0.3`, store the transformer's inputs and
   the float32 latents entering the step. This is the last step of the 30-step grid: sigma 0.262 for Nano and 0.256
   for Edge (Edge uses its native linear schedule). The closure `v(x)` reruns the transformer with every other input
   frozen and zeroes the conditioning frame's velocity, as the pipeline does.
2. **Probes** (`divergence.py`). One forward pass, then S backward passes with Rademacher probes eps. Each pass gives
   the adjoint product `J^T eps`, and `q = eps * (J^T eps)` is an unbiased per-cell estimate of J_ii. All per-probe
   products are stored, so smaller budgets and the noise variances can be read later from the same draw.
3. **Raw diagonal**: `d = mean_s q`. Its noise variance is `v = (mean_s q^2 - d^2) / (S - 1)`. Per probe, the
   variance of cell i equals its column energy `sum_{j != i} J_ji^2`, so cells with large off-diagonal mass are noisy.
4. **Hier** (`hier.py`). Shrinks each noisy d_i toward a prior built from cells that share its channel, frame and
   location. Weights follow the estimated noise, so precise cells barely move.
5. **Maps and scores** (`maps.py`). U per cell and channel, the channel-mean map M (frames x h x w), and the clip
   score `u_smooth3_q99` (99th percentile of the 3x3 box-smoothed M). The scalar trace always uses the raw
   dot-product mean, without shrinkage or clamping.
6. **Heatmap** (`heatmap.py`). See section 5.
7. **Calibration** (`calibration.py`). See section 6.
8. **Likelihood** (`likelihood.py`). See section 7.

Only the predicted (noisy) latent frames are used. The conditioning frame is a passthrough whose U is uniform and
about ten times larger than the rest, so it would dominate every statistic.

## 3. Hier in detail

Cells are indexed (channel c, frame f, h, w). Inputs are the per-probe products q of shape (S, C, F, H, W).

1. **Channel-frame re-basing.** Subtract the median of each (c, f) slice: `r = d - mu_cf`. The per-cell noise
   variances do not change.
2. **Location level.** Average the deviations over channels, `r_bar(f, h, w)`. The noise variances of `r_bar` and of
   its 3x3 neighbourhood mean are computed from the per-probe channel means, which keeps the covariance between
   channels that share a probe. Treating channels as independent under-predicts these variances by 13-28 percent on
   Cosmos3. The between-location variance s^2 comes from the Paule-Mandel fixed point. The neighbourhood mean is
   shrunk toward the global median, and then `r_bar` is shrunk toward that neighbourhood value.
3. **Cell level.** The prior of each cell is `p = mu_cf + m(f, h, w)`. The within-location variance s_w^2 comes
   from a precision-weighted fixed point on the across-channel spread minus its noise. Each cell becomes
   `p + s_w^2 / (s_w^2 + v_i) * (d_i - p)`.

Hier is a biased plug-in regularizer, not an exact posterior. It helps when the raw estimate is noise-dominated and the
diagonal is shared across many channels. It hurts when the raw estimate is already accurate, because it pulls real
local structure toward the neighbourhood (findings.md, sections 1 and 2). Two ablations ship in `ESTIMATORS`:
`spatial`, which skips the channel-frame re-basing, and `channel`, which shrinks each cell straight to its slice
median.

## 4. Settings

| setting | value |
|---|---|
| probe budget for maps | S = 8-16 (Hier error is nearly flat from S = 16) |
| query sigma | 0.3 requested, which captures the last 30-step grid entry |
| product | vjp (adjoint), Rademacher probes, fixed seed |
| Nano | 480 tier, 832x480, latent 48 x 6 x 30 x 52, needs an 80 GB GPU (16B bf16 plus the retained graph) |
| Edge | 480 or 256 tier; 256 tier latent 48 x 6 x 11 x 20 = 52,800 active cells |
| cost (Nano, A100 80 GB) | load 74 s, rollout 8-10 s, 16 probes 5 s, about 0.3 s per exact cell |
| cost (Edge full exact) | about 57 rows/s batched, 16 min per record at the 256 tier |

**Actions.** The AV domain takes 9D actions per step: the pose of camera k+1 in the frame of camera k, in OpenCV
axes (x right, y down, z forward), as translation in metres followed by the first two columns of the rotation matrix
(rot6d). `examples/assets/av_traj_*.json` are lists of such rows.

**Manifest records.** A JSON list of objects with `id`, `image` (conditioning frame), `actions` (.json list or .npy,
shape [T, 9]), and optionally `chunk_size` (20), `fps` (10), `resolution_tier` (480), `seed`, `prompt`, and
`gt_video`. `gt_video` is the real video starting at the conditioning frame with at least `chunk_size + 1` frames.
It is needed for latent error maps and calibration. Relative paths resolve against the manifest's folder.

## 5. Heatmap display rule

- Tint each cell of the channel-mean map by its percentile rank within the clip, on a fixed p50 -> p99 ramp, and
  paint it orange over a grey copy of the rollout.
- Outline in black the cells above the clip-wide 99th percentile of the 3x3-smoothed map. This is the region behind
  the clip score.
- Ranks are clip-wide, so frames of one clip can be compared but colours cannot be compared across clips. To compare
  clips, use the clip score or the calibrated confidence.
- The VAE compresses time by 4: latent frame k >= 1 covers pixel frames 4k-3 .. 4k. Spatially, one latent cell is
  16 x 16 pixels.

## 6. Calibration

The event is "latent cell is accurate", `|generated - real| <= eps`, measured on normalized latents over the
predicted frames. The real latent is encoded through the pipeline's own action-video path
(`rollout.encode_video_latents`), so it lines up cell by cell with the rollout.

- **Raw rules.**
  - `two_draw`, erf(eps / (2 sqrt U)): treats the rollout and the real future as two posterior draws. This is the rule
    for rollout error.
  - `velocity`, erf(eps sigma / sqrt(2U)): the probability for the one-step velocity error at the captured sigma.
  - Both are badly over-confident on rollout error, because U at the last step is far smaller than the accumulated
    rollout error.
- **Isotonic recalibration.**
  - Fit a non-decreasing map from -U to accuracy, separately for each eps, on a calibration split of rollouts with
    real futures, using pool adjacent violators on 2,000 quantile bins. Apply it to new rollouts.
  - The fit must use rollout-error labels. A fit on velocity error does not transfer.
- **Measures.** Pooled ECE/MCE on 20 equal-width bins, the mean of per-clip ECEs (pooled ECE hides per-clip bias),
  and AUROC of -U for the accuracy indicator.

## 7. Likelihood

The same velocity-Jacobian trace gives the log-likelihood of a whole video under the model. A flow-matching model is a
continuous normalizing flow, so along the probability-flow ODE path from the data latent x0 (sigma = 0) to the noise
latent x1 (sigma = 1),

    log p(x0) = log N(x1; 0, I) + int_0^1 tr J(x_sigma, sigma) dsigma.

- **What is scored.** A video (a rollout of the model or a real recording) is encoded with the pipeline's VAE path;
  latent frame 0 is the clean conditioning frame and the record's actions condition the transformer. The result is
  the model's conditional density of the predicted latent frames given the first frame and the actions.
- **Integration.** Logit-uniform sigma grid from 1e-3 to 1 (dense at both ends, where the trace changes fastest),
  Heun steps, trapezoid rule for the trace integral. The transformer is evaluated at sigma <= 0.999.
- **Trace estimators**, all along one shared path:
  - `ffjord`: one Gaussian probe held fixed over the whole path (the standard FFJORD estimator); each of the G probes
    gives one likelihood estimate, so their spread shows the cost of a single probe.
  - `raw`: S fresh Rademacher vjp probes per step, the plain Hutchinson diagonal summed.
  - `hier`: the same S probes shrunk per cell by Hier, then summed.
  - `ref`: R further disjoint probes per step, a reference for the trace and the per-cell diagonal.
- **Per-cell maps.** The prior term and the trace both split over cells, so `prior_i + int J_ii dsigma` is an
  additive per-cell decomposition of log p. `lik_maps.npz` stores the per-cell prior and trace integrals and
  channel-mean NLL maps for the heatmap (high = unlikely under the model). Only diagonal estimators give these maps.
- **Which estimator for what.** Hier does not preserve sums: shrinkage moves the total, and its trace is biased where
  the diagonal is far from its neighbourhood prior. Use raw for log p of the clip and for any per-location or
  per-region sum; Hier only for per-cell maps. `likelihood-summary` scores both against the reference, per cell and
  per location (sum over channels), after removing the reference's own probe noise.
- **Grid error.** The sigma grid, not the probes, usually limits accuracy. Compare 32 steps against 64 or 128 on a
  few clips before trusting differences smaller than a few hundredths of a nat per dimension. The selftest checks the
  integrator against an exact Gaussian density: coarse grids err most on low-variance data.
- **Cost.** One forward per probe step and per Heun predictor, one backward per probe. On Cosmos3-Edge at 480 tier
  with 32 steps, 4 Gaussian + 16 + 48 Rademacher probes per step: about 2 minutes per clip and 14 GB on an A100.
  Nano needs an 80 GB GPU and a smaller `--batch`.

## 8. Pitfalls

- **Sigma grid.** The checkpoint config sets flow shift 1 with Karras sigmas. Override it to shift 10 without Karras
  (`load_pipe` does). Edge's `use_native_flow_schedule` changes the grid. If the grid is wrong, the capture never
  fires and `rollout` raises.
- **Convention gate.** `x0_pred.png` (x0 = x_sigma - sigma v at the captured step) must decode to a clean scene. If it
  does not, the sign or time convention is wrong.
- **Clamping.** U is clamped at 0. Use `u_map_unclamped` before reading a sign change across sigma. Record the clamp
  fraction.
- **Hier needs S >= 2** and the active frames only.
- **Precision.** The closure takes float32 latents and casts them to the model's bf16; the products are kept in
  float32.
- **Selection bias.** Ranking salient cells with the same probes that are then scored favours Hier. Rank with
  separate probes or with the exact column energy, as `exact.py` does.
- **Zero padding.** When the latent height is odd (11 or 15 rows), the transformer pads the bottom token row with
  zeros. Inputs in that row can become hubs with huge column energy (findings.md, section 3).
- **Determinism.** Rollouts from the diffusers backend are deterministic for a fixed seed and settings, but they are
  not bit-identical to other Cosmos backends.
