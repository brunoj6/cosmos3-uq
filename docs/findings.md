# Findings

This is a summary of results from experiments whose full pipelines are not in this repository. The numbers are
reference points for new work, not reproducible artifacts. Notation is in [method.md](method.md). "rel" is the
relative l2 error of an estimated Jacobian diagonal against the exact one, and rho is the Spearman correlation with
it.

## 1. Raw vs Hier by model

| model | latent | S | raw rel / rho | Hier rel / rho | Hier better on |
|---|---|---|---|---|---|
| Cosmos3-Nano, 4 demo rollouts, 512 uniform exact cells each | 48-ch Wan2.2 VAE | 16 | 0.195 / 0.87 | 0.046 / 0.978 | 4 / 4 rollouts |
| Cosmos3-Nano, same, 128 salient cells | | 16 | 0.38 | 0.077 | 4 / 4 |
| Cosmos3-Edge, 8 driving clips, exact J_ii on all 52,800 active cells, 512-cell uniform sample | 48-ch Wan2.2 VAE | 16 | 0.119 / 0.908 | 0.044 / 0.980 | 8 / 8 at every S from 4 to 64 |
| Cosmos3-Edge, same, all cells | | 16 | 0.136 | 0.046 | 8 / 8 |
| Cosmos3-Edge, top 15 % cells by exact column energy | | 16 | 0.282 | 0.087 | 8 / 8 |
| N0-TWAM (Wan2.2-5B video + tactile), 38 chunks, sampled exact cells | 48-ch Wan2.2 VAE | not recorded | 0.020 / 0.93 | 0.011 / 0.98 | 38 / 38 |
| two-car pixel flow world model, per-patch likelihood | 3 RGB channels, pixels | 4 / 16 | rho 0.91 / 0.98 | rho 0.88 / 0.96 | raw better |
| Dubins vision world model (small AE latent, frame axis as grid) | few channels | 8 | rho 0.66 (t = 0.9) | rho 0.79 | Hier, worth about 2x the probes |
| Dubins state world model | 3 state channels | 8 | flat diagonal | no gain | none |

- **Ordering on Cosmos3.** Raw is worst, then spatial-only prior, then channel-only prior, then Hier. This holds at
  every budget.
- **Budget.** Hier error is nearly flat from S = 16 (Nano: 0.046 at 16, 0.037 at 64). Hier at S = 32 beats raw at
  S = 64 on 7/8 Edge clips; Hier at S = 4 does so on 1/8.
- **Noise model.** The raw z-gate (error / predicted noise sd) has standard deviation 0.999 at every S on Edge when
  the exact column energy is used. The probe noise model is exact, and the estimated noise variance is close to
  N(0, 1) from S >= 8.
- **Scalar trace.** Raw probes are enough; no shrinkage is needed. On Edge the 128-probe trace was within 1.5
  standard errors of the exact trace on all 8 clips. Small models needed 8 probes for under 1 % error.
- **Edge caveats.**
  - The gain in the channel-mean U map (what the heatmaps show) comes mostly from one heavy-tailed clip.
  - Raw and Hier maps tie on 7/8 clips at S = 16, and raw is slightly better at S = 64.
  - Hier flattens true outlier cells: 15 cells with exact J_ii up to 19.9 came out near 3. Raw was no better
    there, with noise sd 36.
  - On cells chosen as Hier's worst case, Hier still wins 8/8 at S = 16 but only 5/8 at S = 64, by a few percent.

## 2. When Hier helps

- **Controlled regime study.**
  - Setup: exact diagonal and column energy on a pixel world model, raw vs Hier at many probe budgets, cell sizes
    and integration step counts, plus a dial that multiplies column energy.
  - Hier lowers error only where the raw estimate is mostly noise: raw noise-to-signal above about 0.3-0.9.
  - It adds error wherever raw relative error is already below about 0.5, because it shrinks real local contrast
    toward the neighbourhood mean. A small sprite's contrast was shrunk by 30-100 %.
  - Hier never improved rank correlation (12 wins out of 700 settings).
- **Rule of thumb.**
  - Use Hier when the per-cell noise-to-signal ratio NSR_J = column energy / J_ii^2 is large and the diagonal is
    shared across many channels. On Cosmos3 and N0-TWAM, the channel correlation of J_ii is 0.93-0.95 over 48
    channels.
  - Keep raw for pixel-space models, for models with a few channels, and for any quantity that already averages many
    cells (patch sums, clip traces).
- **Sigma dependence.** On Edge, 1 - sigma * J_ii is 0.19-0.29 at sigma 0.256 but 0.04 at sigma 0.526. NSR_J drops
  from 0.071 to 0.012 between those two sigmas, so Hier matters most at the late query step that the pipeline uses.

## 3. Where Cosmos3's column energy comes from

- **Method.**
  - Full active-by-active Jacobians of Cosmos3-Edge at the 256 tier: 52,800 x 52,800, every row J^T e_i.
  - Column energy split by source, compared with a stationary Gaussian fitted to VAE-encoded driving windows, and
    rerun with modified token padding.
- **The bulk is the latent posterior covariance, not the transformer's architecture.**
  - For typical clips at sigma 0.256, the off-diagonal energy splits as about 30 % other channels at the same place,
    30-40 % the same place in other frames, about 30 % the same or neighbouring 2x2 token, and under 4 % anything
    farther.
  - The Jacobian is close to symmetric and positive semi-definite, as a covariance should be.
  - A Gaussian latent model with the same statistics reproduces this split, though not the overall size.
- **The VAE shapes the covariance.**
  - Wan2.2's 48 channels all carry smooth, correlated content. Latent lag-1 correlation is 0.70-0.78, against
    0.13-0.33 for a 48-component pixel-block PCA.
  - Perturbing one 16 x 16 x 4 pixel block leaves 35 % of the latent change in its own cell, 89 % within 5 x 5 cells
    and 99 % within +-1 latent frame.
  - In the channel-summed U map, cross-channel terms within a cell are 44-50 % of the noise. A global channel
    rotation does not remove them (5 % change); a per-cell oracle basis halves them, so the structure is
    content-specific.
- **The heavy tail is a padding artifact.**
  - 6 of 48 Edge records had "hub" inputs with column energy up to 63,000x typical. In one clip they held 83 % of the
    top 1 % of energy.
  - Hubs sit on the zero-padded bottom token row when the latent height is odd (11 or 15). They are non-symmetric and
    not PSD.
  - Filling the padding with noise instead of zeros removed 4/4 hubs; copying the edge row removed 2/4.
  - Hubs are not reproducible across reruns: 2 of 6 vanished.
- **Prediction.** Hier pays off for latent world models built on large-channel video VAEs (Wan2.2 family). Pixel
  models and low-channel latents keep raw.

## 4. Calibration of U (Cosmos3-Edge, MetaDrive clips)

- **Setup.**
  - 84 MetaDrive driving clips: 7 driving programs x 12 seeds, rendered and rolled out by frozen Cosmos3-Edge.
  - Split by seed: 56 train, 14 calibration, 14 test.
  - Event: |generated - real latent| <= eps. Pooled ECE on 20 bins, eps = 0.5 unless noted.
- **Raw U is badly over-confident for rollout error.**
  - At sigma 0.262 the mean U is 0.008-0.021, against a mean absolute latent error of 0.3-0.5. The erf rule gives
    confidence about 0.998 and ECE 0.28.
  - A single fitted inflation factor (6.2 at sigma 0.262, 2.8 at 0.971) still leaves ECE 0.08-0.17, because the
    needed inflation varies with U.
- **The velocity rule is calibrated for what it describes.** erf(eps sigma / sqrt(2U)) against one-step velocity
  error gives ECE 0.008-0.016 for sigma <= 0.9. Scored against rollout error it gives ECE 0.062-0.067
  (over-confident).
- **Isotonic recalibration fitted on rollout-error labels.**
  - Test ECE drops to 0.004-0.02. Per-cell AUROC is 0.62-0.66 at every sigma, for both the raw and the isotonic
    confidence.
  - An isotonic map fitted on velocity error does not fix rollout calibration (ECE 0.066).
  - After isotonic the pooled bias is near 0, but individual clips sit at +-0.05-0.15. Per-clip ECE has a floor that
    pooled ECE hides.
- **Choice of score.** Per-channel U ranks cells better than the smoothed channel-mean map for a per-channel target:
  AUROC 0.621 against 0.52.
- **Reading earlier in sampling.** Ranking barely changes: AUROC 0.633 at sigma 0.26, 0.648 at 0.67, 0.645 at 0.81,
  0.588 at 0.97. Clamping rises from 7 % to 42 % of cells.
- **Comparison with a trained confidence head (C3, frozen Edge, same clips).**
  - C3 head trained on velocity-error labels: raw ECE 0.047, isotonic 0.011, AUROC 0.645.
  - C3 head trained on rollout-error labels: AUROC 0.81-0.82, isotonic ECE 0.005.
  - Small MLP on log U at 6 sigmas, trained on rollout error: ECE 0.003, AUROC 0.747.
  - Most of the advantage of any method comes from training or recalibrating on rollout-error labels.

## 5. Likelihood estimation

- **Two-car pixel world model (3 RGB channels).** Per-patch log-likelihood against a 64-probe reference: raw Spearman
  0.91 at S = 4 and 0.98 at S = 16, Hier 0.88 and 0.96. Relative error raw 0.0004 vs Hier 0.0024 (reference noise
  0.0001). Hier shrank a real per-patch offset of the ego car (-0.075 nats per element) to -0.010. Frame-level log p
  from raw and Hier correlated at 0.999.
- **Cosmos3 (general lessons, no numbers kept).**
  - The one-probe Gaussian (FFJORD) estimate is about ten times noisier per clip than 16 Rademacher vjp probes per step.
  - The integration grid dominates the error: going from 32 to 128 steps moves log p more than the probe noise does.
  - Hier's summed trace is biased (negative at very small sigma, positive near sigma 1); raw stays unbiased.
  - Per cell, Hier is closer to the reference than raw; per location (sum over the 48 channels), raw is closer than
    Hier. Use Hier for per-cell maps only and raw for any sum.
  - The clip log-likelihood tracks how hard a clip is overall (it correlates strongly with the clip-level U score)
    and separated clips with a specific object failure from their scene siblings less well than the U score did.

## 6. Practical notes

- **Default.** S = 16, Hier for per-cell maps, raw for traces, sums and likelihoods, capture at the last grid step.
- **What the heatmap shows.** It marks where the model is undecided at the end of sampling. Use the outline (top
  1 % of the smoothed map) for localization and the clip score or calibrated confidence for comparisons across clips.
- **Calibration data.** Calibrated probabilities need a calibration split of rollouts with real futures from the
  same domain. Fit isotonic maps per tolerance with `run calibrate` and keep the fit file with the model.
