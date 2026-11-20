# cosmos3_uq: training-free uncertainty maps for Cosmos3 world models

`cosmos3_uq` computes per-cell uncertainty maps for rollouts of NVIDIA's Cosmos3-Nano and Cosmos3-Edge
action-conditioned world models. It does no training. For a flow-matching model, the diagonal of the velocity
Jacobian gives the posterior variance of each latent cell. The package estimates that diagonal with
Hutchinson probes, denoises the estimate with a hierarchical shrinkage estimator (Hier), and turns it into:

- a per-cell, per-channel uncertainty U, a channel-mean map, and a scalar clip score;
- clip-level heatmaps on the decoded rollout (PNG strip and MP4);
- calibrated probabilities that a cell is accurate, via isotonic recalibration against real futures;
- the log-likelihood of a whole video under the model, with per-cell likelihood maps.

How it works: [docs/method.md](docs/method.md). What we found (where Hier helps, where the noise comes from, how well
U is calibrated): [docs/findings.md](docs/findings.md).

## Install

Python 3.10+ and a CUDA GPU. Cosmos3-Nano needs an 80 GB GPU. Cosmos3-Edge runs on smaller GPUs: about 20 GB at
the 480 tier and about 10 GB for exact rows at the 256 tier.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

`pyproject.toml` pins diffusers to the commit this code was tested with (the Cosmos3 pipeline was not yet in a
diffusers release). The weights download from Hugging Face on first use (`nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Edge`).

## Quick start

```bash
# CPU check of the estimators and the calibration code.
python -m cosmos3_uq.run selftest

# Rollout + raw and Hier uncertainty for each manifest record (Nano or Edge).
python -m cosmos3_uq.run record --manifest examples/demo.json --out outputs/nano --model nano
python -m cosmos3_uq.run record --manifest examples/demo.json --out outputs/edge --model edge

# Clip-level heatmaps: heatmap_strip.png and heatmap.mp4 in each record folder.
python -m cosmos3_uq.run heatmap outputs/nano/av0_forward outputs/nano/av0_left

# Estimator accuracy against the exact diagonal cells that `record` samples (--exact, --salient).
python -m cosmos3_uq.run fidelity --out outputs/nano

# Exact diagonal on every active cell (Edge, 256 tier), raw vs Hier on all of it.
python -m cosmos3_uq.run exact --manifest examples/demo_edge_exact.json --out outputs/edge_exact
python -m cosmos3_uq.run exact-summary --out outputs/edge_exact

# Isotonic recalibration (records need gt_video, see below).
python -m cosmos3_uq.run calibrate --fit outputs/edge/a outputs/edge/b --test outputs/edge/c --eps 0.3 0.5 \
    --out outputs/edge_calib.json

# Log-likelihood of rollouts (or real videos: add "video" to the manifest records), then raw vs Hier vs reference.
python -m cosmos3_uq.run likelihood --manifest examples/demo.json --rollouts outputs/edge --out outputs/edge_lik --model edge
python -m cosmos3_uq.run likelihood-summary --out outputs/edge_lik
python -m cosmos3_uq.run heatmap outputs/edge_lik/av0_forward      # nll_strip.png and nll.mp4
```

`examples/demo.json` expects the Cosmos3 cookbook demo inputs (`av_0.jpg`, `av_traj_{forward,left,right}.json`) in
`examples/assets/`. They are not tracked here; copy them from NVIDIA's Cosmos3 action-conditioned cookbook or point
the manifest at your own frame and actions.

## Inputs

A manifest is a JSON list of records:

```json
{"id": "clip0", "image": "frame0.jpg", "actions": "actions.json", "chunk_size": 20, "fps": 10,
 "resolution_tier": 480, "seed": 0, "gt_video": "real_future.mp4"}
```

- `image`: the conditioning frame.
- `actions`: a JSON list or `.npy` of 9D AV actions (camera k+1 pose in camera k frame, OpenCV axes, metres + rot6d).
- `gt_video` (optional): the real video starting at the conditioning frame, at least `chunk_size + 1` frames. It
  enables latent error maps and calibration.
- `video` (for `likelihood`): the video to score, with the same frame requirements. Without it, `--rollouts DIR`
  scores `DIR/<id>/rollout.mp4`.
- Relative paths resolve against the manifest's folder.

## Outputs of `record`

Per record, in `<out>/<id>/`:

| file | content |
|---|---|
| `rollout.mp4` | the generated clip |
| `x0_pred.png` | x0 prediction at the captured step; must look like a clean scene (convention check) |
| `u_divergence.npz` | `d_raw`, `d_hier`, `noise_var`, `u_hier`, `u_raw` (C, F, h, w); `map_hier`, `map_raw` (F, h, w); stored probes; exact cells; `latents_final`, `latents_gt` |
| `u_divergence.json` | settings, captured sigma, clip scores (`u_smooth3_q99` and others), Hier diagnostics, timings |

## Layout

```
cosmos3_uq/      rollout capture, estimators, Hier, maps, heatmaps, calibration, fidelity, exact diagonal,
                 likelihood, CLI
docs/method.md   math, settings, pitfalls
docs/findings.md summary of results
examples/        demo manifests (assets not tracked)
```
