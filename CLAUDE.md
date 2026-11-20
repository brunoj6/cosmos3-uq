# Agent guide

Training-free uncertainty maps for Cosmos3-Nano and Cosmos3-Edge. Start with `README.md`, then `docs/method.md`
(math, settings, pitfalls) and `docs/findings.md` (results to build on).

## Rules
- Do not commit outputs, datasets, model weights, videos or `examples/assets/`; `.gitignore` covers them.
- Heavy runs (rollouts, exact Jacobians) need a GPU; this package is not meant for laptops beyond `selftest`.
- When a new result changes a conclusion, update `docs/findings.md` as text. Do not add experiment pipelines,
  result pages or planning notes to the repo.

## Code
- Python: snake_case for functions and variables, PascalCase for classes; comments are full sentences ending with a
  period.
- Shapes: per-probe products (S, C, F, H, W) and maps (C, F, H, W) or (F, H, W) cover the predicted latent frames only;
  the conditioning frame is always dropped.
- `python -m cosmos3_uq.run selftest` must pass after any change to `hier.py`, `maps.py` or `calibration.py`.
