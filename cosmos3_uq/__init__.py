"""Training-free per-cell uncertainty maps for Cosmos3-Nano and Cosmos3-Edge rollouts.

The uncertainty of a flow-matching sample is read from the divergence of the velocity field: the diagonal of the
velocity Jacobian gives the per-cell posterior variance of the clean latent (docs/method.md).

Modules:
  rollout      forward-dynamics rollout with the diffusers Cosmos3OmniPipeline; captures the velocity closure at the
               query noise level, the final latents, and encodes real videos into the same latent space.
  divergence   Hutchinson diagonal estimates with adjoint (vjp) products, exact Jacobian rows.
  hier         hierarchical variance-adaptive shrinkage (Hier) of the stochastic diagonal.
  maps         per-cell uncertainty U, channel-mean maps, clip scores.
  heatmap      clip-level heatmap overlays on the decoded rollout (PNG strip and MP4).
  calibration  confidence rules, isotonic recalibration, ECE/MCE and AUROC.
  fidelity     estimator error against exact diagonal entries (sampled cells).
  exact        exact diagonal on every active cell of a small configuration, scored against raw and Hier.
  likelihood   log-likelihood of a video along the probability-flow ODE, per-cell maps, FFJORD/raw/Hier traces.
  run          command line entry point.

Conventions (diffusers/Cosmos): time runs data -> noise, x_sigma = sigma * noise + (1 - sigma) * x0, v = noise - x0,
so U_i = max(0, sigma^2 / (1 - sigma) * (1 - sigma * J_ii)).
"""
