"""Uncertainty map and clip score from a Jacobian diagonal (docs/method.md).

Arrays are (C, F, H, W) over the active frames only; the conditioning frame must already be excluded, because it is a
passthrough whose U is uniform and about ten times larger than the predicted frames.
"""
import numpy as np

from .hier import box_mean


def u_map(d, sigma):
    """Per-cell, per-channel uncertainty in Cosmos convention: U = max(0, sigma^2 / (1 - sigma) (1 - sigma d)).
    Returns (U, clamp_fraction); the clamp fraction is the share of raw cells that were negative."""
    scale = sigma ** 2 / (1.0 - sigma)
    raw = scale * (1.0 - sigma * d)
    return np.maximum(raw, 0.0), float((raw < 0).mean())


def u_map_unclamped(d, sigma):
    """Unclamped version, needed before interpreting a sign change across sigma."""
    return sigma ** 2 / (1.0 - sigma) * (1.0 - sigma * d)


def channel_mean(u):
    """Channel mean -> (F, H, W) map M."""
    return u.mean(0)


def clip_scores(m, k=3):
    """Scalar clip scores from a channel-mean map over the active frames: the headline u_smooth3_q99 (box-smoothed,
    99th percentile), and the plain u_mean, u_q50, u_q99, u_max for comparisons."""
    ms = box_mean(m, k)
    return {
        "u_smooth3_q99": float(np.quantile(ms.ravel(), 0.99)),
        "u_smooth3_mean": float(ms.mean()),
        "u_mean": float(m.mean()),
        "u_q50": float(np.quantile(m.ravel(), 0.5)),
        "u_q99": float(np.quantile(m.ravel(), 0.99)),
        "u_max": float(m.max()),
    }


def trace_uncertainty(diag_sum, num_cells, sigma):
    """Scalar U = Tr Cov(x0 | x_sigma) = sigma^2 / (1 - sigma) (N - sigma sum_i J_ii) (Cosmos
    convention), from the raw dot-product trace with no shrinkage and no clamp."""
    return sigma ** 2 / (1.0 - sigma) * (num_cells - sigma * diag_sum)


def rank_tint(m, lo=0.5, hi=0.99):
    """Within-map percentile ranks on a fixed p50 -> p99 ramp in [0, 1], the display rule used by heatmap.py."""
    flat = m.ravel()
    ranks = np.argsort(np.argsort(flat)) / max(flat.size - 1, 1)
    return np.clip((ranks - lo) / (hi - lo), 0.0, 1.0).reshape(m.shape)
