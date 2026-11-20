"""Hier: hierarchical variance-adaptive shrinkage of the stochastic Jacobian diagonal (docs/method.md, section Hier).
NumPy, one record at a time.

Cells are indexed (channel c, frame f, h, w). The per-probe products q have shape (S, C, F, H, W) and must contain the
active (denoised) frames only: the conditioning frame is a passthrough and has to be dropped before shrinkage.

The estimator is a biased plug-in regularizer, not an exact posterior. Two ablations are
included: the spatial-only prior (no channel-frame re-basing) and the channel-frame-only prior (each cell shrunk
straight to its slice median).
"""
import numpy as np
from scipy.ndimage import uniform_filter

S2_FLOOR = 1e-5     # Floor of the between-location variance s^2, in d^2 units.
S2W_FLOOR = 1e-4    # Floor of the within-location variance s_w^2.


def paule_mandel(y, v, center, iters=200, tol=1e-9):
    """Solve sum_l (y_l - center)^2 / (s^2 + v_l) = n for s^2 (Paule-Mandel) with the fixed point
    s^2 <- s^2 + (Q - n) / sum_l (s^2 + v_l)^-1, floored at S2_FLOOR. Returns (s2, iterations)."""
    dev2 = (y - center) ** 2
    n = y.size
    s2 = 0.0
    for it in range(iters):
        tot = s2 + v
        q = float((dev2 / tot).sum())
        new = max(s2 + (q - n) / float((1.0 / tot).sum()), S2_FLOOR)
        done = abs(new - s2) <= tol * max(new, 1e-12)
        s2 = new
        if done:
            break
    return s2, it + 1


def weighted_variance_mean(diff, vbar, iters=200, tol=1e-9, cap=None):
    """Precision-weighted fixed point for s_w^2: weights w_l = (s_w^2 + mean_c v_cl)^-2, started at the
    unweighted mean, floored at S2W_FLOOR and optionally capped. Returns (s2w, iterations, floor_hit)."""
    s2w = max(float(diff.mean()), S2W_FLOOR)
    floor_hit = 0
    for it in range(iters):
        w = (s2w + vbar) ** -2
        new = float((w * diff).sum() / w.sum())
        if new < S2W_FLOOR:
            new, floor_hit = S2W_FLOOR, floor_hit + 1
        if cap is not None:
            new = min(new, cap)
        done = abs(new - s2w) <= tol * max(new, 1e-12)
        s2w = new
        if done:
            break
    return s2w, it + 1, floor_hit


def box_mean(a, k=3):
    """Within-frame k x k mean with edge replication, applied over the last two axes."""
    size = (1,) * (a.ndim - 2) + (k, k)
    return uniform_filter(a, size=size, mode="nearest")


def pooled_variances(q, k=3, mode="covariance"):
    """Noise variances of the channel-pooled deviation r_bar and of its k x k neighbourhood mean m_tilde.

    covariance: per-probe channel means q_bar^(s) and their per-probe box means, var over probes (ddof 1) / S. This
    keeps the covariance between channels that share a probe (the default).
    independence: sum_c v / C^2 and its box mean / k^2 (under-predicts by 13-28 percent on Cosmos3).
    """
    S, C = q.shape[:2]
    if mode == "covariance":
        qbar = q.mean(1)                                                    # (S, F, H, W)
        v_pool = qbar.var(0, ddof=1) / S
        v_nb = box_mean(qbar, k).var(0, ddof=1) / S
    elif mode == "independence":
        d = q.mean(0)
        c = (q ** 2).mean(0)
        v = np.maximum(c - d ** 2, 0) / max(S - 1, 1)
        v_pool = v.sum(0) / C ** 2
        v_nb = box_mean(v_pool, k) / k ** 2
    else:
        raise ValueError(mode)
    return np.maximum(v_pool, 1e-12), np.maximum(v_nb, 1e-12)


def hier_from_products(q, k=3, pooled="covariance", variant="full"):
    """Shrink the raw diagonal estimate d = mean_s q^(s). q: (S, C, F, H, W) with active frames only.

    Returns a dict with d_raw, v (noise variance of d_raw), d_hier, the prior p and diagnostics worth
    storing (s2, s2w, w_mean, fixed-point iterations, floor hits, pooled-variance mode).
    """
    q = np.asarray(q, dtype=np.float64)
    S, C = q.shape[:2]
    if S < 2:
        raise ValueError("Hier needs at least two probes to estimate the per-cell noise variance.")
    d = q.mean(0)
    c = (q ** 2).mean(0)
    v = np.maximum(c - d ** 2, 0) / (S - 1)                                  # Unbiased noise variance of d.
    v = np.maximum(v, 1e-12)
    diag = {"S": int(S), "pooled_var": pooled, "variant": variant}

    # (1) Channel-frame re-basing: an exact shift, the per-cell noise variances are unchanged.
    mu_cf = np.median(d, axis=(2, 3), keepdims=True) if variant in ("full", "channel") else np.zeros((C, d.shape[1], 1, 1))
    r = d - mu_cf

    if variant == "channel":
        # Ablation: each cell straight to its slice median, with the between-cell variance from Paule-Mandel.
        s2, it = paule_mandel(r.ravel(), v.ravel(), 0.0)
        w = s2 / (s2 + v)
        diag.update({"s2": s2, "pm_iters": it, "w_mean": float(w.mean())})
        return {"d_raw": d, "v": v, "d_hier": mu_cf + w * r, "p": np.broadcast_to(mu_cf, d.shape).copy(), "diag": diag}

    # (2) Location level: pool the deviations across channels, shrink the neighbourhood mean toward the global
    # median, then the pooled value toward that.
    rbar = r.mean(0)                                                          # (F, H, W)
    v_pool, v_nb = pooled_variances(q - mu_cf[None], k, pooled)
    mu = float(np.median(rbar))
    s2, pm_iters = paule_mandel(rbar.ravel(), v_pool.ravel(), mu)
    mtil = box_mean(rbar, k)
    m_prime = mu + s2 / (s2 + v_nb) * (mtil - mu)                             # Neighbourhood mean toward the global median.
    m = m_prime + s2 / (s2 + v_pool) * (rbar - m_prime)                       # Pooled value toward the neighbourhood prior.

    # (3) Cell level: restore the slice offset, then shrink each cell toward its prior.
    p = mu_cf + m[None]
    var_ch = r.var(0, ddof=1)
    vbar = v.mean(0)
    s2w, wv_iters, floor_hits = weighted_variance_mean(var_ch - vbar, vbar)
    w = s2w / (s2w + v)
    d_hier = p + w * (d - p)
    diag.update({"s2": s2, "s2w": s2w, "pm_iters": pm_iters, "wv_iters": wv_iters, "floor_hits": floor_hits,
                 "w_mean": float(w.mean()), "mu": mu})
    return {"d_raw": d, "v": v, "d_hier": d_hier, "p": p, "diag": diag}


ESTIMATORS = {
    "raw": lambda q: hier_from_products(q)["d_raw"],
    "spatial": lambda q: hier_from_products(q, variant="spatial")["d_hier"],
    "channel": lambda q: hier_from_products(q, variant="channel")["d_hier"],
    "hier": lambda q: hier_from_products(q, variant="full")["d_hier"],
}
