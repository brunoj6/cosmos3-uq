"""Turning the per-cell uncertainty U into calibrated confidences, and measuring calibration.

Event: a latent cell is "accurate" at tolerance eps when |generated - real| <= eps, on the pipeline's normalized
latents over the predicted frames. U estimates the posterior variance of the clean latent given x_sigma.

Raw confidence rules (no fitting):
  two_draw   erf(eps / (2 sqrt(U))): the rollout and the real future are treated as two independent posterior draws,
             so their difference has variance 2U. This is the rule for rollout error.
  velocity   erf(eps sigma / sqrt(2U)): the error of the one-step velocity prediction at the captured sigma
             (v error = x0 error / sigma). It is calibrated for velocity error, not for rollout error.

Isotonic recalibration: a non-decreasing map from the score -U to the accuracy, fitted per tolerance with pool
adjacent violators on a calibration split of rollouts with real futures, then applied to new rollouts. It keeps the
ranking (AUROC) and fixes the scale. Fit it on rollout-error labels; a map fitted on velocity error stays
over-confident on rollouts (docs/findings.md).

Measures: pooled ECE and MCE over equal-width confidence bins (cells of all clips pooled), the mean of per-clip ECEs,
and AUROC of the score for the accuracy indicator.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import erf
from scipy.stats import rankdata

MAX_CELLS = 3_000_000


def conf_two_draw(u, eps):
    return erf(eps / (2.0 * np.sqrt(np.maximum(u, 1e-12))))


def conf_velocity(u, eps, sigma):
    return erf(eps * sigma / np.sqrt(2.0 * np.maximum(u, 1e-12)))


RULES = {"two_draw": lambda u, eps, sigma: conf_two_draw(u, eps),
         "velocity": lambda u, eps, sigma: conf_velocity(u, eps, sigma)}


class Isotonic:
    """Non-decreasing map score -> probability, fitted with pool adjacent violators on quantile bins."""

    def __init__(self, x=None, y=None):
        self.x = None if x is None else np.asarray(x, dtype=np.float64)
        self.y = None if y is None else np.asarray(y, dtype=np.float64)

    def fit(self, score, label, bins=2000):
        order = np.argsort(score)
        s, lab = np.asarray(score, np.float64)[order], np.asarray(label, np.float64)[order]
        edges = np.linspace(0, s.size, bins + 1).astype(int)
        blocks = []                                   # [weight, weighted label sum, weighted score sum].
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            blocks.append([b - a, lab[a:b].sum(), s[a:b].sum()])
            while len(blocks) > 1 and blocks[-2][1] / blocks[-2][0] > blocks[-1][1] / blocks[-1][0]:
                w, ly, sx = blocks.pop()
                blocks[-1] = [blocks[-1][0] + w, blocks[-1][1] + ly, blocks[-1][2] + sx]
        self.x = np.array([bl[2] / bl[0] for bl in blocks])
        self.y = np.array([bl[1] / bl[0] for bl in blocks])
        return self

    def __call__(self, score):
        out = np.interp(score, self.x, self.y, left=self.y[0], right=self.y[-1])
        return np.clip(out, 1e-6, 1 - 1e-6)

    def to_dict(self):
        return {"x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["x"], d["y"])


def reliability(conf, acc, num_bins=20):
    """Pooled reliability bins (conf in (lo, hi]) with ECE and MCE over non-empty bins."""
    edges = np.linspace(0, 1, num_bins + 1)
    idx = np.clip(np.ceil(conf * num_bins).astype(int) - 1, 0, num_bins - 1)
    n = np.bincount(idx, minlength=num_bins).astype(float)
    a = np.bincount(idx, weights=acc, minlength=num_bins)
    c = np.bincount(idx, weights=conf, minlength=num_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc_b, conf_b = np.where(n > 0, a / n, np.nan), np.where(n > 0, c / n, np.nan)
    gap = np.where(n > 0, np.abs(np.nan_to_num(acc_b) - np.nan_to_num(conf_b)), 0.0)
    return {"edges": edges.tolist(), "count": n.tolist(), "acc": acc_b.tolist(), "conf": conf_b.tolist(),
            "ece": float((gap * n).sum() / max(n.sum(), 1)), "mce": float(gap.max())}


def auroc(score, label, rng=None, max_n=MAX_CELLS):
    """AUROC of score for the positive class (label 1), tie-aware, on a random subsample above max_n cells."""
    score, label = np.asarray(score).ravel(), np.asarray(label).ravel()
    if score.size > max_n:
        idx = (rng or np.random.default_rng(0)).choice(score.size, max_n, replace=False)
        score, label = score[idx], label[idx]
    pos = label > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(score)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def record_cells(record_dir, estimator="hier"):
    """Per-cell U and absolute latent error of one record (both flattened), plus its sigma.

    The record must store `latents_gt`, i.e. it was recorded with a `gt_video` in its manifest entry.
    """
    z = np.load(Path(record_dir) / "u_divergence.npz")
    if "latents_gt" not in z.files:
        raise ValueError(f"{record_dir}: no latents_gt; add gt_video to the manifest record and re-run `record`")
    u = z[f"u_{estimator}"].astype(np.float64)
    err = np.abs(z["latents_final"].astype(np.float64) - z["latents_gt"].astype(np.float64))
    if u.shape != err.shape:
        raise ValueError(f"{record_dir}: U {u.shape} and latent error {err.shape} do not line up")
    return u.ravel(), err.ravel(), float(z["sigma"])


def _pool(dirs, estimator, rng):
    cells = [record_cells(d, estimator) for d in dirs]
    cap = max(1, MAX_CELLS // max(len(cells), 1))
    us, errs = [], []
    for u, e, _ in cells:
        keep = rng.choice(u.size, min(u.size, cap), replace=False)
        us.append(u[keep])
        errs.append(e[keep])
    return np.concatenate(us), np.concatenate(errs), cells


def evaluate(fit_dirs, test_dirs, eps_list, estimator="hier", rule="two_draw", num_bins=20, seed=0):
    """Fit isotonic maps on fit_dirs, score raw and recalibrated confidences on test_dirs, per tolerance."""
    rng = np.random.default_rng(seed)
    u_fit, err_fit, _ = _pool(fit_dirs, estimator, rng)
    test = [record_cells(d, estimator) for d in test_dirs]
    result = {"estimator": estimator, "rule": rule, "num_bins": num_bins, "n_fit": len(fit_dirs),
              "n_test": len(test_dirs), "fits": {}, "test": {}}
    for eps in eps_list:
        iso = Isotonic().fit(-u_fit, (err_fit <= eps).astype(np.float64))
        result["fits"][str(eps)] = iso.to_dict()
        raw_c, iso_c, accs, scores, clip_ece = [], [], [], [], {"raw": [], "iso": []}
        for u, e, sigma in test:
            acc = (e <= eps).astype(np.float64)
            rc = RULES[rule](u, eps, sigma)
            ic = iso(-u)
            clip_ece["raw"].append(reliability(rc, acc, num_bins)["ece"])
            clip_ece["iso"].append(reliability(ic, acc, num_bins)["ece"])
            raw_c.append(rc)
            iso_c.append(ic)
            accs.append(acc)
            scores.append(-u)
        acc = np.concatenate(accs)
        entry = {"mean_accuracy": float(acc.mean()), "auroc": auroc(np.concatenate(scores), acc, rng)}
        for name, conf in (("raw", np.concatenate(raw_c)), ("iso", np.concatenate(iso_c))):
            rel = reliability(conf, acc, num_bins)
            entry[name] = {"mean_confidence": float(conf.mean()), "ece_pooled": rel["ece"], "mce_pooled": rel["mce"],
                           "ece_per_clip_mean": float(np.mean(clip_ece[name])), "bins": rel}
        result["test"][str(eps)] = entry
    return result


def save_fits(result, path):
    Path(path).write_text(json.dumps({"estimator": result["estimator"], "fits": result["fits"]}, indent=1))


def load_fits(path):
    """{eps: Isotonic} from a file written by save_fits, to recalibrate new rollouts."""
    d = json.loads(Path(path).read_text())
    return {float(k): Isotonic.from_dict(v) for k, v in d["fits"].items()}
