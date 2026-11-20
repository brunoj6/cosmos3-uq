"""Velocity-Jacobian diagonal estimation for a captured closure v(x).

All functions take a closure that maps a latent tensor x to the velocity v(x) of the same shape, with every other
input frozen. The forward pass is run once and its graph is reused for every backward pass (adjoint products
J^T eps by backpropagation).
"""
import time

import numpy as np
import torch


def rademacher(shape, generator, device):
    """Fixed-seed Rademacher probe in float32."""
    return torch.randint(0, 2, shape, generator=generator, device=device).float() * 2 - 1


def hutchinson_vjp(closure, x, num_probes, seed=0, active=None):
    """Hutchinson diagonal estimate with adjoint products.

    Returns the per-probe products q^(s) = eps (x) (J^T eps) as a (S, *x.shape) float32 CPU tensor, so that nested
    budgets (the first S' probes), the noise variances and the covariance-aware pooled variances of Hier can all be
    read from one draw. `active` is an optional {0,1} mask; probes are zero outside it.
    """
    generator = torch.Generator(device=x.device).manual_seed(seed)
    x = x.detach().clone().float().requires_grad_(True)
    with torch.enable_grad():
        v = closure(x)
    products = []
    t0 = time.time()
    for s in range(num_probes):
        eps = rademacher(x.shape, generator, x.device)
        if active is not None:
            eps = eps * active
        (g,) = torch.autograd.grad(v, x, grad_outputs=eps, retain_graph=s < num_probes - 1)
        products.append((eps * g.float()).cpu())
    return torch.stack(products), time.time() - t0


def exact_diagonal(closure, x, flat_indices):
    """Exact J_ii for the given flat coordinates, one backward pass per cell with the unit cotangent e_i, since
    (J^T e_i)_i = J_ii. Returns (values, seconds)."""
    x = x.detach().clone().float().requires_grad_(True)
    with torch.enable_grad():
        v = closure(x)
    out = torch.empty(len(flat_indices), dtype=torch.float32)
    e = torch.zeros(x.numel(), device=x.device)
    t0 = time.time()
    for n, i in enumerate(flat_indices):
        e.zero_()
        e[i] = 1.0
        (g,) = torch.autograd.grad(v, x, grad_outputs=e.view_as(x), retain_graph=n < len(flat_indices) - 1)
        out[n] = g.reshape(-1)[i].float().cpu()
    return out, time.time() - t0


def exact_rows(closure, x, flat_indices, batch=16):
    """J_ii for every listed coordinate and the column energies sum_i J_ij^2 over those rows.

    Row i of J is J^T e_i (one adjoint pass with a unit cotangent). `batch` unit cotangents go through one batched
    backward (is_grads_batched); if the model has no batching rule the loop falls back to one row per pass.
    Returns (diag float64 [n], column energy float64 [x.numel()], seconds, rows per pass actually used).
    """
    x = x.detach().clone().float().requires_grad_(True)
    with torch.enable_grad():
        v = closure(x)
    n_all = x.numel()
    idx = torch.as_tensor(np.asarray(flat_indices), device=x.device, dtype=torch.long)
    diag = torch.empty(idx.numel(), dtype=torch.float64, device=x.device)
    col = torch.zeros(n_all, dtype=torch.float64, device=x.device)
    t0, last = time.time(), time.time()
    start = 0
    while start < idx.numel():
        rows = idx[start: start + batch]
        b = rows.numel()
        e = torch.zeros(b, n_all, device=x.device)
        e[torch.arange(b, device=x.device), rows] = 1.0
        try:
            if b == 1:
                (g,) = torch.autograd.grad(v, x, grad_outputs=e.view_as(x), retain_graph=True)
            else:
                (g,) = torch.autograd.grad(v, x, grad_outputs=e.view(b, *x.shape), retain_graph=True,
                                           is_grads_batched=True)
        except RuntimeError as err:
            if batch == 1:
                raise
            print(f"batched backward failed ({str(err).splitlines()[0][:160]}); falling back to one row per pass",
                  flush=True)
            batch = 1
            continue
        g = g.reshape(b, n_all).double()
        diag[start: start + b] = g[torch.arange(b, device=x.device), rows]
        col += (g ** 2).sum(0)
        start += b
        if time.time() - last > 300:
            rate = start / (time.time() - t0)
            print(f"  exact rows {start}/{idx.numel()} ({rate:.1f}/s, eta {(idx.numel() - start) / rate / 60:.0f} min)",
                  flush=True)
            last = time.time()
    if x.is_cuda:
        torch.cuda.synchronize()
    return diag.cpu().numpy(), col.cpu().numpy(), time.time() - t0, batch


def moments(products):
    """d = mean of the products, c = mean of their squares, v = (c - d^2) / (S - 1), the unbiased noise variance of d.

    products: (S, ...) tensor.
    """
    S = products.shape[0]
    d = products.mean(0)
    c = (products ** 2).mean(0)
    v = (c - d ** 2).clamp(min=0) / max(S - 1, 1)
    return d, c, v


def trace_estimate(products):
    """Scalar Hutchinson trace: the plain dot-product mean, with no shrinkage and no clamp."""
    return products.sum(dim=tuple(range(1, products.ndim))).mean().item()
