"""Subspace loss: distance between span(prediction) and span(GMsFEM basis).

    L = k - ||Q_t^T Q_p||_F^2 = k - tr(P_t P_p) ∈ [0, k],

with Q_t, Q_p orthonormal bases of the target / predicted k-dimensional subspaces. The
target factor Q_t is computed once (float32 QR on device) and stored instead of the raw
targets. The prediction side uses Cholesky-QR, which needs only k x k Gram matrices:
tr(P_t P_p) = ||L^{-1} C^T||_F^2,  C = Q_t^T X,  L L^T = X^T X.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HI = jax.lax.Precision.HIGHEST


def orthonormal_targets(targets: jax.Array, chunk: int = 2048) -> jax.Array:
    """(N, k, P0, P1) -> (N, P0*P1, k) orthonormal columns spanning the same subspace."""
    qr = jax.jit(jax.vmap(lambda t: jnp.linalg.qr(t.reshape(t.shape[0], -1).T)[0]))
    return jnp.concatenate([qr(targets[i:i + chunk]) for i in range(0, len(targets), chunk)])


def subspace_loss(pred: jax.Array, q_target: jax.Array, eps: float = 1e-6) -> jax.Array:
    """pred (k, P0, P1), q_target (P0*P1, k) -> scalar in [0, k]."""
    X = pred.reshape(pred.shape[0], -1).T                        # (P, k)
    G = jnp.matmul(X.T, X, precision=_HI)
    G = G + eps * jnp.trace(G) / G.shape[0] * jnp.eye(G.shape[0])
    C = jnp.matmul(q_target.T, X, precision=_HI)                 # (k, k)
    Lc = jnp.linalg.cholesky(G)
    Y = jax.scipy.linalg.solve_triangular(Lc, C.T, lower=True)
    return q_target.shape[1] - jnp.sum(Y * Y)
