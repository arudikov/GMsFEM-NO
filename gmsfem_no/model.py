"""Factorized Fourier Neural Operator (F-FNO) and checkpoint I/O."""
from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn import gelu

# per-axis mode mixing: weights (out, in, modes) applied along spatial axis 1 or 2
_MIX = {1: "oim,imy->omy", 2: "oim,ixm->oxm"}


def _he_conv(key, c_in, c_out):
    conv = eqx.nn.Conv(2, c_in, c_out, 1, key=key)
    conv = eqx.tree_at(lambda c: c.weight, conv, conv.weight * jnp.sqrt(2.0 / c_in))
    return eqx.tree_at(lambda c: c.bias, conv, jnp.zeros_like(conv.bias))


class FFNO(eqx.Module):
    """u = enc([x, κ]);  u += gelu(W2 gelu(W1 Σ_d K_d u))  (n_layers);  out = dec(u).

    K_d is a truncated spectral convolution along spatial axis d (factorized FNO), with
    ``modes[d]`` retained rFFT modes.
    """
    encoder: eqx.nn.Conv
    decoder: eqx.nn.Conv
    convs1: list
    convs2: list
    A: list                                   # [A_x, A_y], each complex (L, C, C, modes_d)

    def __init__(self, n_layers: int, features: tuple[int, int, int], modes, key):
        c_in, c, c_out = features
        modes = (modes, modes) if isinstance(modes, int) else tuple(modes)
        keys = jax.random.split(key, 4 + 2 * n_layers)
        self.encoder = _he_conv(keys[-1], c_in, c)
        self.decoder = _he_conv(keys[-2], c, c_out)
        self.convs1 = [_he_conv(k, c, c) for k in keys[:n_layers]]
        self.convs2 = [_he_conv(k, c, c) for k in keys[n_layers:2 * n_layers]]
        self.A = [jax.random.normal(keys[-3 - d], (n_layers, c, c, m), dtype=jnp.complex64) * jnp.sqrt(2.0 / c)
                  for d, m in enumerate(modes)]

    def __call__(self, kappa: jax.Array, coords: jax.Array) -> jax.Array:
        u = self.encoder(jnp.concatenate([coords, kappa], 0))
        for layer, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):
            u = u + gelu(conv2(gelu(conv1(self._spectral(u, [A[layer] for A in self.A])))))
        return self.decoder(u)

    @staticmethod
    def _spectral(v: jax.Array, weights) -> jax.Array:
        out = 0.0
        for axis, W in enumerate(weights, start=1):
            vh = jax.lax.slice_in_dim(jnp.fft.rfft(v, axis=axis), 0, W.shape[-1], axis=axis)
            out = out + jnp.fft.irfft(jnp.einsum(_MIX[axis], W, vh), n=v.shape[axis], axis=axis)
        return out


# --- checkpoints -------------------------------------------------------------------------

def save(path: str | Path, model: FFNO, meta: dict) -> None:
    """One file: a JSON header line (hyper-parameters, κ scale, history) + model leaves."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write((json.dumps(meta) + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(path: str | Path) -> tuple[FFNO, dict]:
    with open(path, "rb") as f:
        meta = json.loads(f.readline().decode())
        like = FFNO(meta["n_layers"], tuple(meta["features"]), tuple(meta["modes"]), jax.random.PRNGKey(0))
        return eqx.tree_deserialise_leaves(f, like), meta


def count_params(model: FFNO) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return int(sum(x.size * (2 if jnp.iscomplexobj(x) else 1) for x in leaves))


@eqx.filter_jit
def _forward(model: FFNO, kappa, coords):
    return jax.vmap(model)(kappa, coords)


def predict(model: FFNO, kappa, coords, batch: int = 512) -> jax.Array:
    """Batched inference: kappa (N, 1, P0, P1) normalised, coords (N, 2, P0, P1)."""
    outs = [_forward(model, kappa[i:i + batch], coords[i:i + batch]) for i in range(0, len(kappa), batch)]
    return jnp.concatenate(outs)
