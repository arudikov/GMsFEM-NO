"""Training of one patch-kind operator with the subspace loss.

The whole dataset lives on the GPU; an epoch is a single jitted ``lax.scan`` over
minibatches (no host round-trips). Coordinates are stored once per patch and gathered by
index. Targets are pre-orthonormalised, so a step needs one Cholesky-QR per sample.
"""
from __future__ import annotations

import time
from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import ModelConfig, TrainConfig
from .loss import orthonormal_targets, subspace_loss
from .model import FFNO, count_params


def _to_device(split: dict, scale: float):
    n_kle, n_p = split["features"].shape[:2]
    X = jnp.asarray(split["features"].reshape(n_kle * n_p, *split["features"].shape[2:]) / scale)
    C = jnp.asarray(split["coordinates"])
    T = split["targets"]
    Q = orthonormal_targets(jnp.asarray(T.reshape(n_kle * n_p, *T.shape[2:])))
    return X, C, Q


def _batch_loss(params, static, X, C, Q, idx):
    model = eqx.combine(params, static)
    pred = jax.vmap(model)(X[idx], C[idx % C.shape[0]])
    return jnp.mean(jax.vmap(subspace_loss)(pred, Q[idx]))


def fit(train: dict, test: dict, mcfg: ModelConfig, tcfg: TrainConfig, log=print, writer=None):
    """Returns (model, meta) where meta holds everything needed to reload and use the model."""
    scale = float(np.abs(train["features"]).max())
    t0 = time.time()
    Xtr, Ctr, Qtr = _to_device(train, scale)
    Xte, Cte, Qte = _to_device(test, scale)
    n_basis = train["targets"].shape[2]
    log(f"data on device in {time.time() - t0:.1f}s: train {Xtr.shape[0]}, test {Xte.shape[0]} samples, "
        f"patch {Xtr.shape[2:]}, κ scale {scale:.2f}")

    features = (Ctr.shape[1] + Xtr.shape[1], mcfg.width, n_basis)
    model = FFNO(mcfg.n_layers, features, mcfg.modes, jax.random.PRNGKey(mcfg.seed))
    params, static = eqx.partition(model, eqx.is_array)
    log(f"FFNO: {count_params(model):,} real parameters")

    B = tcfg.batch_size
    steps_per_epoch = Xtr.shape[0] // B
    schedule = optax.cosine_decay_schedule(tcfg.lr, tcfg.epochs * steps_per_epoch, alpha=tcfg.lr_final_fraction)
    opt = optax.adamw(schedule, weight_decay=tcfg.weight_decay)
    opt_state = opt.init(params)
    grad_fn = jax.value_and_grad(_batch_loss)

    @jax.jit
    def train_epoch(params, opt_state, key, X, C, Q):
        perm = jax.random.permutation(key, X.shape[0])[:steps_per_epoch * B].reshape(steps_per_epoch, B)

        def step(carry, idx):
            params, opt_state = carry
            loss, g = grad_fn(params, static, X, C, Q, idx)
            g = jax.tree_util.tree_map(jnp.conj, g)       # complex weights: descend along conj(grad)
            updates, opt_state = opt.update(g, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), loss

        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), perm)
        return params, opt_state, losses.mean()

    eval_B = 256
    n_eval = Xte.shape[0] // eval_B * eval_B

    @jax.jit
    def evaluate(params, X, C, Q):
        idx = jnp.arange(n_eval).reshape(-1, eval_B)
        return jax.lax.map(lambda i: _batch_loss(params, static, X, C, Q, i), idx).mean()

    history = {"epoch": [], "train": [], "test": []}
    key = jax.random.PRNGKey(tcfg.seed)
    t0 = time.time()
    for epoch in range(1, tcfg.epochs + 1):
        key, sub = jax.random.split(key)
        params, opt_state, train_loss = train_epoch(params, opt_state, sub, Xtr, Ctr, Qtr)
        if epoch % tcfg.eval_every == 0 or epoch in (1, tcfg.epochs):
            train_loss, test_loss = float(train_loss), float(evaluate(params, Xte, Cte, Qte))
            history["epoch"].append(epoch)
            history["train"].append(train_loss)
            history["test"].append(test_loss)
            el = time.time() - t0
            log(f"epoch {epoch:4d}  train {train_loss:.5f}  test {test_loss:.5f}  "
                f"{el / epoch:.2f}s/epoch  ETA {el / epoch * (tcfg.epochs - epoch) / 60:.0f} min")
            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/test", test_loss, epoch)

    meta = {"n_layers": mcfg.n_layers, "features": list(features), "modes": list(mcfg.modes),
            "kappa_scale": scale, "model_config": asdict(mcfg), "train_config": asdict(tcfg),
            "train_kle": [int(train["kle_ids"][0]), int(train["kle_ids"][-1])],
            "history": history, "train_time_s": time.time() - t0}
    return eqx.combine(params, static), meta
