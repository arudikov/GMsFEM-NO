"""Online stage: GMsFEM with operator-predicted bases, u|∂Ω = 0, constant source f.

    python scripts/evaluate.py --gpu 0                          # diffusion, checkpoints/ffno_{kind}.eqx
    python scripts/evaluate.py --gpu 0 --problem richards       # -div(κ/(1+1e5 u) ∇u) = f, Picard

Writes per-KLE relative L2 / H1-seminorm errors (%) and Picard iteration counts
to results/<name>.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per worker
    os.environ.setdefault(_v, "1")
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gmsfem_no.config import Paths, Problem
from gmsfem_no.patches import KINDS
from gmsfem_no.problems import PROBLEMS

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default="0")
    p.add_argument("--ckpt", default="checkpoints/ffno_{kind}.eqx", help="pattern with {kind}")
    p.add_argument("--kle", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"), help="inclusive range")
    p.add_argument("--problem", choices=sorted(PROBLEMS), default="diffusion")
    p.add_argument("--f", type=float, default=1.0, help="constant source term")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--name", default=None, help="output name (results/<name>.csv)")
    return p.parse_args()


# ---------------------------------------------------------------- online workers (CPU only)
_W = {}


def _init_worker(pred_files: dict, problem):
    from gmsfem_no.data import kappa_store, load_mesh
    from gmsfem_no.online import OnlineGMsFEM
    from gmsfem_no.patches import CoarseGrid
    paths, prob = Paths(), Problem()
    mesh = load_mesh(paths)
    _W.update(kappa=kappa_store(paths, mesh), problem=problem,
              solver=OnlineGMsFEM(CoarseGrid(mesh, prob.n_coarse), prob.n_basis),
              preds={k: np.load(v, mmap_mode="r") for k, v in pred_files.items()})


def _solve_one(args):
    row, kle = args
    sol = _W["solver"].solve(np.asarray(_W["kappa"][kle]), {k: v[row] for k, v in _W["preds"].items()},
                             _W["problem"])
    return kle, sol.l2, sol.h1, sol.iters_ms, sol.iters_ref


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax.numpy as jnp

    from gmsfem_no import model as M
    from gmsfem_no.data import kappa_store, load_mesh
    from gmsfem_no.patches import CoarseGrid

    paths, prob = Paths(), Problem()
    paths.results.mkdir(parents=True, exist_ok=True)
    mesh = load_mesh(paths)
    cg = CoarseGrid(mesh, prob.n_coarse)
    kappa = kappa_store(paths, mesh)
    kles = np.arange(args.kle[0], args.kle[1] + 1) if args.kle else np.array(prob.test_kle)
    problem = PROBLEMS[args.problem](f=args.f)
    name = args.name or f"ffno_{problem.name}_f{args.f:g}"

    # 1) operator inference for every patch of every KLE (GPU, batched)
    t0 = time.time()
    pred_files = {}
    for kind in KINDS:
        feats = cg.features(kappa[kles], kind).astype(np.float32)          # (n_kle, n_p, 1, P0, P1)
        model, meta = M.load(Path(args.ckpt.format(kind=kind)))
        scale = meta["kappa_scale"]
        n_kle, n_p = feats.shape[:2]
        coords = np.broadcast_to(cg.all_coordinates(kind), (n_kle, *cg.all_coordinates(kind).shape))
        pred = M.predict(model, jnp.asarray(feats.reshape(n_kle * n_p, *feats.shape[2:]) / scale),
                         jnp.asarray(coords.reshape(n_kle * n_p, *coords.shape[2:])))
        out = paths.results / f"{name}_pred_{kind}.npy"
        np.save(out, np.asarray(pred, dtype=np.float32).reshape(n_kle, n_p, *pred.shape[1:]))
        pred_files[kind] = str(out)
        print(f"[{kind:6s}] {M.count_params(model):,} params, κ scale {scale:.2f}, "
              f"{n_kle * n_p} patches predicted", flush=True)
    t_pred = time.time() - t0

    # 2) online coarse solves (CPU, one process per worker)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"),
                             initializer=_init_worker, initargs=(pred_files, problem)) as ex:
        res = sorted(ex.map(_solve_one, enumerate(kles.tolist()), chunksize=2))
    t_online = time.time() - t0
    for f in pred_files.values():
        os.remove(f)

    res = np.array(res)
    out = paths.results / f"{name}.csv"
    np.savetxt(out, res, delimiter=",", header="kle,L2_error,H1_error,iters_ms,iters_ref", comments="",
               fmt=["%d", "%.10g", "%.10g", "%d", "%d"])
    l2, h1 = res[:, 1], res[:, 2]
    print(f"\n{name}: {len(res)} KLE, inference {t_pred:.1f}s, online {t_online:.1f}s "
          f"({t_online / len(res) * args.workers:.2f}s per KLE per worker)")
    print(f"  L2 error, %: mean {l2.mean():.4f}  median {np.median(l2):.4f}  max {l2.max():.4f}")
    print(f"  H1 error, %: mean {h1.mean():.4f}  median {np.median(h1):.4f}  max {h1.max():.4f}")
    print(f"  iterations: multiscale mean {res[:, 3].mean():.1f} (max {res[:, 3].max():.0f}), "
          f"fine mean {res[:, 4].mean():.1f} (max {res[:, 4].max():.0f})")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
