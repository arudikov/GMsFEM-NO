"""Offline stage: GMsFEM multiscale bases (targets for the operators) for every KLE sample.

    python scripts/make_dataset.py --workers 96            # all KLE ids, resumable

Writes data/bases_{full,half,corner}.npy with shape (max_kle + 1, n_patch, n_basis, P0, P1)
(row = KLE id) and data/bases_done.npy (bool mask of finished rows).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gmsfem_no.config import Paths, Problem
from gmsfem_no.data import kappa_store, load_mesh
from gmsfem_no.offline import LocalProblem
from gmsfem_no.patches import KINDS, CoarseGrid

_W = {}


def _init_worker():
    paths, prob = Paths(), Problem()
    mesh = load_mesh(paths)
    cg = CoarseGrid(mesh, prob.n_coarse)
    _W.update(cg=cg, nb=prob.n_basis, kappa=kappa_store(paths, mesh),
              problems={w: LocalProblem(cg, w) for w in range(cg.n_nodes)},
              out={k: np.load(paths.bases(k), mmap_mode="r+") for k in KINDS})


def _make_one(kle: int) -> int:
    cg, kappa = _W["cg"], np.asarray(_W["kappa"][kle])
    for kind in KINDS:
        _W["out"][kind][kle] = np.stack([_W["problems"][w].basis(kappa, _W["nb"])[0] for w in cg.nodes(kind)])
        _W["out"][kind].flush()
    return kle


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--kle", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"))
    args = p.parse_args()

    paths, prob = Paths(), Problem()
    mesh = load_mesh(paths)
    cg = CoarseGrid(mesh, prob.n_coarse)
    kappa = kappa_store(paths, mesh)
    n_rows = kappa.shape[0]
    for kind in KINDS:
        if not paths.bases(kind).exists():
            shape = (n_rows, len(cg.nodes(kind)), prob.n_basis, *cg.patches[cg.nodes(kind)[0]].grid.shape[1:])
            np.lib.format.open_memmap(paths.bases(kind), mode="w+", dtype=np.float32, shape=shape).flush()
    done_file = paths.data_dir / "bases_done.npy"
    done = np.load(done_file) if done_file.exists() else np.zeros(n_rows, dtype=bool)

    ids = range(args.kle[0], args.kle[1] + 1) if args.kle else range(1, n_rows)
    todo = [k for k in ids if not done[k]]
    print(f"{len(todo)} KLE samples to process with {args.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init_worker) as ex:
        futures = [ex.submit(_make_one, k) for k in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            done[fut.result()] = True
            if i % 50 == 0 or i == len(todo):
                np.save(done_file, done)
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el:.0f}s elapsed, ETA {el / i * (len(todo) - i):.0f}s", flush=True)
    print(f"done: {int(done.sum())} KLE samples in data/bases_*.npy")


if __name__ == "__main__":
    main()
