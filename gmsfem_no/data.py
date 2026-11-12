"""Coefficient fields and operator-learning datasets."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from .config import Paths
from .fem import Mesh, read_dg0_function
from .patches import CoarseGrid


def load_mesh(paths: Paths) -> Mesh:
    return Mesh.from_xml(paths.mesh_xml, cache=paths.cache_dir / "mesh.npz")


def kappa_store(paths: Paths, mesh: Mesh, workers: int = 32) -> np.ndarray:
    """All KLE coefficients as a read-only memmap (max_id + 1, n_cells); row index = KLE id.

    Row 0 is unused (KLE files are numbered from 1). Built once from the XML files.
    """
    cache = paths.cache_dir / "kappa_cells.npy"
    if not cache.exists():
        ids = sorted(int(p.stem) for p in Path(paths.kle_dir).glob("*.xml"))
        tmp = cache.with_name("kappa_cells.tmp.npy")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float64, shape=(max(ids) + 1, mesh.n_cells))
        read = partial(read_dg0_function, n_cells=mesh.n_cells)
        with ProcessPoolExecutor(workers) as ex:
            for i, row in zip(ids, ex.map(read, [Path(paths.kle_dir) / f"{i}.xml" for i in ids], chunksize=8)):
                out[i] = row
        out.flush()
        del out
        tmp.rename(cache)
    return np.load(cache, mmap_mode="r")


def load_split(paths: Paths, cg: CoarseGrid, kappa: np.ndarray, kind: str, kle_ids,
               with_targets: bool = True) -> dict:
    """Arrays for one patch kind and a set of KLE ids.

    features    (n_kle, n_patch, 1, P0, P1)        raw κ at the patch grid points
    coordinates (n_patch, 2, P0, P1)               shared by all samples
    targets     (n_kle, n_patch, n_basis, P0, P1)  GMsFEM bases (generated offline)
    """
    kle_ids = np.asarray(kle_ids)
    out = {"kle_ids": kle_ids,
           "features": cg.features(kappa[kle_ids], kind).astype(np.float32),
           "coordinates": cg.all_coordinates(kind)}
    if with_targets:
        out["targets"] = np.asarray(np.load(paths.bases(kind), mmap_mode="r")[kle_ids])
    return out
