"""Coarse grid, GMsFEM patches (oversampled neighbourhoods of coarse nodes) and NO inputs.

Every coarse node w = j*(Nc+1) + i owns the patch [x_i - H, x_i + H] x [y_j - H, y_j + H]
clipped to the unit square. Patches come in three shapes, each with its own operator:

* ``full``   - interior node,     (2m+1) x (2m+1) grid points
* ``half``   - boundary edge node, (2m+1) x (m+1), axis 1 points away from the boundary
* ``corner`` - corner node,        (m+1) x (m+1), axes point away from the corner

Patches are rotated/reflected into this canonical orientation exactly as in the dataset the
reference checkpoints were trained on; coordinate channels are always (x, y).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .fem import Mesh

KINDS = ("full", "half", "corner")


@dataclass(frozen=True)
class Patch:
    node: int
    kind: str
    grid: np.ndarray        # (2, P0, P1) fine-grid indices (i, j) in canonical orientation


class CoarseGrid:
    def __init__(self, mesh: Mesh, n_coarse: int = 10):
        assert mesh.n % n_coarse == 0
        self.mesh = mesh
        self.nc = n_coarse
        self.m = mesh.n // n_coarse
        self.n_nodes = (n_coarse + 1) ** 2

    # --- node sets (fixed ordering: the patch axis of data/bases_*.npy) --------------------
    def nodes(self, kind: str) -> list[int]:
        n = self.nc
        if kind == "corner":
            return [0, n, (n + 1) * n, (n + 1) * (n + 1) - 1]
        if kind == "half":
            return (list(range(1, n)) + [(n + 1) * i + n for i in range(1, n)]
                    + [(n + 1) * i for i in range(1, n)] + [(n + 1) * n + i for i in range(1, n)])
        if kind == "full":
            return [(n + 1) * i + j for i in range(1, n) for j in range(1, n)]
        raise ValueError(kind)

    def kind_of(self, w: int) -> str:
        j, i = divmod(w, self.nc + 1)
        on_x, on_y = i in (0, self.nc), j in (0, self.nc)
        return "corner" if on_x and on_y else "half" if on_x or on_y else "full"

    # --- patch geometry ---------------------------------------------------------------------
    def patch(self, w: int) -> Patch:
        n, m, nc = self.mesh.n, self.m, self.nc
        j, i = divmod(w, nc + 1)
        cx, cy = i * m, j * m
        xs = np.arange(max(cx - m, 0), min(cx + m, n) + 1)
        ys = np.arange(max(cy - m, 0), min(cy + m, n) + 1)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        kind = self.kind_of(w)

        if kind == "corner":
            sx, sy = (-1 if i == nc else 1), (-1 if j == nc else 1)

            def canon(a): return a[::sx, ::sy]
        elif kind == "half":
            if j == 0:                       # bottom
                def canon(a): return a
            elif i == 0:                     # left
                def canon(a): return a.T
            elif i == nc:                    # right
                def canon(a): return a.T[:, ::-1]
            else:                            # top
                def canon(a): return a[:, ::-1]
        else:
            def canon(a): return a
        grid = np.stack([canon(gx), canon(gy)])
        return Patch(node=w, kind=kind, grid=np.ascontiguousarray(grid))

    @cached_property
    def patches(self) -> list[Patch]:
        return [self.patch(w) for w in range(self.n_nodes)]

    def vertices(self, w: int) -> np.ndarray:
        """Global vertex ids of the patch grid points, canonical orientation, shape (P0, P1)."""
        g = self.patches[w].grid
        return self.mesh.vertex_at[g[0], g[1]]

    def coordinates(self, w: int) -> np.ndarray:
        """(2, P0, P1) float32 channels (x, y) of the patch points, fed to the operator."""
        lin = np.linspace(0.0, 1.0, self.mesh.n + 1)
        return lin[self.patches[w].grid].astype(np.float32)

    @cached_property
    def _cell_grid(self) -> np.ndarray:
        return self.mesh.grid_index[self.mesh.cells]              # (Nc, 3, 2)

    def cells(self, w: int) -> np.ndarray:
        """Ascending ids of fine cells inside the patch (the FEniCS SubMesh cells)."""
        g = self._cell_grid
        lo = self.patches[w].grid.reshape(2, -1).min(1)
        hi = self.patches[w].grid.reshape(2, -1).max(1)
        inside = ((g >= lo) & (g <= hi)).all(axis=(1, 2))
        return np.flatnonzero(inside)

    def feature_cells(self, w: int) -> np.ndarray:
        """For each patch grid point, the DG0 cell whose value the CG1 interpolant takes.

        DOLFIN interpolates DG0 -> CG1 cell by cell, so a vertex keeps the value of the
        last (highest-index) SubMesh cell containing it. Shape (P0, P1).
        """
        pc = self.cells(w)
        owner = np.full(self.mesh.n_vertices, -1, dtype=np.int64)
        owner[self.mesh.cells[pc].ravel()] = np.repeat(pc, 3)
        return owner[self.vertices(w)]

    @cached_property
    def _feature_cells(self) -> dict[str, np.ndarray]:
        return {k: np.stack([self.feature_cells(w) for w in self.nodes(k)]) for k in KINDS}

    def features(self, kappa_cells: np.ndarray, kind: str) -> np.ndarray:
        """Raw coefficient on patches: kappa_cells (..., Nc) -> (..., n_patches, 1, P0, P1)."""
        return np.asarray(kappa_cells)[..., self._feature_cells[kind]][..., None, :, :]

    def all_coordinates(self, kind: str) -> np.ndarray:
        return np.stack([self.coordinates(w) for w in self.nodes(kind)])

    def all_vertices(self, kind: str) -> np.ndarray:
        return np.stack([self.vertices(w) for w in self.nodes(kind)])

    def hat(self, w: int) -> np.ndarray:
        """Bilinear coarse hat function of node w at the patch grid points (P0, P1)."""
        j, i = divmod(w, self.nc + 1)
        g = self.patches[w].grid
        phi = (np.clip(1 - np.abs(g[0] - i * self.m) / self.m, 0, None)
               * np.clip(1 - np.abs(g[1] - j * self.m) / self.m, 0, None))
        phi[phi <= 1e-5] = 0.0
        return phi
