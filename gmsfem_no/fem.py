"""Vectorised P1 finite elements on the fine triangulation (no FEniCS at run time).

The fine mesh is the 250x250 transfinite gmsh mesh stored in DOLFIN XML. All vertices lie
on the uniform grid {i/n, j/n}, which lets patches be addressed by integer grid indices.
Assembly reuses a fixed CSR pattern: one ``np.bincount`` per coefficient field.
"""
from __future__ import annotations

import re
from functools import cached_property
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_VERTEX_RE = re.compile(rb'<vertex index="(\d+)" x="([^"]+)" y="([^"]+)"')
_CELL_RE = re.compile(rb'<triangle index="(\d+)" v0="(\d+)" v1="(\d+)" v2="(\d+)"')
_DOF_RE = re.compile(rb'value="([^"]+)" cell_index="(\d+)"')


def read_dolfin_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse vertices (Nv, 2) and triangles (Nc, 3) from a DOLFIN XML mesh.

    Triangle vertices are sorted ascending, as DOLFIN does on load, so cell-local
    numbering matches FEniCS.
    """
    raw = Path(path).read_bytes()
    v = np.array(_VERTEX_RE.findall(raw), dtype=np.float64)
    c = np.array(_CELL_RE.findall(raw), dtype=np.int64)
    points = np.empty((len(v), 2))
    points[v[:, 0].astype(np.int64)] = v[:, 1:3]
    cells = np.empty((len(c), 3), dtype=np.int64)
    cells[c[:, 0]] = np.sort(c[:, 1:4], axis=1)
    return points, cells


def read_dg0_function(path: str | Path, n_cells: int) -> np.ndarray:
    """Read a DG0 field (one value per cell) from a DOLFIN XML function file."""
    rec = np.array(_DOF_RE.findall(Path(path).read_bytes()), dtype=np.float64)
    out = np.empty(n_cells)
    out[rec[:, 1].astype(np.int64)] = rec[:, 0]
    return out


class Mesh:
    """Triangulation of the unit square whose vertices sit on an (n+1)x(n+1) grid."""

    def __init__(self, points: np.ndarray, cells: np.ndarray):
        self.points = points
        self.cells = cells
        self.n = int(round(1.0 / np.diff(np.unique(points[:, 0].round(10)))[0]))
        assert len(points) == (self.n + 1) ** 2, "vertices must form a uniform grid"

    @classmethod
    def from_xml(cls, path: str | Path, cache: str | Path | None = None) -> "Mesh":
        if cache is not None and Path(cache).exists():
            d = np.load(cache)
            return cls(d["points"], d["cells"])
        points, cells = read_dolfin_mesh(path)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, points=points, cells=cells)
        return cls(points, cells)

    @property
    def n_vertices(self) -> int:
        return len(self.points)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @cached_property
    def grid_index(self) -> np.ndarray:
        """Integer grid coordinates (i, j) of every vertex, i along x."""
        return np.rint(self.points * self.n).astype(np.int64)

    @cached_property
    def vertex_at(self) -> np.ndarray:
        """(n+1, n+1) array: global vertex id of grid node (i, j)."""
        out = np.empty((self.n + 1, self.n + 1), dtype=np.int64)
        out[self.grid_index[:, 0], self.grid_index[:, 1]] = np.arange(self.n_vertices)
        return out

    @cached_property
    def boundary_vertices(self) -> np.ndarray:
        g = self.grid_index
        return np.flatnonzero((g == 0).any(1) | (g == self.n).any(1))

    def assembler(self) -> "P1Assembler":
        return P1Assembler(self.points, self.cells, self.boundary_vertices)


class P1Assembler:
    """Assembles P1 stiffness/mass matrices and load vectors with a cached CSR pattern.

    ``boundary`` lists the Dirichlet vertices used by :meth:`apply_dirichlet`.
    """

    def __init__(self, points: np.ndarray, cells: np.ndarray, boundary: np.ndarray | None = None):
        self.cells = cells
        self.n_vertices = n = len(points)
        self.boundary = np.empty(0, dtype=np.int64) if boundary is None else boundary
        p = points[cells]                                         # (Nc, 3, 2)
        d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        self.area = 0.5 * np.abs(det)
        g1 = np.stack([d2[:, 1], -d2[:, 0]], 1) / det[:, None]   # grad of barycentric coords
        g2 = np.stack([-d1[:, 1], d1[:, 0]], 1) / det[:, None]
        grads = np.stack([-g1 - g2, g1, g2], 1)                   # (Nc, 3, 2)
        self._k_local = self.area[:, None, None] * np.einsum("cid,cjd->cij", grads, grads)
        self._m_local = self.area[:, None, None] / 12.0 * (np.ones((3, 3)) + np.eye(3))

        rows = np.repeat(cells, 3, axis=1).ravel()
        cols = np.tile(cells, (1, 3)).ravel()
        keys, self._slot = np.unique(rows * n + cols, return_inverse=True)
        self._indptr = np.searchsorted(keys // n, np.arange(n + 1)).astype(np.int32)
        self._indices = (keys % n).astype(np.int32)
        self._shape = (n, n)

        r = np.repeat(np.arange(n), np.diff(self._indptr))
        bnd = np.zeros(n, dtype=bool)
        bnd[self.boundary] = True
        offdiag = r != self._indices
        self._bc_rows = bnd[r] & offdiag                               # off-diagonals of boundary rows
        self._bc_rows_cols = (bnd[r] | bnd[self._indices]) & offdiag   # ... of boundary rows and columns
        self._bc_diag = np.flatnonzero(bnd[r] & ~offdiag)              # diagonal slots of boundary rows
        self._bc_diag_value = np.bincount(cells.ravel(), minlength=n)[r[self._bc_diag]].astype(np.float64)

    def _csr(self, local: np.ndarray) -> sp.csr_matrix:
        data = np.bincount(self._slot, weights=local.ravel(), minlength=len(self._indices))
        return sp.csr_matrix((data, self._indices.copy(), self._indptr.copy()), shape=self._shape)

    def stiffness(self, kappa: np.ndarray | float = 1.0) -> sp.csr_matrix:
        """∫ κ ∇u·∇v for κ given per cell (DG0) or as a scalar."""
        return self._csr(np.asarray(kappa, dtype=np.float64).reshape(-1, 1, 1) * self._k_local)

    def mass(self, weight: np.ndarray | float = 1.0) -> sp.csr_matrix:
        """∫ w u v for w given per cell (DG0) or as a scalar."""
        return self._csr(np.asarray(weight, dtype=np.float64).reshape(-1, 1, 1) * self._m_local)

    def load(self, f: np.ndarray | float = 1.0) -> np.ndarray:
        """∫ f v for f constant or a P1 nodal vector."""
        if np.ndim(f) == 0:
            w = np.repeat(self.area * (float(f) / 3.0), 3)
            return np.bincount(self.cells.ravel(), weights=w, minlength=self.n_vertices)
        return self.mass() @ np.asarray(f, dtype=np.float64)

    def apply_dirichlet(self, A: sp.csr_matrix, b: np.ndarray,
                        symmetric: bool = False) -> tuple[sp.csr_matrix, np.ndarray]:
        """Homogeneous Dirichlet BC on a matrix produced by this assembler.

        ``symmetric=False``: off-diagonals of boundary rows are zeroed, the diagonal and the
        columns are kept (row elimination of the reference diffusion pipeline).
        ``symmetric=True``: boundary rows and columns are zeroed and the diagonal is set to the
        number of cells sharing the vertex (DOLFIN ``assemble_system``).
        Both give the same fine solution; they differ once A is projected onto a coarse
        space whose basis functions do not vanish on the boundary.
        """
        A = A.copy()
        if symmetric:
            A.data[self._bc_rows_cols] = 0.0
            A.data[self._bc_diag] = self._bc_diag_value
        else:
            A.data[self._bc_rows] = 0.0
        b = b.copy()
        b[self.boundary] = 0.0
        return A, b


def solve_fine(A: sp.csr_matrix, b: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Reference fine-scale solution with u = 0 on the boundary (SPD interior solve)."""
    interior = np.setdiff1d(np.arange(A.shape[0]), boundary)
    u = np.zeros(A.shape[0])
    A_ii = A[interior][:, interior].tocsc()
    u[interior] = spla.splu(A_ii, permc_spec="MMD_AT_PLUS_A").solve(b[interior])
    return u


def relative_errors(u_ms: np.ndarray, u_ref: np.ndarray, M: sp.csr_matrix, K: sp.csr_matrix) -> tuple[float, float]:
    """Relative L2 and H1-seminorm errors in percent (FEniCS-equivalent P1 integrals)."""
    e = u_ms - u_ref
    l2 = np.sqrt((e @ (M @ e)) / (u_ref @ (M @ u_ref))) * 100.0
    h1 = np.sqrt((e @ (K @ e)) / (u_ref @ (K @ u_ref))) * 100.0
    return float(l2), float(h1)
