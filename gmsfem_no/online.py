"""Online GMsFEM: coarse solve in the span of (predicted) multiscale basis functions.

R (N_coarse x N_fine) holds one basis function per row; every linear solve of a problem
(one for diffusion, one per Picard step for Richards) becomes A_H = R A R^T, b_H = R b,
u_ms = R^T A_H^{-1} b_H. Boundary handling is defined by the problem (see problems.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .fem import relative_errors, solve_fine
from .patches import KINDS, CoarseGrid
from .problems import Diffusion


@dataclass
class Solution:
    u_ms: np.ndarray
    u_ref: np.ndarray
    l2: float
    h1: float
    iters_ms: int
    iters_ref: int


class OnlineGMsFEM:
    def __init__(self, cg: CoarseGrid, n_basis: int):
        self.cg, self.nb = cg, n_basis
        self.asm = cg.mesh.assembler()
        self.M = self.asm.mass()
        self.K = self.asm.stiffness(1.0)

        # Fixed sparsity of R: row w*nb + k, columns = patch vertices of node w.
        cols = [np.tile(cg.vertices(w).ravel(), n_basis) for w in range(cg.n_nodes)]
        rows = [np.repeat(w * n_basis + np.arange(n_basis), cg.vertices(w).size) for w in range(cg.n_nodes)]
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        n_rows, n_cols = cg.n_nodes * n_basis, cg.mesh.n_vertices
        self._R = self._pattern(rows, cols, n_rows, n_cols)        # R   in CSR
        self._RT = self._pattern(cols, rows, n_cols, n_rows)       # R^T in CSR

    @staticmethod
    def _pattern(rows, cols, n_rows, n_cols):
        order = np.lexsort((cols, rows))
        indptr = np.searchsorted(rows[order], np.arange(n_rows + 1)).astype(np.int32)
        return order, cols[order].astype(np.int32), indptr, (n_rows, n_cols)

    def projection(self, bases: dict[str, np.ndarray]) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """R and R^T from bases[kind]: (n_patch_kind, n_basis, P0, P1), canonical orientation."""
        per_node = [None] * self.cg.n_nodes
        for kind in KINDS:
            for w, b in zip(self.cg.nodes(kind), bases[kind]):
                per_node[w] = np.asarray(b, dtype=np.float64).ravel()
        data = np.concatenate(per_node)
        return tuple(sp.csr_matrix((data[order], idx, ptr), shape=shape)
                     for order, idx, ptr, shape in (self._R, self._RT))

    def solve(self, kappa_cells: np.ndarray, bases: dict[str, np.ndarray], problem=Diffusion(),
              u_ref: np.ndarray | None = None) -> Solution:
        R, RT = self.projection(bases)

        def coarse(A, b):
            return RT @ np.linalg.solve((R @ (A @ RT)).toarray(), R @ b)

        def fine(A, b):
            return solve_fine(A, b, self.cg.mesh.boundary_vertices)

        u_ms, it_ms = problem.solve(self.asm, kappa_cells, coarse)
        it_ref = 0
        if u_ref is None:
            u_ref, it_ref = problem.solve(self.asm, kappa_cells, fine)
        l2, h1 = relative_errors(u_ms, u_ref, self.M, self.K)
        return Solution(u_ms, u_ref, l2, h1, it_ms, it_ref)
