"""Offline GMsFEM: multiscale partition of unity and spectral basis on every patch.

For coarse node w with patch ω_w:
  1. χ_w solves -div(κ∇χ) = 0 in each coarse cell of ω_w, χ_w = bilinear hat on the
     coarse edges (patch boundary and the two lines through the node);
  2. φ_k are the smallest eigenpairs of the Neumann problem ∫κ∇φ·∇v = λ ∫κ φ v on ω_w;
  3. the basis is ψ_k = χ_w φ_k, k < n_basis.
Only span{ψ_k} matters (the operator is trained with a subspace loss), so eigenvector
scaling and sign are irrelevant.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla

from .fem import P1Assembler
from .patches import CoarseGrid


class LocalProblem:
    """Pre-assembled geometry of one patch; the coefficient changes per sample."""

    def __init__(self, cg: CoarseGrid, w: int):
        mesh = cg.mesh
        self.w = w
        self.shape = cg.patches[w].grid.shape[1:]
        verts = cg.vertices(w).ravel()                       # local id = canonical flat index
        local_of = np.full(mesh.n_vertices, -1, dtype=np.int64)
        local_of[verts] = np.arange(len(verts))
        self.cells = cg.cells(w)
        self.asm = P1Assembler(mesh.points[verts], local_of[mesh.cells[self.cells]])

        j, i = divmod(w, cg.nc + 1)
        g = cg.patches[w].grid.reshape(2, -1)
        lo, hi = g.min(1, keepdims=True), g.max(1, keepdims=True)
        on_edges = ((g == lo) | (g == hi)).any(0) | (g[0] == i * cg.m) | (g[1] == j * cg.m)
        self.fixed = np.flatnonzero(on_edges)
        self.free = np.flatnonzero(~on_edges)
        self.hat = cg.hat(w).ravel()

    def basis(self, kappa_cells: np.ndarray, n_basis: int) -> tuple[np.ndarray, np.ndarray]:
        k = kappa_cells[self.cells]
        A = self.asm.stiffness(k)
        S = self.asm.mass(k)

        pou = self.hat.copy()
        A_ff = A[self.free][:, self.free].tocsc()
        pou[self.free] = spla.spsolve(A_ff, -(A[self.free][:, self.fixed] @ self.hat[self.fixed]))

        # Neumann spectral problem; shift-invert around a small negative shift (A is singular).
        sigma = -1e-3 * (A.diagonal().sum() / S.diagonal().sum())
        try:
            lam, phi = spla.eigsh(A.tocsc(), k=n_basis, M=S.tocsc(), sigma=sigma, which="LM")
        except spla.ArpackNoConvergence:
            lam, phi = scipy.linalg.eigh(A.toarray(), S.toarray(), subset_by_index=[0, n_basis - 1])
        order = np.argsort(lam)
        psi = (phi[:, order] * pou[:, None]).T
        return psi.reshape(n_basis, *self.shape), lam[order]


def patch_bases(cg: CoarseGrid, problems: dict[int, LocalProblem], kappa_cells: np.ndarray,
                kind: str, n_basis: int) -> np.ndarray:
    """Targets for every patch of one kind: (n_patches, n_basis, P0, P1) float32."""
    return np.stack([problems[w].basis(kappa_cells, n_basis)[0] for w in cg.nodes(kind)]).astype(np.float32)
