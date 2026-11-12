"""Model problems for the online stage, all with u = 0 on ∂Ω.

    diffusion:  -div(κ ∇u) = f
    richards:   -div(κ k(u) ∇u) = f,   k(u) = 1 / (1 + α u),   Picard iterations from u = 0

``solve(asm, kappa, linear_solve)`` runs the same algorithm for the fine reference and for the
multiscale solution; only ``linear_solve`` differs (fine LU vs. coarse Galerkin solve).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .fem import P1Assembler

LinearSolve = Callable[[object, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Diffusion:
    f: float = 1.0
    symmetric_bc: bool = False

    name = "diffusion"

    def solve(self, asm: P1Assembler, kappa: np.ndarray, linear_solve: LinearSolve) -> tuple[np.ndarray, int]:
        A, b = asm.apply_dirichlet(asm.stiffness(kappa), asm.load(self.f), symmetric=self.symmetric_bc)
        return linear_solve(A, b), 1


@dataclass(frozen=True)
class Richards:
    f: float = 1.0
    alpha: float = 1e5
    tol: float = 1e-8              # on max |u_{n+1} - u_n|
    max_iter: int = 25
    symmetric_bc: bool = True

    name = "richards"

    def solve(self, asm: P1Assembler, kappa: np.ndarray, linear_solve: LinearSolve) -> tuple[np.ndarray, int]:
        load = asm.load(self.f)
        u = np.zeros(asm.n_vertices)
        for it in range(1, self.max_iter + 1):
            # k(u) is evaluated at cell centroids: the 1-point rule FEniCS picks for this form
            coef = kappa / (1.0 + self.alpha * u[asm.cells].mean(axis=1))
            A, b = asm.apply_dirichlet(asm.stiffness(coef), load, symmetric=self.symmetric_bc)
            u_new = linear_solve(A, b)
            converged = np.abs(u_new - u).max() < self.tol
            u = u_new
            if converged:
                break
        return u, it


PROBLEMS = {"diffusion": Diffusion, "richards": Richards}
