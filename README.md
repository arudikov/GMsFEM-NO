# GMsFEM-NO

Code for the paper **Locally Subspace-Informed Neural Operators for Efficient Multiscale PDE Solving**
(ICLR 2026): [OpenReview](https://openreview.net/forum?id=RqXxkiCYip) · [arXiv:2505.16030](https://arxiv.org/abs/2505.16030).

GMsFEM with multiscale basis functions predicted by a neural operator (F-FNO).
Problems are posed in Ω = (0,1)², `u = 0` on ∂Ω, f = const:

* **diffusion:** `-div(κ∇u) = f`;
* **richards:** `-div(κ k(u) ∇u) = f`, `k(u) = 1/(1 + 10⁵u)`, Picard iterations starting from u = 0
  (tolerance 1e-8 in max|uₙ₊₁ − uₙ|, at most 25 iterations).

Fine mesh 250×250 (P1), coarse mesh 10×10, 8 basis functions per coarse node. The coefficients κ are 991 KLE realizations: 1–793 for training, 794–991 for testing.

## Layout

```
gmsfem_no/
  config.py    paths and hyperparameters (a single source instead of scattered argparse defaults)
  fem.py       mesh and KLE reading from XML, vectorized P1 assembly, BCs, reference solution, errors
  patches.py   coarse grid, full/half/corner patches, canonical orientation, operator inputs
  offline.py   offline GMsFEM: multiscale partition of unity + local spectral problem (scipy)
  data.py      KLE cache (memmap), training samples
  model.py     F-FNO (one class for isotropic and anisotropic modes), checkpoints
  loss.py      subspace loss  k − ||Q_tᵀ Q_p||²_F  (Cholesky-QR, targets orthonormalized in advance)
  train.py     training loop: a whole epoch is one jit-compiled scan, all data on the GPU
  problems.py  Diffusion / Richards problems: one algorithm for both the reference and the multiscale solution
  online.py    online GMsFEM: R, A_H = R A Rᵀ at every linear step of the problem, L2/H1 errors
scripts/
  make_dataset.py   generates targets (bases) for all KLEs, in parallel, resumable
  train.py          trains one model (--kind full|half|corner)
  evaluate.py       online stage (--problem diffusion|richards, --f) on test KLEs → results/*.csv
checkpoints/        ffno_{full,half,corner}.eqx — model weights (not included in the repository)
run_all.sh          full pipeline including retraining
```

FEniCS is not required: assembly, BCs, error integrals and local spectral problems
are implemented with numpy/scipy and verified against dolfin to machine precision.

## Installation

Tested with Python 3.13.5:

```bash
pip install -r requirements.txt     # jax 0.4.34 (CUDA 12), equinox, optax, numpy, scipy, tensorboardX
```

## Data

The input data are not included in the repository. A directory with the following layout is required:

```
<raw>/
  mesh_250/mesh.xml     250×250 mesh in DOLFIN XML format
  KLE_250/<id>.xml      coefficients κ (DG0, DOLFIN XML), id = 1…991
```

By default it is looked up at `raw/` in the project root (a symlink works); otherwise set the
`GMSFEM_RAW_DIR` environment variable. On the first run the XML files are read once and cached
in `data/cache/`. Everything generated (`data/`, `checkpoints/`, `logs/`) is excluded from git.

## Usage

```bash
export GMSFEM_RAW_DIR=/path/to/data_250   # or: ln -s /path/to/data_250 raw
PY=python
$PY scripts/make_dataset.py --workers 30           # ~3 min on 30 cores, data/bases_*.npy (8.3 GB)
$PY scripts/train.py --kind full   --gpu 0         # checkpoints/ffno_full.eqx
$PY scripts/train.py --kind half   --gpu 1
$PY scripts/train.py --kind corner --gpu 2
$PY scripts/evaluate.py --gpu 0                       # diffusion, f = 1 → results/ffno_diffusion_f1.csv
$PY scripts/evaluate.py --gpu 0 --problem richards    # Richards, f = 1 → results/ffno_richards_f1.csv
```

## Results: one model, f(x) = 1

Model: a triple of F-FNOs (full / half / corner; 5 layers, width 128, modes 18 / (14,8) / 10),
trained on KLE 1–793. Test set: 198 KLEs (794–991).
The basis depends only on κ, so the same model serves both problems.

| problem | L2, % (mean / median / max) | H1, % (mean / median / max) |
|---|---|---|
| diffusion (`results/ffno_diffusion_f1.csv`) | 1.173 / 1.048 / 7.09 | 14.00 / 11.88 / 47.4 |
| richards (`results/ffno_richards_f1.csv`) | 1.870 / 1.679 / 8.57 | 14.36 / 11.12 / 60.1 |

Picard iterations for Richards converge on all KLEs: the multiscale solver takes 11.3 iterations on average (max 16–17),
the fine-grid solver 11.1 (max 14).

Time for 198 test KLEs: inference ~25 s (including JIT compilation). Online stage on 24 processes:
diffusion ~15 s, about 0.6 s per KLE, of which 0.35 s is the reference fine-grid solve.
Richards ~85 s: per Picard iteration the coarse step costs ~0.25 s (of which R·A·Rᵀ ≈ 0.2 s),
the fine step ~0.35 s.

**Boundary conditions.** Diffusion eliminates boundary rows (the diagonal
and columns are kept), while Richards uses DOLFIN's symmetric elimination `assemble_system`:
rows and columns are zeroed, and the diagonal holds the number of triangles at the node. The fine-grid
solution does not depend on this, but the multiscale one does, because the predicted basis functions
are nonzero on ∂Ω. On 20 KLEs switching between the variants changes L2 by 0.04–0.12 percentage points (`symmetric_bc` in `problems.py`).

## Verification

* Assembly of A, M, b matches dolfin (rel. 3e-16); the fine-grid solution matches a precomputed reference solution.
* Richards: the coefficient κ/(1+10⁵u) at the triangle centroid matches FEniCS quadrature (3e-16),
  boundary conditions match `assemble_system`, the Picard solution matches a precomputed reference
  solution (rel. 5e-8, the float32 limit of the stored file).
* The basis (PoU × eigenfunctions) matches dolfin/SLEPc: subspace distance 1e-15,
  eigenvalues 1e-10.

## Training

A full-model step (batch 32, patch 51×51) takes ~27 ms on an A100: almost all of the time goes to FFTs in the F-FNO,
the loss is nearly free. A full epoch ≈ 2000 steps, 600 epochs ≈ 10–14 h on a shared GPU,
half ≈ 4–5 h, corner ≈ 0.5 h (the three models can be trained in parallel on three GPUs).

## Citation

```bibtex
@inproceedings{rudikov2026locally,
  title     = {Locally Subspace-Informed Neural Operators for Efficient Multiscale {PDE} Solving},
  author    = {Rudikov, Alexander and Fanaskov, Vladimir and Stepanov, Sergei and Shan, Buzheng and
               Muravleva, Ekaterina and Efendiev, Yalchin and Oseledets, Ivan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=RqXxkiCYip}
}
```
