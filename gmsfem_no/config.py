"""Paths and hyper-parameters (one place, overridable from the command line)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
# Raw inputs (mesh_250/mesh.xml, KLE_250/<id>.xml) are not part of the repository.
RAW = Path(os.environ.get("GMSFEM_RAW_DIR", PROJECT / "raw"))


@dataclass
class Paths:
    mesh_xml: Path = RAW / "mesh_250" / "mesh.xml"
    kle_dir: Path = RAW / "KLE_250"
    data_dir: Path = PROJECT / "data"
    checkpoints: Path = PROJECT / "checkpoints"
    results: Path = PROJECT / "results"
    logs: Path = PROJECT / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def bases(self, kind: str) -> Path:
        return self.data_dir / f"bases_{kind}.npy"


@dataclass
class Problem:
    n_coarse: int = 10            # coarse cells per side (fine grid is 250 x 250)
    n_basis: int = 8              # multiscale basis functions per coarse node
    train_kle: range = range(1, 794)
    test_kle: range = range(794, 992)


# FFNO configuration of the reference model: 5 layers, width 128, per-kind Fourier modes.
MODES = {"full": (18, 18), "half": (14, 8), "corner": (10, 10)}


@dataclass
class ModelConfig:
    n_layers: int = 5
    width: int = 128
    modes: tuple = (18, 18)
    seed: int = 22


@dataclass
class TrainConfig:
    epochs: int = 600
    batch_size: int = 32
    lr: float = 1e-3
    lr_final_fraction: float = 0.1        # cosine decay down to lr * this
    weight_decay: float = 1e-4
    eval_every: int = 10
    seed: int = 0
    extra: dict = field(default_factory=dict)
