"""Train the F-FNO for one patch kind (full / half / corner).

    python scripts/train.py --kind full --gpu 0
    python scripts/train.py --kind half --gpu 1 --epochs 600

Needs data/bases_*.npy (scripts/make_dataset.py). Saves checkpoints/ffno_<kind>.eqx with
the κ normalisation and loss history in its header; TensorBoard logs go to logs/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmsfem_no.config import MODES, ModelConfig, Paths, Problem, TrainConfig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["full", "half", "corner"], required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    p.add_argument("--width", type=int, default=ModelConfig.width)
    p.add_argument("--layers", type=int, default=ModelConfig.n_layers)
    p.add_argument("--modes", type=int, nargs=2, default=None, help="Fourier modes per axis (default per kind)")
    p.add_argument("--train-kle", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"))
    p.add_argument("--out", default=None, help="checkpoint path (default checkpoints/ffno_<kind>.eqx)")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import numpy as np
    from tensorboardX import SummaryWriter

    from gmsfem_no import model as M
    from gmsfem_no.data import kappa_store, load_mesh, load_split
    from gmsfem_no.patches import CoarseGrid
    from gmsfem_no.train import fit

    paths, prob = Paths(), Problem()
    mcfg = ModelConfig(n_layers=args.layers, width=args.width, modes=tuple(args.modes or MODES[args.kind]))
    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, eval_every=args.eval_every)
    train_ids = np.arange(args.train_kle[0], args.train_kle[1] + 1) if args.train_kle else np.array(prob.train_kle)
    out = Path(args.out) if args.out else paths.checkpoints / f"ffno_{args.kind}.eqx"

    mesh = load_mesh(paths)
    cg = CoarseGrid(mesh, prob.n_coarse)
    kappa = kappa_store(paths, mesh)
    train = load_split(paths, cg, kappa, args.kind, train_ids)
    test = load_split(paths, cg, kappa, args.kind, np.array(prob.test_kle))

    log_file = paths.logs / f"{out.stem}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a") as fh:
            fh.write(msg + "\n")

    log(f"== {args.kind}: {mcfg} {tcfg}")
    writer = SummaryWriter(str(paths.logs / out.stem))
    model, meta = fit(train, test, mcfg, tcfg, log=log, writer=writer)
    writer.close()
    meta["kind"] = args.kind
    M.save(out, model, meta)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
