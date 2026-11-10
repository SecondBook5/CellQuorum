"""Train on control K562 cells and hold out GATA1 perturbations and independent control cells."""

import argparse
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.dataset.open("rb") as handle:
        if hashlib.file_digest(handle, "md5").hexdigest() != "d8cba17576d1a8afc0f7d71b79cad0f7":
            raise ValueError("Unexpected checksum for scPerturb Replogle K562 Essential.")
    d = ad.read_h5ad(args.dataset, backed="r")
    rng = np.random.default_rng(42)
    ids = np.arange(d.n_obs)
    pert = d.obs.perturbation.to_numpy()
    train = []
    held = []
    for batch in sorted(d.obs.batch.unique()):
        controls = ids[(pert == "control") & (d.obs.batch == batch)]
        treated = ids[(pert == "GATA1") & (d.obs.batch == batch)]
        if not len(treated) or len(controls) < 2:
            continue
        controls = rng.permutation(controls)
        reserve = min(max(10, len(treated) * 2), len(controls) // 2)
        held.extend([*treated, *controls[:reserve]])
        train.extend(controls[reserve : reserve + 300])
    if not train or not held:
        raise ValueError(
            "Need control and measured GATA1 cells from matching experimental batches."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, indices in (("training_controls", train), ("held_out", held)):
        cells = d[sorted(indices)].to_memory()
        x = csr_matrix(cells.X)
        if not (np.isfinite(x.data) & (x.data >= 0) & (x.data == np.floor(x.data))).all():
            raise ValueError("Expected original integer count data.")
        clean = ad.AnnData(
            x, obs=cells.obs[["batch", "perturbation", "guide_id"]].copy(), var=cells.var.copy()
        )
        if name == "training_controls":
            clean.layers["counts"] = x.copy()
        clean.write_h5ad(args.output / f"{name}.h5ad", compression="gzip")
        print(
            f"{name}: {clean.n_obs} cells",
            {str(k): int(v) for k, v in clean.obs.perturbation.value_counts().items()},
        )
    d.file.close()


if __name__ == "__main__":
    main()
