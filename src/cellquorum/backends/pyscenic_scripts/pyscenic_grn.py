#!/usr/bin/env python
"""CellQuorum in-env pySCENIC GRN backend (grn->ctx pipeline).

Runs the pySCENIC grn->ctx protocol in an isolated frozen conda environment:
exports raw counts to loom, runs `pyscenic grn` (GRNBoost2) -> `pyscenic ctx`
(cisTarget motif pruning) -> regulons. This is CellQuorum's generic in-env
backend; study-specific subsetting removed.

Two failure regimes, deliberately distinct:
  - NOT CONFIGURED (missing loom/HDF5 deps, cisTarget DBs, motifs, or TFs): writes empty output +
    a grn_SKIPPED_{tag}.txt and exits 0 -> a harmless skip, nothing else affected.
  - CLI ERROR (once configured, `pyscenic grn`/`ctx` itself exits non-zero -- numpy/dask clash,
    TF-gene mismatch, etc.): tees the full pyscenic output to a PERSISTENT scenic_log_{tag}.txt on
    the data area, writes grn_FAILED_{tag}.txt, and exits NON-ZERO so Snakemake marks the job
    failed. keep-going still isolates it from the other GRN networks.

Resources:
  --tfs      allTFs_hg38.txt
  --rankings hg38-*.genes_vs_motifs.rankings.feather
  --motifs   motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _write_empty(out_dir: Path, tag: str, reason: str) -> None:
    """Write the expected empty outputs for a harmless configuration skip.

    This preserves the existing isolation contract: missing optional SCENIC
    resources do not affect the rest of the workflow.

    Args:
        out_dir: Directory in which SCENIC outputs are written.
        tag: Tag or dataset name used in output filenames.
        reason: Human-readable explanation for the skip.
    """

    # Create the output directory when it does not already exist.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write an empty regulon table with the expected schema.
    (out_dir / f"scenic_regulons_{tag}.csv").write_text(
        "TF,MotifID,AUC,NES,TargetGenes\n",
        encoding="utf-8",
    )

    # Write an empty adjacency table with the expected schema.
    (out_dir / f"scenic_adjacencies_{tag}.tsv").write_text(
        "TF\ttarget\timportance\n",
        encoding="utf-8",
    )

    # Write a persistent marker explaining why the backend was skipped.
    (out_dir / f"grn_SKIPPED_{tag}.txt").write_text(
        f"pySCENIC skipped (isolated backend, no downstream effect): {reason}\n",
        encoding="utf-8",
    )

    # Report the harmless skip in the job log.
    print(f"[scenic] SKIPPED gracefully: {reason}")


def _decode_attr(value: Any) -> str:
    """Normalize an HDF5 attribute value to a Python string.

    Args:
        value: HDF5 attribute value, potentially stored as bytes.

    Returns:
        The decoded string representation.
    """

    # Return an empty string for a missing attribute.
    if value is None:
        return ""

    # Decode byte strings explicitly.
    if isinstance(value, bytes):
        return value.decode("utf-8")

    # Convert all other scalar values to strings.
    return str(value)


def _decode_strings(values: Any) -> Any:
    """Decode a one-dimensional HDF5 string array.

    Args:
        values: Array-like object containing bytes or strings.

    Returns:
        A one-dimensional NumPy object array containing Python strings.
    """

    # Import NumPy locally after the dependency gate in main.
    import numpy as np

    # Convert the input to a flat NumPy array.
    array = np.asarray(values).reshape(-1)

    # Decode each element while preserving ordinary strings.
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in array],
        dtype=object,
    )


def _read_h5_vector(node: Any) -> Any:
    """Read the H5AD vector encodings needed by this workflow.

    The pySCENIC environment's older AnnData cannot decode the newer
    ``nullable-string-array`` encoding used by the typed H5AD files. This
    helper reads only the required vectors directly with h5py.

    Args:
        node: h5py Dataset or Group containing one H5AD vector.

    Returns:
        A one-dimensional NumPy array.

    Raises:
        ValueError: If the encoding is unsupported or malformed.
    """

    # Import h5py and NumPy locally after dependency validation.
    import h5py
    import numpy as np

    # Read either spelling used for the encoding metadata.
    encoding_type = _decode_attr(
        node.attrs.get("encoding-type", node.attrs.get("encoding_type", ""))
    )

    # Read ordinary datasets directly.
    if isinstance(node, h5py.Dataset):
        # Load the dataset values.
        values = np.asarray(node[()])

        # Decode string-like datasets into normal Python strings.
        if values.dtype.kind in {"O", "S", "U"} or encoding_type == "string-array":
            return _decode_strings(values)

        # Flatten numeric or Boolean vectors.
        return values.reshape(-1)

    # Read nullable arrays from their values and mask children.
    if encoding_type.startswith("nullable-") or ("values" in node and "mask" in node):
        # Read the underlying values recursively.
        values = _read_h5_vector(node["values"])

        # Read the missing-value mask.
        mask = np.asarray(node["mask"][()], dtype=bool).reshape(-1)

        # Reject malformed nullable vectors.
        if values.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Malformed nullable vector {node.name!r}: "
                f"{values.shape[0]} values but {mask.shape[0]} mask entries."
            )

        # Promote to object dtype so missing values can be represented.
        result = np.asarray(values, dtype=object)

        # Mark missing entries explicitly.
        result[mask] = None

        # Return the decoded nullable vector.
        return result

    # Read categorical vectors from categories and integer codes.
    if encoding_type == "categorical" or ("categories" in node and "codes" in node):
        # Read the category labels.
        categories = _read_h5_vector(node["categories"])

        # Read the integer category codes.
        codes = np.asarray(node["codes"][()], dtype=int).reshape(-1)

        # Allocate the decoded result.
        result = np.empty(codes.shape[0], dtype=object)

        # Decode each categorical value.
        for index, code in enumerate(codes):
            # Treat negative codes as missing.
            if code < 0:
                result[index] = None

            # Resolve valid category codes.
            elif code < categories.shape[0]:
                result[index] = categories[code]

            # Reject out-of-range codes.
            else:
                raise ValueError(
                    f"Malformed categorical vector {node.name!r}: "
                    f"code {code} exceeds {categories.shape[0]} categories."
                )

        # Return the decoded categorical vector.
        return result

    # Reject unknown group encodings rather than silently misreading data.
    raise ValueError(
        f"Unsupported H5AD vector encoding at {node.name!r}: "
        f"encoding_type={encoding_type!r}, children={list(node.keys())!r}."
    )


def _read_h5_matrix(node: Any) -> Any:
    """Read a dense, CSR, or CSC matrix directly from an H5AD file.

    Args:
        node: h5py Dataset or Group containing the matrix.

    Returns:
        A NumPy dense array or SciPy sparse matrix.

    Raises:
        ValueError: If the matrix encoding is unsupported or malformed.
    """

    # Import h5py, NumPy, and SciPy locally after dependency validation.
    import h5py
    import numpy as np
    import scipy.sparse as sp

    # Read either spelling used for the encoding metadata.
    encoding_type = _decode_attr(
        node.attrs.get("encoding-type", node.attrs.get("encoding_type", ""))
    )

    # Read dense matrices directly.
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[()])

    # Require the sparse matrix shape metadata.
    raw_shape = node.attrs.get("shape")

    # Reject malformed sparse matrices without a shape.
    if raw_shape is None:
        raise ValueError(f"Sparse matrix {node.name!r} is missing its shape.")

    # Convert the HDF5 shape to a Python tuple.
    shape = tuple(int(value) for value in np.asarray(raw_shape).reshape(-1))

    # Read the sparse matrix components.
    data = np.asarray(node["data"][()])
    indices = np.asarray(node["indices"][()], dtype=int)
    indptr = np.asarray(node["indptr"][()], dtype=int)

    # Construct a CSR matrix.
    if encoding_type == "csr_matrix":
        return sp.csr_matrix((data, indices, indptr), shape=shape)

    # Construct a CSC matrix.
    if encoding_type == "csc_matrix":
        return sp.csc_matrix((data, indices, indptr), shape=shape)

    # Reject unsupported matrix encodings.
    raise ValueError(
        f"Unsupported H5AD matrix encoding at {node.name!r}: " f"encoding_type={encoding_type!r}."
    )


def _read_axis_names(handle: Any, axis_name: str) -> Any:
    """Read the observation or variable names directly from an H5AD file.

    Args:
        handle: Open h5py file handle.
        axis_name: Either ``"obs"`` or ``"var"``.

    Returns:
        A one-dimensional NumPy string array.

    Raises:
        KeyError: If the axis group is missing.
        ValueError: If its index cannot be read.
    """

    # Import NumPy locally after dependency validation.
    import numpy as np

    # Require the requested axis group.
    if axis_name not in handle:
        raise KeyError(f"H5AD file is missing /{axis_name}.")

    # Resolve the H5AD dataframe group.
    group = handle[axis_name]

    # Read the configured index key.
    index_key = _decode_attr(group.attrs.get("_index", "_index"))

    # Require the index node.
    if index_key not in group:
        raise ValueError(f"/{axis_name} declares index {index_key!r}, but that child is absent.")

    # Decode the index vector directly.
    raw_names = _read_h5_vector(group[index_key])

    # Reject missing names.
    if any(value is None for value in raw_names):
        raise ValueError(f"/{axis_name}/{index_key} contains missing names.")

    # Convert names to strings.
    return np.asarray([str(value) for value in raw_names], dtype=object)


def _assert_raw_counts(matrix: Any, source: str) -> None:
    """Verify that a matrix contains non-negative integer raw counts.

    Args:
        matrix: Dense or sparse expression matrix.
        source: Human-readable matrix source used in errors.

    Raises:
        ValueError: If values are non-finite, negative, or non-integer.
    """

    # Import NumPy and SciPy locally after dependency validation.
    import numpy as np
    import scipy.sparse as sp

    # Inspect only stored values for sparse matrices.
    values = np.asarray(matrix.data) if sp.issparse(matrix) else np.asarray(matrix).reshape(-1)

    # Accept an empty matrix structurally.
    if values.size == 0:
        return

    # Reject non-finite values.
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains NaN or infinite values.")

    # Reject negative values.
    if np.any(values < 0):
        raise ValueError(f"{source} contains negative values and is not raw counts.")

    # Reject non-integer values.
    if not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-6):
        raise ValueError(
            f"{source} contains non-integer values. pySCENIC requires raw counts, "
            "not log-normalized expression."
        )


def main() -> None:
    """Export raw counts to loom and run the pySCENIC CLI."""

    # Create the argument parser.
    p = argparse.ArgumentParser(
        description="pySCENIC GRN inference (protocol CLI; backend, opt-in)"
    )

    # Add the H5AD input argument.
    p.add_argument("--h5ad", required=True)

    # Add the transcription-factor list argument.
    p.add_argument("--tfs", default="")

    # Add the motif annotation argument.
    p.add_argument("--motifs", default="")

    # Add the cisTarget ranking database argument.
    p.add_argument(
        "--rankings",
        default="",
        help="feather ranking DB(s): file, space-joined list, or glob",
    )

    # Add the output directory argument.
    p.add_argument("--out-dir", required=True)

    # Add the tag argument.
    p.add_argument("--tag", required=True)

    # Add the worker count argument.
    p.add_argument("--num-workers", type=int, default=16)

    # Add the maximum cell count argument.
    p.add_argument("--max-cells", type=int, default=20000)

    # Add the layer argument.
    p.add_argument("--layer", default="counts")

    # Add the seed argument.
    p.add_argument("--seed", type=int, default=0)

    # Parse command-line arguments.
    args = p.parse_args()

    # Resolve the output directory.
    out_dir = Path(args.out_dir)

    # 1. Graceful bail-outs -----------------------------------------------------

    # Import only the dependencies required by the direct HDF5 exporter.
    try:
        # Import HDF5 access.
        import h5py

        # Import loom writing support.
        import loompy as lp

        # Import NumPy.
        import numpy as np

        # Import sparse matrix support.
        import scipy.sparse as sp

    # Preserve the existing harmless-skip behavior for missing dependencies.
    except Exception as e:
        _write_empty(
            out_dir,
            args.tag,
            f"loom/HDF5 import failed: {type(e).__name__}: {e}",
        )
        sys.exit(0)

    # Expand the ranking database glob once into a real list.
    ranking_files = sorted(glob.glob(args.rankings))

    # Accept a literal existing file when the argument is not a glob.
    if not ranking_files and args.rankings and os.path.exists(args.rankings):
        ranking_files = [args.rankings]

    # Skip harmlessly when no ranking database is configured.
    if not ranking_files:
        _write_empty(
            out_dir,
            args.tag,
            f"ranking DB missing ({args.rankings!r}) — download hg38 cisTarget DBs",
        )
        sys.exit(0)

    # Validate the motif and transcription-factor resources.
    for name, path in [("motifs", args.motifs), ("tfs", args.tfs)]:
        # Skip harmlessly when either resource is absent.
        if not path or not os.path.exists(path):
            _write_empty(
                out_dir,
                args.tag,
                f"cisTarget resource '{name}' missing ({path!r})",
            )
            sys.exit(0)

    # 2. Export raw counts to loom ---------------------------------------------

    # Create the output directory.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open the H5AD directly to bypass AnnData's incompatible nullable-string reader.
    with h5py.File(args.h5ad, "r") as f:
        # Prefer the configured count layer.
        if "layers" in f and args.layer in f["layers"]:
            # Read raw counts from the specified layer.
            matrix = _read_h5_matrix(f["layers"][args.layer])

            # Record the selected matrix source.
            matrix_source = f"layers/{args.layer}"

        # Fall back to X only when the count layer is absent.
        elif "X" in f:
            # Read the fallback matrix.
            matrix = _read_h5_matrix(f["X"])

            # Record the fallback matrix source.
            matrix_source = "X"

        # Reject H5AD files without an expression matrix.
        else:
            raise KeyError(f"H5AD contains neither /layers/{args.layer} nor /X.")

        # Read cell identifiers directly from /obs.
        cell_ids = _read_axis_names(f, "obs")

        # Read gene identifiers directly from /var.
        gene_ids = _read_axis_names(f, "var")

    # Validate matrix dimensions.
    if matrix.shape != (cell_ids.shape[0], gene_ids.shape[0]):
        raise ValueError(
            f"H5AD dimension mismatch: matrix={matrix.shape}, "
            f"obs={cell_ids.shape[0]}, var={gene_ids.shape[0]}."
        )

    # Verify that the selected matrix is genuinely raw counts.
    _assert_raw_counts(matrix, matrix_source)

    # Report the exact matrix source.
    print(f"[scenic] matrix source: {matrix_source} " f"shape={matrix.shape} dtype={matrix.dtype}")

    # Downsample before converting the sparse count matrix to dense.
    if matrix.shape[0] > args.max_cells:
        # Draw a deterministic random sample.
        idx = np.random.RandomState(args.seed).choice(
            matrix.shape[0],
            args.max_cells,
            replace=False,
        )

        # Sort sampled indices to preserve stable source order.
        idx.sort()

        # Subset the expression matrix.
        matrix = matrix[idx, :]

        # Subset the cell identifiers.
        cell_ids = cell_ids[idx]

        # Report the deterministic downsampling.
        print(f"[scenic] downsampled to {args.max_cells} cells")

    # Convert only the selected matrix to a dense array for loompy.
    X = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)

    # Define the loom output path.
    loom_path = out_dir / f"scenic_input_{args.tag}.loom"

    # Remove a stale partial loom before writing.
    if loom_path.exists():
        loom_path.unlink()

    # Write genes by cells, as required by loom and pySCENIC.
    lp.create(
        str(loom_path),
        X.T,
        {"Gene": np.asarray(gene_ids, dtype=str)},
        {"CellID": np.asarray(cell_ids, dtype=str)},
    )

    # Report the completed loom export.
    print(f"[scenic] wrote loom {loom_path} " f"({X.shape[0]} cells x {X.shape[1]} genes)")

    # Define the adjacency output path.
    adj = out_dir / f"scenic_adjacencies_{args.tag}.tsv"

    # Define the regulon output path.
    reg = out_dir / f"scenic_regulons_{args.tag}.csv"

    # 3. pySCENIC CLI: grn -> ctx ----------------------------------------------

    # Define the persistent pySCENIC log path.
    log_path = out_dir / f"scenic_log_{args.tag}.txt"

    def _run_logged(cmd: list[str], step: str) -> None:
        """Run one pySCENIC command and tee output to a persistent log.

        Args:
            cmd: Executable and command-line arguments.
            step: Human-readable step name.

        Raises:
            subprocess.CalledProcessError: If the command exits non-zero.
            RuntimeError: If the subprocess output stream cannot be opened.
        """

        # Open the persistent log in append mode.
        with open(log_path, "a", encoding="utf-8") as lg:
            # Write the command header.
            lg.write(f"\n===== [scenic:{args.tag}] " f"{step}: {' '.join(cmd)} =====\n")

            # Flush the command header before execution.
            lg.flush()

            # Start the command with stderr redirected into stdout.
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Reject an unexpected missing output stream.
            if proc.stdout is None:
                proc.kill()
                raise RuntimeError(f"Could not capture output for {step!r}.")

            # Stream output to both SLURM and the persistent log.
            for line in proc.stdout:
                # Write to the live job log.
                sys.stdout.write(line)

                # Flush the live job log immediately.
                sys.stdout.flush()

                # Write to the persistent log.
                lg.write(line)

                # Flush the persistent log immediately.
                lg.flush()

            # Wait for the command to finish.
            rc = proc.wait()

        # Convert a non-zero exit into a real workflow failure.
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)

    # Run the configured pySCENIC stages.
    try:
        # Run GRNBoost2 co-expression inference.
        _run_logged(
            [
                "pyscenic",
                "grn",
                str(loom_path),
                args.tfs,
                "-o",
                str(adj),
                "--num_workers",
                str(args.num_workers),
            ],
            "grn (GRNBoost2)",
        )

        # Run cisTarget motif pruning with every matched ranking database.
        _run_logged(
            [
                "pyscenic",
                "ctx",
                str(adj),
                *ranking_files,
                "--annotations_fname",
                args.motifs,
                "--expression_mtx_fname",
                str(loom_path),
                "--output",
                str(reg),
                "--mask_dropouts",
                "--num_workers",
                str(args.num_workers),
            ],
            "ctx (cisTarget)",
        )

        # Report the completed regulon output.
        print(f"[scenic] regulons -> {reg}")

    # Preserve any real pySCENIC failure.
    except subprocess.CalledProcessError as e:
        # Write a persistent failure marker.
        (out_dir / f"grn_FAILED_{args.tag}.txt").write_text(
            f"pyscenic CLI failed (exit {e.returncode}): {' '.join(e.cmd)}\n"
            f"full output -> {log_path}\n",
            encoding="utf-8",
        )

        # Report the failure to standard error.
        print(
            f"[scenic] FAILED: pyscenic exited {e.returncode}; see {log_path}",
            file=sys.stderr,
        )

        # Exit non-zero so Snakemake marks the rule as failed.
        sys.exit(e.returncode or 1)


if __name__ == "__main__":
    main()
