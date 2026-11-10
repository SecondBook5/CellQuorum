"""Import 10x expression matrices and library metadata for the guided workflow."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csr_matrix


def _first(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise ValueError(f"{directory} needs one of {', '.join(names)}.")


def _read_counts(source: Path) -> ad.AnnData:
    if source.is_dir() and (source / "outs").is_dir():
        source = source / "outs"
    if source.is_dir() and any(source.glob("filtered_feature_bc_matrix*")):
        source = _first(source, ("filtered_feature_bc_matrix.h5", "filtered_feature_bc_matrix"))
    if source.suffix.lower() == ".h5ad":
        data = ad.read_h5ad(source)
        if "gene_ids" not in data.var:
            data.var["gene_ids"] = data.var_names.astype(str)
    elif source.suffix == ".h5":
        data = sc.read_10x_h5(source, gex_only=True)
    else:
        directory = source if source.is_dir() else source.parent
        matrix = _first(directory, ("matrix.mtx.gz", "matrix.mtx")) if source.is_dir() else source
        if matrix.name not in {"matrix.mtx", "matrix.mtx.gz"}:
            raise ValueError(
                f"Expected a 10x H5, Matrix Market directory or Cell Ranger outs: {source}"
            )
        features = _first(
            directory, ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv")
        )
        genes = pd.read_csv(features, sep="\t", header=None, dtype=str, keep_default_na=False)
        barcodes = pd.read_csv(
            _first(directory, ("barcodes.tsv.gz", "barcodes.tsv")),
            sep="\t",
            header=None,
            dtype=str,
            keep_default_na=False,
        )
        data = ad.AnnData(
            csr_matrix(mmread(matrix).T),
            obs=pd.DataFrame(index=barcodes[0].tolist()),
            var=pd.DataFrame({"gene_ids": genes[0].tolist()}, index=genes[1].tolist()),
        )
        if 2 in genes:
            data = data[:, genes[2].eq("Gene Expression").to_numpy()].copy()
    if not data.obs_names.is_unique or not data.var["gene_ids"].is_unique or not data.n_vars:
        raise ValueError(
            f"Empty gene expression matrix or duplicate feature IDs/barcodes in {source}"
        )
    if "gene_symbol" not in data.var:
        data.var["gene_symbol"] = data.var_names.astype(str)
    data.var_names = data.var["gene_ids"].astype(str).tolist()
    return data


def import_libraries(source: Path, destination: Path) -> pd.DataFrame:
    """Import counts and preserve original library and barcode identifiers."""
    source = source.resolve()
    if source.suffix.lower() in {".csv", ".tsv"}:
        sheet = pd.read_csv(
            source,
            sep="\t" if source.suffix.lower() == ".tsv" else ",",
            dtype=str,
            keep_default_na=False,
        )
    else:
        sheet = pd.DataFrame([{"sample_id": "library_1", "path": str(source)}])
    if "sample_id" not in sheet or sheet.empty or sheet["sample_id"].duplicated().any():
        raise ValueError("Provide one row per unique sample_id in the sample sheet.")
    parts: list[ad.AnnData] = []
    records = []
    for row in sheet.to_dict("records"):
        sample = row["sample_id"]
        locator = row.get("path") or row.get("cellranger_path")
        if not sample.strip() or not locator:
            raise ValueError("Each library needs a sample_id and path or cellranger_path.")
        path = (source.parent / locator).resolve()
        data = _read_counts(path)
        record = {"sample_id": sample, "path": str(path)}
        if path.is_dir():
            outs = path / "outs" if (path / "outs").is_dir() else path
            if (outs / "filtered_feature_bc_matrix.h5").exists() or (
                outs / "filtered_feature_bc_matrix"
            ).exists():
                record["cellranger_path"] = str(outs.parent if outs.name == "outs" else outs)
                record["filtered_matrix_path"] = str(
                    _first(outs, ("filtered_feature_bc_matrix.h5", "filtered_feature_bc_matrix"))
                )
                for key, name in (
                    ("raw_matrix_path", "raw_feature_bc_matrix.h5"),
                    ("bam_path", "possorted_genome_bam.bam"),
                ):
                    if (outs / name).is_file():
                        record[key] = str(outs / name)
                if (outs / "raw_feature_bc_matrix").is_dir() and "raw_matrix_path" not in record:
                    record["raw_matrix_path"] = str(outs / "raw_feature_bc_matrix")
        if row.get("raw_matrix_path") and "filtered_matrix_path" not in record:
            record["filtered_matrix_path"] = str(
                path.parent if path.name.startswith("matrix.mtx") else path
            )
        original_barcodes = data.obs_names.copy()
        record["counts_source"] = row.get("counts_layer") or "X"
        if "cellranger_path" in record:
            parent = Path(record["cellranger_path"])
            looms = sorted((parent / "velocyto").glob("*.loom"))
            if len(looms) == 1:
                record["loom_path"] = str(looms[0])
        if row.get("counts_layer"):
            if row["counts_layer"] not in data.layers:
                raise ValueError(
                    f"Missing original counts layer {row['counts_layer']} for {sample}."
                )
            if "cellquorum_input_X" in data.layers:
                raise ValueError(f"Reserved layer cellquorum_input_X already exists in {sample}.")
            data.layers["cellquorum_input_X"] = data.X.copy()
            data.X = data.layers[row["counts_layer"]].copy()
        values = data.X.data if hasattr(data.X, "tocsr") else np.asarray(data.X)
        if not (np.isfinite(values) & (values >= 0) & (values == np.floor(values))).all():
            raise ValueError(
                f"{sample} needs nonnegative integer raw counts in X or a counts_layer."
            )
        if row.get("metadata_path"):
            metadata = pd.read_csv(
                source.parent / row["metadata_path"], dtype={"barcode": str}, keep_default_na=False
            ).set_index("barcode")
            if not metadata.index.is_unique or set(metadata.index) != set(data.obs_names):
                raise ValueError(f"Cell metadata barcodes must match {sample} exactly.")
            aligned = metadata.loc[data.obs_names]
            for key in aligned:
                if key in data.obs and not data.obs[key].astype(str).equals(
                    aligned[key].astype(str)
                ):
                    raise ValueError(f"Conflicting cell metadata {key} for {sample}")
                data.obs[key] = aligned[key]
        if "barcode" not in data.obs:
            data.obs["barcode"] = original_barcodes
        for heading, value in row.items():
            key = str(heading)
            if key.endswith("_path") and key != "metadata_path" and value:
                record[key] = str((source.parent / value).resolve())
                if not Path(record[key]).exists():
                    raise ValueError(f"Missing {key} for {sample}: {record[key]}")
            elif (
                key not in {"path", "metadata_path", "counts_layer"}
                and not key.endswith("_path")
                and value
            ):
                if key in data.obs and not data.obs[key].astype(str).eq(value).all():
                    raise ValueError(f"Conflicting {key} in cell and library metadata for {sample}")
                data.obs[key] = value
                record[key] = value
        data.obs_names = [f"{sample}:{barcode}" for barcode in data.obs_names]
        if parts:
            if set(data.layers) != set(parts[0].layers):
                raise ValueError(
                    f"Library {sample} has different expression "
                    "layers. Supply comparable libraries."
                )
            reference = parts[0].var
            if set(data.var_names) != set(reference.index):
                raise ValueError(f"Library {sample} uses a different feature reference.")
            data = data[:, reference.index].copy()
            if not reference["gene_symbol"].equals(data.var["gene_symbol"]):
                raise ValueError(f"Gene identifiers and symbols disagree in {sample}")
        records.append(record)
        parts.append(data)
    merged = ad.concat(parts, merge="first", uns_merge="same", join="outer")
    if not merged.obs_names.is_unique:
        raise ValueError("Library and barcode combinations must give unique cell identifiers.")
    merged.var_names = merged.var["gene_symbol"].tolist()
    merged.var_names_make_unique()
    merged.uns["cellquorum_input_format"] = "library_gene_expression_counts"
    merged.uns["cellquorum_count_sources"] = {r["sample_id"]: r["counts_source"] for r in records}
    merged.var.index.name = None
    merged.write_h5ad(destination, compression="gzip")
    return pd.DataFrame(records)
