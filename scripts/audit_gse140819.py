"""Audit the local GEO extraction without loading expression matrices."""

import argparse
import csv
import gzip
import hashlib
import re
import tarfile
from pathlib import Path

import h5py


def barcode_sequence(identifier: str) -> str:
    match = re.search(r"[_-]([ACGT]+)$", identifier)
    if match is None:
        raise ValueError(f"Unrecognized metadata barcode: {identifier}")
    return match[1]


def audit(root: Path) -> list[dict]:
    records = []
    with tarfile.open(root / "GSE140819_RAW.tar") as archive:
        members = archive.getnames()
        for path in sorted((root / "metadata").glob("*.csv")):
            sample = path.stem
            candidates = [
                name
                for name in members
                if re.fullmatch(rf"GSM\d+_metadata_{re.escape(sample)}_channel\d+\.csv\.gz", name)
            ]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous archive metadata for {sample}")
            source = candidates[0]
            accession, rest = source.split("_metadata_", 1)
            channel = re.search(r"_channel(\d+)\.csv", rest)[1]
            matrix_members = [
                name
                for name in members
                if name.startswith(f"{accession}_{sample}_channel{channel}_raw_")
                and name.endswith(".h5.gz")
            ]
            if len(matrix_members) != 1:
                raise ValueError(f"Ambiguous archive matrix for {sample}")
            with archive.extractfile(source) as stream:
                archived_metadata = gzip.decompress(stream.read())
            if archived_metadata != path.read_bytes():
                raise ValueError(f"Metadata differs from archive: {sample}")
            with path.open() as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                fields = reader.fieldnames
            ids = [row[""] for row in rows]
            sequences = [barcode_sequence(identifier) for identifier in ids]
            if len(set(sequences)) != len(sequences):
                raise ValueError(f"Ambiguous metadata sequences: {sample}")
            if any(
                match[1] != channel
                for key in ids
                if (match := re.search(r"channel(\d+)[_-]", key)) is not None
            ):
                raise ValueError(f"Metadata channel differs from archive: {sample}")
            with h5py.File(root / "raw" / f"{sample}.h5") as handle:
                group = handle["matrix"] if "matrix" in handle else handle[next(iter(handle))]
                raw = group["barcodes"].asstr()[:]
                shape = group["shape"][:]
            matrix_sequences = [barcode.rsplit("-", 1)[0] for barcode in raw]
            if len(set(matrix_sequences)) != len(matrix_sequences):
                raise ValueError(f"Ambiguous matrix sequences: {sample}")
            missing = set(sequences) - set(matrix_sequences)
            if missing:
                raise ValueError(f"{sample}: {len(missing)} metadata sequences absent")
            record = dict(
                sample=sample,
                accession=accession,
                source_channel=channel,
                archive_matrix=matrix_members[0],
                archive_metadata=source,
                genes=int(shape[0]),
                raw_barcodes=int(shape[1]),
                metadata_barcodes=len(rows),
                unmatched_sequences=len(missing),
                annotation_categories=len({row["annotate"] for row in rows}),
                metadata_sha256=hashlib.sha256(archived_metadata).hexdigest(),
            )
            for column in ("doublet", "emptydrop"):
                values = [row[column] for row in rows] if column in fields else None
                if values is not None and set(values) - {"TRUE", "FALSE"}:
                    raise ValueError(f"Unexpected {column} values in {sample}")
                record[f"reference_{column}"] = values.count("TRUE") if values is not None else ""
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/gse140819"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = audit(args.data)
    if not records:
        raise ValueError("No metadata found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    print(f"Validated {len(records)} preparations; wrote {args.output}")


if __name__ == "__main__":
    main()
