"""Download and checksum the human NicheNet and cisTarget priors used by the SCC recipe."""

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

ZENODO = "https://zenodo.org/records/7074291/files/"
AERTS = "https://resources.aertslab.org/cistarget/"
RANKING = "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
FILES = {
    "nichenet/ligand_target_matrix_nsga2r_final.rds": (
        ZENODO + "ligand_target_matrix_nsga2r_final.rds",
        "699fce17ff65e2511277359696306bb130aa833bd7f30dd9290ad9efd9dc9c5d",
    ),
    "nichenet/lr_network_human_21122021.rds": (
        ZENODO + "lr_network_human_21122021.rds",
        "47c971d2fbba4ecd0ba7485d1846a74054432a0d1a97ea3e6a79ae27d0da8094",
    ),
    "nichenet/weighted_networks_nsga2r_final.rds": (
        ZENODO + "weighted_networks_nsga2r_final.rds",
        "3de79d664b09a5cc89b569fa9d8b0e8ad650f07f968d0d41f9c1cb7a00032e3a",
    ),
    "scenic/allTFs_hg38.txt": (
        AERTS + "tf_lists/allTFs_hg38.txt",
        "3953034f84112c60d3d8ef15b0e0c8ac5fce0b40d2c7c0824c2945c70cee2523",
    ),
    "scenic/motifs-v10nr_clust-nr.hgnc.tbl": (
        AERTS + "motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl",
        "81eb754118e27e854974301b1400fcf519489f8be5249239671fb288cb501c31",
    ),
    "scenic/" + RANKING: (
        AERTS + "databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/" + RANKING,
        "50bd3a2e0da17bf74ad31b1a2f284ebbacfad700a07a540a7a1d3db5e185eaac",
    ),
}


def verify(path: Path, expected: str) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path} has SHA256 {actual}. Expected {expected}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for name, (url, digest) in FILES.items():
        dest = args.directory / name
        if dest.is_file():
            verify(dest, digest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=dest.parent) as directory:
                tmp = Path(directory) / dest.name
                with urlopen(url, timeout=120) as src, tmp.open("wb") as out:
                    shutil.copyfileobj(src, out)
                verify(tmp, digest)
                tmp.replace(dest)
        print(f"{name} verified")


if __name__ == "__main__":
    main()
