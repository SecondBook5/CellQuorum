"""Regression coverage for published metadata identifier variants."""

import runpy
from pathlib import Path

import pytest

barcode_sequence = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "audit_gse140819.py")
)["barcode_sequence"]


@pytest.mark.parametrize(
    "identifier",
    [
        "CY112_nuclei_channel2-AAACCCAAGAAACTAC",
        "HTAPP-951-SMP-4652TST-V2channel1_AAACCTGAGACCTTTG",
        "14-AAACCTGAGCCCAGCT",
        "NSC005bLT_FT-AAACCTGAGACCGGAT",
    ],
)
def test_published_barcode_formats(identifier):
    assert barcode_sequence(identifier) == identifier[-16:]


@pytest.mark.parametrize("identifier", ["", "sample", "sample-ACGT-1", "sample-ACNT"])
def test_unrecognized_barcode_rejected(identifier):
    with pytest.raises(ValueError, match="Unrecognized"):
        barcode_sequence(identifier)
