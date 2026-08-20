"""Backward-compatible re-export shim — see ``cellquorum.ambient_correction.soupx``."""

from __future__ import annotations

from cellquorum.ambient_correction.soupx import (
    SoupXError,
    corrected_output_exists,
    import_corrected_matrix,
    parse_rho,
    read_rho_sidecar,
    run_soupx_library,
    write_rho_sidecar,
)

__all__ = [
    "SoupXError",
    "corrected_output_exists",
    "import_corrected_matrix",
    "parse_rho",
    "read_rho_sidecar",
    "run_soupx_library",
    "write_rho_sidecar",
]
