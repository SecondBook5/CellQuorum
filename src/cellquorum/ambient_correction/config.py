"""Configuration for the ambient-RNA correction stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class AmbientCorrectionConfig(StrictBaseModel):
    """SoupX ambient-RNA correction settings (opt-in; from-raw-data step)."""

    # Whether ambient correction runs. Off by default: only from-CellRanger runs
    # need it; the default h5ad-in pipeline starts downstream of it.
    enabled: bool = False

    # Ambient-correction method registry key (soupx now; decontx/cellbender later).
    method: str = "soupx"

    # Root under which manifest cellranger_path entries resolve. When None, the
    # manifest paths are treated as absolute or run-cwd-relative.
    cellranger_root: str | None = None

    # Subdirectory (under the run objects dir) for corrected per-library matrices.
    output_dir: str = "soupx"

    # Round corrected counts to integers (valid counts layer downstream).
    round_to_int: bool = True

    # Leiden/quick-cluster resolution SoupX autoEstCont uses.
    cluster_resolution: float = 0.5

    # Per-library R timeout (SoupX can take minutes on large libraries).
    timeout_seconds: int = 1800

    # Reuse a library's already-corrected matrix (+ its rho sidecar) instead of
    # re-running SoupX when a complete output exists. Makes a re-run after a
    # later-stage failure skip the multi-minute-per-library SoupX step.
    resume: bool = True


__all__ = ["AmbientCorrectionConfig"]
