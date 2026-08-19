"""Base class for R-backed analysis methods.

Owns the Rscript-availability, backend-resolution, and R-package guards that
every R method (edgeR, Milo, propeller, NicheNet, MultiNicheNet, hdWGCNA,
scDiagnostics) otherwise copy-pastes. Subclasses set ``r_package`` and call
``_resolve_rscript_backend`` at the top of ``_run``.
"""

from __future__ import annotations

import shutil

from cellquorum.methods.base import AnalysisMethod, MethodSkip


class RAnalysisMethod(AnalysisMethod):
    # Execution backend is always the Rscript backend.
    backend = "rscript"

    # R package the concrete method needs (e.g. "edgeR"); set by subclasses.
    r_package: str

    def _resolve_rscript_backend(
        self, context: object, config: dict | None = None
    ) -> tuple[object | None, MethodSkip | None]:
        """Return (rscript_backend, None) when R is runnable, else (None, skip)."""
        # Resolve package name with config override support
        pkg = (config or {}).get("r_package", self.r_package)

        if shutil.which("Rscript") is None:
            return None, self._skip("Rscript unavailable")
        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("rscript")
            except Exception:
                backend = None
        if backend is None:
            return None, self._skip("rscript backend unavailable")
        if not backend._r_package_available(pkg):
            return None, self._skip(f"{pkg} R package unavailable", r_package=pkg)
        return backend, None


__all__ = ["RAnalysisMethod"]
