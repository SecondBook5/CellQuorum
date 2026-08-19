"""Base class for R-backed analysis methods.

Owns the Rscript-availability, backend-resolution, and R-package guards that
every R method (edgeR, Milo, propeller, NicheNet, MultiNicheNet, hdWGCNA,
scDiagnostics) otherwise copy-pastes. Subclasses set ``r_package`` and call
``_resolve_rscript_backend`` at the top of ``_run``.
"""

from __future__ import annotations

from cellquorum.methods.base import AnalysisMethod, MethodSkip


class RAnalysisMethod(AnalysisMethod):
    # Execution backend is always the Rscript backend.
    backend = "rscript"

    # R package the concrete method needs (e.g. "edgeR"); set by subclasses.
    r_package: str

    def _resolve_rscript_backend(
        self, context: object, config: dict | None = None
    ) -> tuple[object | None, MethodSkip | None]:
        """Return (rscript_backend, None) when R is runnable, else (None, skip).

        Availability is delegated to the resolved backend's own
        ``_rscript_available`` primitive, which checks the backend's *configured*
        ``rscript_path`` (threaded from ``r.rscript_path``) — the same primitive
        ``run_script`` gates on, so the check and the execution agree. Gating on
        a hardcoded bare ``Rscript`` on PATH instead would ignore that
        configuration and wrongly skip whenever R lives at a non-default path —
        the exact failure mode of the layered container image, where Rscript is
        provisioned outside the default PATH.
        """
        # Resolve package name with config override support.
        pkg = (config or {}).get("r_package", self.r_package)

        # Resolve the Rscript backend from the run's registry.
        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("rscript")
            except Exception:
                backend = None
        if backend is None:
            return None, self._skip("rscript backend unavailable")

        # Availability honors the backend's configured rscript_path.
        if not backend._rscript_available():
            return None, self._skip("Rscript unavailable")
        if not backend._r_package_available(pkg):
            return None, self._skip(f"{pkg} R package unavailable", r_package=pkg)
        return backend, None


__all__ = ["RAnalysisMethod"]
