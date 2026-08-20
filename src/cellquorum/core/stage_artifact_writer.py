"""Stage-facing artifact writer for CellQuorum pipeline stages.

Pipeline stages historically hand-rolled the same artifact-writing skeleton:
compute a path under ``context.paths.results``, ``mkdir`` its parent, call
``DataFrame.to_csv`` (or ``json.dumps``), then construct a ``StageArtifact`` and
append it to the stage result. ``StageArtifactWriter`` centralizes that skeleton
so a stage declares *what* it produced, not *how* to write it.

The writer is a thin, namespace-aware facade over ``ArtifactManager`` (which owns
the tested format-dispatch, directory-creation, and registration logic). It adds
the ergonomics stages actually need: address an output by run namespace
(``results``, ``reports``, ``objects``, ...) plus an optional subdirectory and a
filename, instead of hand-computing an absolute path. Because every namespace
directory is ``root / "<name>"`` (see ``PipelinePaths.from_output_dir``), the
absolute path the writer produces is byte-identical to the hand-computed path it
replaces, so migrating a stage to the writer changes no output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cellquorum.core.artifacts import ArtifactManager
from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.core.stage import StageArtifact

# The run namespaces a stage may address. Each maps to a PipelinePaths attribute
# that resolves to ``root / "<name>"``. ``scratch`` is intentionally excluded:
# scratch holds R-subprocess inputs, which are never artifacts.
_ALLOWED_NAMESPACES = ("results", "figures", "reports", "objects", "provenance", "logs")


class StageArtifactWriter:
    """Write stage artifacts by namespace + filename, returning ``StageArtifact``.

    A stage builds one writer per run via :meth:`from_context`, then calls
    :meth:`table` / :meth:`json` / :meth:`register`. Each call resolves
    ``<namespace>/<subdir>/<filename>`` under the run root, writes through the
    shared :class:`ArtifactManager`, and returns the :class:`StageArtifact` the
    stage appends to its ``StageResult.artifacts``.

    Args:
        paths: The run's standardized directory layout.
        manager: The ``ArtifactManager`` rooted at ``paths.root`` that performs
            the actual write + registration.
        default_namespace: Namespace used when a call omits ``namespace``.
        default_subdir: Subdirectory used when a call omits ``subdir``.
    """

    def __init__(
        self,
        paths: PipelinePaths,
        manager: ArtifactManager,
        *,
        default_namespace: str = "results",
        default_subdir: str | None = None,
    ) -> None:
        # Store the run directory layout for namespace resolution.
        self._paths = paths

        # Store the ArtifactManager that owns the tested write/register logic.
        self._manager = manager

        # Store the default namespace applied when a call omits one.
        self._default_namespace = default_namespace

        # Store the default subdirectory applied when a call omits one.
        self._default_subdir = default_subdir

    @classmethod
    def from_context(
        cls,
        context: PipelineContext,
        *,
        default_namespace: str = "results",
        default_subdir: str | None = None,
    ) -> StageArtifactWriter:
        """Build a writer for a run from its :class:`PipelineContext`.

        Args:
            context: The active pipeline context; its ``paths.root`` roots the
                underlying ArtifactManager.
            default_namespace: Namespace used when a call omits ``namespace``.
            default_subdir: Subdirectory used when a call omits ``subdir``.

        Returns:
            A writer bound to the run's directory layout.
        """

        # Root the ArtifactManager at the run root so relative paths mirror the
        # on-disk namespace layout and produced absolute paths are unchanged.
        manager = ArtifactManager.from_root(context.paths.root)

        # Return a writer bound to this run's paths and manager.
        return cls(
            context.paths,
            manager,
            default_namespace=default_namespace,
            default_subdir=default_subdir,
        )

    def _relative_path(self, filename: str, namespace: str | None, subdir: str | None) -> Path:
        """Resolve ``<namespace>/<subdir>/<filename>`` as a root-relative path.

        Args:
            filename: Output filename (with suffix).
            namespace: Run namespace, or ``None`` to use the default.
            subdir: Optional subdirectory under the namespace, or ``None`` to use
                the default.

        Returns:
            The target path relative to the run root, for ArtifactManager.

        Raises:
            ValueError: If ``namespace`` is not an allowed run namespace.
        """

        # Resolve the effective namespace and validate it.
        effective_namespace = namespace or self._default_namespace
        if effective_namespace not in _ALLOWED_NAMESPACES:
            raise ValueError(
                "Unknown artifact namespace "
                f"{effective_namespace!r}. Allowed: {_ALLOWED_NAMESPACES}."
            )

        # Resolve the effective subdirectory (default when the call omits one).
        effective_subdir = subdir if subdir is not None else self._default_subdir

        # Build the absolute target under the namespace directory.
        base: Path = getattr(self._paths, effective_namespace)
        target = base / effective_subdir / filename if effective_subdir else base / filename

        # Express the target relative to the run root for the ArtifactManager.
        return target.relative_to(self._paths.root)

    def table(
        self,
        dataframe: pd.DataFrame,
        filename: str,
        *,
        name: str,
        description: str,
        index: bool = False,
        namespace: str | None = None,
        subdir: str | None = None,
    ) -> StageArtifact:
        """Write a table artifact (CSV or Parquet, inferred from ``filename``).

        Args:
            dataframe: The table to write.
            filename: Output filename ending in ``.csv`` or ``.parquet``.
            name: Stable artifact name.
            description: Human-readable artifact description.
            index: Whether to write the DataFrame index (default ``False``,
                matching the common hand-rolled ``to_csv(index=False)``).
            namespace: Run namespace, or ``None`` for the default.
            subdir: Subdirectory under the namespace, or ``None`` for the default.

        Returns:
            The written table's :class:`StageArtifact`.
        """

        # Resolve the target and delegate to the tested table writer.
        relative_path = self._relative_path(filename, namespace, subdir)
        return self._manager.write_dataframe(
            dataframe,
            name=name,
            relative_path=relative_path,
            description=description,
            index=index,
        )

    def json(
        self,
        payload: dict[str, Any] | list[Any],
        filename: str,
        *,
        name: str,
        description: str,
        namespace: str | None = None,
        subdir: str | None = None,
    ) -> StageArtifact:
        """Write a JSON artifact (``indent=2, sort_keys=True``).

        Args:
            payload: JSON-serializable dict or list.
            filename: Output filename ending in ``.json``.
            name: Stable artifact name.
            description: Human-readable artifact description.
            namespace: Run namespace, or ``None`` for the default.
            subdir: Subdirectory under the namespace, or ``None`` for the default.

        Returns:
            The written JSON file's :class:`StageArtifact`.
        """

        # Resolve the target and delegate to the tested JSON writer.
        relative_path = self._relative_path(filename, namespace, subdir)
        return self._manager.write_json(
            payload,
            name=name,
            relative_path=relative_path,
            description=description,
        )

    def register(
        self,
        *,
        name: str,
        filename: str,
        kind: str,
        description: str,
        namespace: str | None = None,
        subdir: str | None = None,
    ) -> StageArtifact:
        """Register an artifact a domain library already wrote to disk.

        Use this when a stage writes an object through a specialized library
        (e.g. ``adata.write_h5ad`` for an ``.h5ad``) but still needs to expose it
        to reporting and provenance.

        Args:
            name: Stable artifact name.
            filename: The already-written filename (with suffix).
            kind: Artifact kind (e.g. ``h5ad``, ``directory``).
            description: Human-readable artifact description.
            namespace: Run namespace, or ``None`` for the default.
            subdir: Subdirectory under the namespace, or ``None`` for the default.

        Returns:
            The registered :class:`StageArtifact`.
        """

        # Resolve the target and register the pre-written artifact.
        relative_path = self._relative_path(filename, namespace, subdir)
        return self._manager.register(
            name=name,
            relative_path=relative_path,
            kind=kind,
            description=description,
        )
