"""Runtime progress and output reporting for CellQuorum runs."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

if TYPE_CHECKING:
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.core.stage import StageExecutionRecord


class RunReporter:
    """
    Report CellQuorum run progress and results to the console.

    The reporter produces the startup banner, resolved-config echo, per-stage
    progress, and final run summary. This is what makes CellQuorum announce
    itself as a polished, reusable, config-driven tool at runtime.

    When verbose=False, every method is a no-op producing zero output. This is
    load-bearing: quiet runs must be truly silent and backward-compatible.

    Args:
        verbose: Whether to produce output. When False, all methods are no-ops.
        level: Output detail level (quiet, normal, verbose).
        console: Optional Rich Console for testing/customization.
    """

    def __init__(
        self,
        verbose: bool = True,
        level: str = "normal",
        console: Console | None = None,
    ) -> None:
        """Initialize the RunReporter."""
        # Store the verbosity flag.
        self._verbose = verbose

        # Store the output detail level.
        self._level = level

        # Store the Rich console (or create a default).
        self._console = console or Console()

        # Track active progress bar for routing output through tqdm.write.
        self._active_bar: tqdm | None = None

    def banner(self, version: str, project_name: str, run_id: str) -> None:
        """
        Print the CellQuorum startup banner.

        Args:
            version: CellQuorum version.
            project_name: Project name from config.
            run_id: Run identifier.
        """
        # Early return when not verbose or quiet.
        if not self._verbose or self._level == "quiet":
            return

        # Build banner text.
        banner_text = f"CellQuorum v{version} · {project_name} · run {run_id}"

        # Render banner as a Rich panel.
        self._console.print(Panel(banner_text, expand=False))

    def config_echo(
        self, config: CellQuorumConfig, planned_stage_names: list[str] | None = None
    ) -> None:
        """
        Echo the resolved configuration to the console.

        This is the "reusable tool" thesis made visible. Show compute backend,
        seed, and enabled stages with their key parameters. Generic — works for
        any config (KC, mast, lung). Truncates list fields to counts.

        Args:
            config: Validated CellQuorum configuration.
            planned_stage_names: Optional list of stage names that will actually
                run (planned and registered). When provided, shows only these
                stages in plan order. When None, shows all enabled stages from
                config (fallback for tests/direct use).
        """
        # Early return when not verbose or quiet.
        if not self._verbose or self._level == "quiet":
            return

        # Create a table for configuration display.
        table = Table(title="Configuration", show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value")

        # Add compute backend and seed.
        table.add_row("Backend", config.compute.backend)
        table.add_row("Random seed", str(config.run.random_seed))

        # Determine which stages to show.
        if planned_stage_names is not None:
            # Use the provided planned stage names (already filtered to
            # enabled+registered, in plan order).
            enabled_stages = planned_stage_names
        else:
            # Fall back to all enabled stages from config.
            enabled_stages = []
            stage_config = config.stages.model_dump()
            # Iterate through StageSelectionConfig fields in order.
            from cellquorum.config.models import StageSelectionConfig

            for field_name in StageSelectionConfig.model_fields.keys():
                if stage_config.get(field_name, False):
                    enabled_stages.append(field_name)

        # Show enabled stages.
        if enabled_stages:
            table.add_row("Enabled stages", ", ".join(enabled_stages))
        else:
            table.add_row("Enabled stages", "none")

        # Show stage-specific parameters for enabled stages.
        config_dict = config.model_dump()
        for stage_name in enabled_stages:
            # Check if this stage has a sub-config in the main config.
            if stage_name in config_dict:
                stage_cfg = config_dict[stage_name]
                if isinstance(stage_cfg, dict) and stage_cfg:
                    # Build compact param line.
                    params = []
                    # Add method if present.
                    if "method" in stage_cfg and stage_cfg["method"]:
                        params.append(f"method={stage_cfg['method']}")
                    # Add up to 4 informative scalar fields.
                    count = 0
                    for key, value in stage_cfg.items():
                        if key == "method" or key == "enabled":
                            continue
                        if count >= 4:
                            break
                        # Truncate list fields to counts.
                        if isinstance(value, list):
                            if len(value) == 0:
                                params.append(f"{key}=[]")
                            else:
                                params.append(f"{key}=[{len(value)} items]")
                            count += 1
                        elif value is not None and value != "" and not isinstance(value, dict):
                            # Show scalar/simple values.
                            params.append(f"{key}={value}")
                            count += 1
                    if params:
                        table.add_row(f"  {stage_name}", ", ".join(params))

        # Print the table.
        self._console.print(table)

    def stage_start(self, name: str, index: int, total: int) -> None:
        """
        Announce the start of a stage.

        Args:
            name: Stage name.
            index: Stage index (1-based).
            total: Total number of stages.
        """
        # Early return when not verbose or quiet.
        if not self._verbose or self._level == "quiet":
            return

        # Build stage start message.
        msg = f"▶ {name} [{index}/{total}]"

        # Route through tqdm.write when a progress bar is active (prevents
        # terminal line corruption). tqdm.write uses the console's file handle.
        if self._active_bar is not None:
            tqdm.write(msg, file=self._console.file)
        else:
            self._console.print(msg)

    def stage_end(self, record: StageExecutionRecord) -> None:
        """
        Report the completion of a stage.

        Args:
            record: Stage execution record with status, timing, and metadata.
        """
        # Early return when not verbose or quiet.
        if not self._verbose or self._level == "quiet":
            return

        # Build status message. When a progress bar is active, route through
        # tqdm.write (prevents terminal corruption). tqdm.write respects the
        # console's file handle so it works with captured buffers in tests.
        if record.status == "success":
            msg = f"✓ {record.stage_name} ({record.duration_seconds:.1f}s)"
            if self._active_bar is not None:
                tqdm.write(msg, file=self._console.file)
            else:
                self._console.print(msg, style="green")
        elif record.status == "skipped":
            reason = record.skip_reason.reason if record.skip_reason else "unknown"
            msg = f"⊘ {record.stage_name} skipped: {reason}"
            if self._active_bar is not None:
                tqdm.write(msg, file=self._console.file)
            else:
                self._console.print(msg, style="yellow")
        elif record.status == "failed":
            error_msg = record.error.message if record.error else "unknown error"
            msg = f"✗ {record.stage_name} failed: {error_msg}"
            if self._active_bar is not None:
                tqdm.write(msg, file=self._console.file)
            else:
                self._console.print(msg, style="red")

        # Print warnings (always). Route through tqdm.write when active.
        for warning in record.warnings:
            warning_msg = f"  ⚠ {warning}"
            if self._active_bar is not None:
                tqdm.write(warning_msg, file=self._console.file)
            else:
                self._console.print(warning_msg, style="yellow")

        # Print notes (only when level is verbose). Route through tqdm.write.
        if self._level == "verbose":
            for note in record.notes:
                note_msg = f"  ℹ {note}"
                if self._active_bar is not None:
                    tqdm.write(note_msg, file=self._console.file)
                else:
                    self._console.print(note_msg, style="dim")

    def run_summary(
        self,
        records: list[StageExecutionRecord],
        run_root: str,
        total_seconds: float,
    ) -> None:
        """
        Print the final run summary.

        Args:
            records: All stage execution records.
            run_root: Root output directory.
            total_seconds: Total run duration in seconds.
        """
        # Early return when not verbose.
        if not self._verbose:
            return

        # Create summary table.
        table = Table(title="Run Summary")
        table.add_column("Stage", style="cyan")
        table.add_column("Status")
        table.add_column("Duration", justify="right")

        # Count statuses.
        success_count = 0
        skipped_count = 0
        failed_count = 0

        # Add each stage to the table.
        for record in records:
            status_style = "green" if record.status == "success" else "yellow"
            if record.status == "failed":
                status_style = "red"

            if record.status == "success":
                success_count += 1
            elif record.status == "skipped":
                skipped_count += 1
            elif record.status == "failed":
                failed_count += 1

            table.add_row(
                record.stage_name,
                record.status,
                f"{record.duration_seconds:.1f}s",
                style=status_style,
            )

        # Print the table.
        self._console.print(table)

        # Print summary stats.
        self._console.print(
            f"\n{success_count} succeeded, {skipped_count} skipped, " f"{failed_count} failed"
        )
        self._console.print(f"Total time: {total_seconds:.1f}s")
        self._console.print(f"Outputs: {run_root}")

    # Define a simple progress handle.
    class _ProgressHandle:
        def __init__(self, bar: tqdm | None) -> None:
            self._bar = bar

        def advance(self, n: int = 1) -> None:
            if self._bar is not None:
                self._bar.update(n)

    @contextmanager
    def progress(self, total: int) -> Generator[_ProgressHandle, None, None]:
        """
        Create a progress bar context manager.

        Yields a handle with an advance() method. When verbose=False or
        level="quiet", the handle's advance() is a no-op and no bar shows.

        Args:
            total: Total number of items to track.

        Yields:
            Progress handle with advance() method.
        """
        # Create progress bar when verbose and not quiet.
        if self._verbose and self._level != "quiet":
            bar = tqdm(total=total, leave=False)
            self._active_bar = bar
            try:
                yield self._ProgressHandle(bar)
            finally:
                self._active_bar = None
                bar.close()
        else:
            # No-op handle when not verbose or quiet.
            yield self._ProgressHandle(None)
