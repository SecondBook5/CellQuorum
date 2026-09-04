"""Propeller differential-abundance method (R speckle)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.backends.script_paths import r_script_path
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod
from cellquorum.stages.comparative.differential_abundance.aggregation import (
    aggregate_celltype_counts,
)
from cellquorum.stats.claim_support import annotate_fdr_reachability

# Path to the bundled propeller script.
_PROPELLER_R = r_script_path("propeller.R")


class PropellerMethod(RAnalysisMethod):
    """Speckle propeller moderated-t proportion test for differential abundance.

    Propeller tests for cell-type proportion differences between conditions using
    transformed proportions and a moderated t-statistic (spec §DA). Aggregates cells
    to sample × cell-type counts, transforms proportions (asin or logit), and fits
    a linear model to detect abundance changes.

    **The design is blocked on donor whenever the cohort allows it.** ``pair_by_donor``
    (or, under its ``auto`` default, ``paired`` from the project ``design`` block) asks
    for the block, and this method then fits the donor factor as a fixed effect so the
    contrast is the arm difference *within* donor. Fitting arms only on a matched cohort
    leaves the between-donor baseline in the residual, which is the same specification
    error that turned real per-gene effects into nulls elsewhere in this project; on a
    nine-donor matched cohort here it moved the whole table (nothing under FDR 0.39
    arms-only, two lineages at 0.031 within donor). Whether the block is estimable is
    decided *here* rather than in R, because this is where the donor/condition table is:
    R re-checks the rank and fails loudly rather than quietly fitting something the
    caller did not ask for.

    The estimability threshold is two donors spanning both arms, which is lower than
    scCODA's :data:`MIN_PAIRED_BLOCKS` floor in the same stage, and deliberately so.
    That floor is the point below which an assumption-free signed-rank test cannot reach
    0.05 at all; this is a moderated t whose variance is borrowed across cell types, so
    the binding constraint is whether the block can be fitted, not whether a
    randomization test could have called it. On a small matched cohort the block matters
    *more*, not less.

    When pairing is requested but the cohort cannot support it, the method fits arms
    only and says so — in ``metrics``, in ``notes``, and in the ``paired`` column of
    the table itself. A silent fallback would leave a reader unable to tell an
    underpowered null from a misspecified one, which is the failure being fixed.

    The table also carries the design's own FDR floor (see
    :func:`cellquorum.stats.claim_support.annotate_fdr_reachability`), so a
    non-significant FDR is readable as either a negative result or a family that could
    not have produced a significant row at any effect size. The per-sample resolution
    table (``da_group_resolution.csv``) is a property of the count matrix rather than
    of any one method and is written by ``proportion_ttest``.
    """

    name = "propeller"
    stage_category = "differential_abundance"
    r_package = "speckle"

    def input_contract(self, config: dict) -> DataContract:
        """Require the design obs columns (no layer needed for DA)."""
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")
        return DataContract(
            required_obs=[condition_col, donor_col, cell_type_col],
        )

    def requires_obs(self, config: dict) -> list[str]:
        """Return the design obs columns that must exist for DA to run."""

        # Read the design columns from config.
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")

        # Require all design columns to exist.
        return [condition_col, donor_col, cell_type_col]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        """Aggregate cell-type counts, fit propeller, and return the DA table."""

        # Resolve config fields (all schema-driven; no hardcoded study assumptions).
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")
        case = config.get("case")
        control = config.get("control")
        transform = config.get("transform", "asin")
        paired_requested = _pairing_requested(config)
        timeout = int(config.get("timeout_seconds", 1800))

        # A comparison needs both case and control labels.
        if not case or not control:
            return self._skip("case/control labels not set in config")

        # Rscript + backend + package guards (hoisted to RAnalysisMethod).
        backend, skip = self._resolve_rscript_backend(context)
        if skip is not None:
            return skip

        # Aggregate to sample × cell-type counts.
        cc = aggregate_celltype_counts(
            adata,
            donor_col=donor_col,
            condition_col=condition_col,
            cell_type_col=cell_type_col,
        )

        # Decide the design on the samples R will actually fit: the two arms only.
        arm_meta = cc.sample_meta[cc.sample_meta[condition_col].isin([case, control])]
        paired, fallback_reason, n_donors_blocked = _resolve_pairing(
            arm_meta,
            donor_col=donor_col,
            condition_col=condition_col,
            case=case,
            requested=paired_requested,
        )

        # Write aggregated inputs to scratch.
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        counts_csv = scratch / "da_counts.csv"
        meta_csv = scratch / "da_meta.csv"

        # counts.csv: first col 'sample', remaining cols = cell types, integer counts.
        cc.counts.reset_index(names="sample").to_csv(counts_csv, index=False)

        # meta.csv: first col = sample id (row index), must contain condition_col column.
        # cc.sample_meta already has columns named by condition_col and donor_col.
        cc.sample_meta.to_csv(meta_csv, index=True)

        # Prepare the output path in the run results directory.
        results_dir = Path(context.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_csv = results_dir / "da_propeller.csv"

        # Invoke the propeller script; non-zero exit -> recorded skip (never crash).
        # propeller.R CLI: <counts.csv> <meta.csv> <out.csv> <condition_col> <case>
        # <control> <transform> <donor_col> <paired>
        args = [
            str(counts_csv),
            str(meta_csv),
            str(out_csv),
            condition_col,
            case,
            control,
            transform,
            donor_col,
            "TRUE" if paired else "FALSE",
        ]
        try:
            proc = backend.run_script(_PROPELLER_R, args, timeout=timeout)
        except FileNotFoundError as exc:
            return self._skip("R execution failed", error=str(exc)[:500])
        except subprocess.TimeoutExpired as exc:
            # A configured timeout must skip this method, not crash the stage
            # and abort the sibling methods still queued after it.
            return self._skip(f"R execution timed out after {timeout}s", error=str(exc)[:500])
        if proc.returncode != 0:
            return self._skip("propeller script failed", stderr=proc.stderr.strip()[:500])

        # Annotate the R table with the floor its own design imposes, so the FDR
        # column is read alongside the best FDR this cohort could have reached.
        floor_metrics = _annotate_floor_in_place(
            out_csv,
            arm_meta,
            donor_col=donor_col,
            condition_col=condition_col,
            case=case,
            paired=paired,
            alpha=float(config.get("fdr", 0.05)),
        )

        design_note = (
            f"donor-blocked (~ 0 + arm + {donor_col}), {n_donors_blocked} donors"
            if paired
            else "arms only (~ 0 + arm)"
        )
        notes = [
            f"Propeller DA: {case} vs {control}, transform={transform}, design={design_note}.",
            *cc.notes,
        ]
        if fallback_reason:
            notes.append(
                "Propeller: paired design requested but fitted arms only — "
                f"{fallback_reason}. A null from this fit is not a within-donor null."
            )

        # Return the DA table as an artifact plus provenance metrics.
        return StageResult(
            adata=adata,
            artifacts=[
                StageArtifact(
                    name="da_results",
                    path=out_csv,
                    kind="csv",
                    description=(
                        f"Propeller DA ({case} vs {control}), {transform} transform, {design_note}."
                    ),
                )
            ],
            notes=notes,
            metrics={
                "case": case,
                "control": control,
                "transform": transform,
                "paired": paired,
                "paired_requested": paired_requested,
                "paired_fallback_reason": fallback_reason,
                "n_donors_blocked": n_donors_blocked,
                "n_samples": int(cc.counts.shape[0]),
                "n_celltypes": int(cc.counts.shape[1]),
                "n_unlabeled_cells": cc.n_unlabeled,
                **floor_metrics,
            },
            backend="rscript",
        )


def _pairing_requested(config: dict) -> bool:
    """Read whether a donor block was asked for, from the one key that governs it.

    Two keys reach this method and they can contradict each other: ``paired`` is
    bridged from the project ``design`` block and describes the cohort, while
    ``pair_by_donor`` is the DA stage's own three-way switch. ``pair_by_donor`` wins
    where it is explicit, because it is the more specific instruction; ``auto`` — its
    default — defers to the declared design rather than inferring pairing from the
    data, so a cohort explicitly declared unpaired is not silently blocked on donor.

    The estimability question is separate and lives in :func:`_resolve_pairing`: this
    function only reports what was asked for.
    """
    pair_mode = str(config.get("pair_by_donor", "auto")).lower()
    if pair_mode == "never":
        return False
    if pair_mode == "always":
        return True
    return bool(config.get("paired", True))


def _resolve_pairing(
    arm_meta: pd.DataFrame,
    *,
    donor_col: str,
    condition_col: str,
    case: str,
    requested: bool,
) -> tuple[bool, str, int]:
    """Decide whether the donor block can be fitted, and say why when it cannot.

    Two donors spanning both arms is the exact threshold rather than a chosen minimum,
    and the derivation is worth stating because it is not the obvious one. The design
    has ``n_samples`` rows and ``2 + (n_donors - 1)`` columns — two arm means plus donor
    treatment contrasts — so its residual degrees of freedom are
    ``n_samples - n_donors - 1``. A donor present in one arm only contributes exactly
    one row *and* one column, so it leaves that quantity untouched; a donor spanning
    both contributes two rows and one column. The residual df is therefore
    ``n_spanning - 1`` however many single-arm donors are present, and a cohort with two
    spanning donors always has at least one df and is always full rank.

    The rank is still computed. Not because a cohort that passes the threshold can fail
    it, but because the design that gets *fitted* is built by R's ``model.matrix`` from
    the same table, and if that ever diverges from the design modelled here the failure
    should be a recorded fallback rather than a fit whose coefficients come back ``NA``.

    Args:
        arm_meta: Sample metadata restricted to the two contrasted arms.
        donor_col: Column holding the donor id.
        condition_col: Column holding the condition label.
        case: The condition label treated as the case arm.
        requested: Whether the project design declared the cohort paired.

    Returns:
        ``(paired, fallback_reason, n_donors_blocked)``. ``fallback_reason`` is empty
        when nothing was given up: either pairing was not requested, or it was fitted.
    """
    if not requested:
        return False, "", 0
    if donor_col not in arm_meta.columns:
        return False, f"the metadata has no {donor_col!r} column", 0

    donors = arm_meta[donor_col].to_numpy(dtype=object)
    is_case = (arm_meta[condition_col] == case).to_numpy(dtype=bool)

    arms_per_donor = pd.Series(is_case).groupby(donors).nunique()
    n_spanning = int((arms_per_donor == 2).sum())
    if n_spanning < 2:
        return (
            False,
            f"only {n_spanning} donor(s) appear in both arms, so the donor block would "
            f"leave {max(n_spanning - 1, 0)} residual degrees of freedom",
            0,
        )

    # The design R will build: two arm-mean columns plus donor treatment contrasts.
    levels = sorted({str(d) for d in donors})
    arms = np.column_stack([~is_case, is_case]).astype(float)
    dummies = (
        np.column_stack([(donors.astype(str) == level).astype(float) for level in levels[1:]])
        if len(levels) > 1
        else np.empty((len(donors), 0))
    )
    design = np.hstack([arms, dummies])
    n_rows, n_cols = design.shape
    if int(np.linalg.matrix_rank(design)) < n_cols or n_rows <= n_cols:
        return (
            False,
            f"the donor block is not estimable ({n_rows} samples, {n_cols} coefficients, "
            f"rank {int(np.linalg.matrix_rank(design))})",
            0,
        )
    return True, "", len(levels)


def _annotate_floor_in_place(
    out_csv: Path,
    arm_meta: pd.DataFrame,
    *,
    donor_col: str,
    condition_col: str,
    case: str,
    paired: bool,
    alpha: float,
) -> dict:
    """Add the design-floor columns to the table R just wrote, and return the metrics.

    The floor has to describe the design that was *fitted*, not the cohort that was
    collected: an arms-only fit on a matched cohort draws on ``C(n, n_case)``
    assignments rather than ``2**k``, and those differ by orders of magnitude on the
    same samples. So when the fit is unpaired the donors are passed as per-sample ids,
    which is what an unpaired fit actually treats them as.

    A failure to read or rewrite the table must not lose the table: the annotation is
    additional information about a result that already exists on disk.
    """
    try:
        table = pd.read_csv(out_csv)
    except (OSError, pd.errors.ParserError):
        return {}

    donors: list[object]
    is_case_list: list[bool]
    if paired:
        donors = list(arm_meta[donor_col].to_numpy(dtype=object))
        is_case_list = list((arm_meta[condition_col] == case).to_numpy(dtype=bool))
    else:
        donors = [f"sample:{i}" for i in range(len(arm_meta))]
        is_case_list = list((arm_meta[condition_col] == case).to_numpy(dtype=bool))

    annotated = annotate_fdr_reachability(
        table, donors=donors, is_case=is_case_list, p_col="PValue", alpha=alpha
    )
    annotated.to_csv(out_csv, index=False)
    if annotated.empty:
        return {}
    return {
        "design_floor_p": float(annotated["design_floor_p"].iloc[0]),
        "family_min_concordant": int(annotated["family_min_concordant"].iloc[0]),
        "family_floor_reachable": bool(annotated["family_floor_reachable"].iloc[0]),
    }


__all__ = ["PropellerMethod"]
