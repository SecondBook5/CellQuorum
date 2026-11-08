"""Pseudobulk edgeR differential-expression method (R)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import pandas as pd

from cellquorum.backends.script_paths import r_script_path
from cellquorum.config.design import (
    DesignConfig,
    validate_design_against_obs,
    validate_design_matrix,
)
from cellquorum.core.contracts import DataContract
from cellquorum.core.exceptions import CellQuorumConfigError
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod
from cellquorum.stages.comparative.differential_expression.ambient_filter import (
    filter_ambient_genes,
    load_expression_profile,
)
from cellquorum.stages.comparative.differential_expression.pseudobulk import (
    aggregate_pseudobulk,
    resolve_donor_pairing,
)
from cellquorum.stages.qc.eligibility import de_eligible_cells

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

# Path to the bundled edgeR script.
_EDGER_R = r_script_path("edger.R")

# Token passed to the edgeR script to request the joint interaction test (an F-test
# over every interaction coefficient) rather than the case-vs-control coefficient.
_INTERACTION_TOKEN = ":interaction"


def build_edger_design_rhs(
    *,
    covariates: list[str],
    paired: bool,
    interactions: list[tuple[str, str]],
    condition_col: str,
    donor_col: str,
) -> tuple[str, str]:
    """Build the edgeR design right-hand side and the coefficient to test.

    The pseudobulk meta CSV renames the condition and donor columns to the fixed
    names ``condition`` and ``donor`` (the edgeR script's contract), so any
    interaction that references those columns must use the renamed alias. The
    additive terms come first (covariates, then the optional donor block, then
    condition) followed by each two-way interaction term.

    When any interaction is requested the tested effect is the interaction itself
    (the ``:interaction`` token, which the R script resolves to a joint
    quasi-likelihood F-test over the interaction coefficients — the factorial
    "is the condition effect modified by this factor?" question). With no
    interaction the token is empty and the R script tests the case-vs-control
    condition coefficient exactly as before.

    Args:
        covariates: Additive covariate columns (kept in their original names).
        paired: Whether a donor block is included.
        interactions: Two-way interaction terms as ``(a, b)`` column pairs.
        condition_col: The obs condition column (aliased to ``condition``).
        donor_col: The obs donor column (aliased to ``donor``).

    Returns:
        ``(design_rhs, test_coef)`` — the formula RHS string and the coefficient
        token for the R script.
    """

    def _alias(name: str) -> str:
        if name == condition_col:
            return "condition"
        if name == donor_col:
            return "donor"
        return name

    rhs_terms = [*covariates]
    if paired:
        rhs_terms.append("donor")
    rhs_terms.append("condition")

    interaction_terms = [f"{_alias(a)}:{_alias(b)}" for a, b in interactions]
    rhs_terms.extend(interaction_terms)

    design_rhs = " + ".join(rhs_terms)
    test_coef = _INTERACTION_TOKEN if interaction_terms else ""
    return design_rhs, test_coef


class _Stop(Exception):
    """A failure that ends the method with a recorded skip."""

    def __init__(self, reason: str, **details: object) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


@dataclass(frozen=True)
class _Settings:
    """Config values ``PseudobulkEdgeRMethod._run`` reads, resolved once."""

    layer: str
    condition_col: str
    donor_col: str
    case: str | None
    control: str | None
    covariates: list[str]
    interactions: list[tuple[str, str]]
    paired: bool
    min_count: int
    min_total_count: int
    timeout: int

    @classmethod
    def from_config(cls, config: dict) -> _Settings:
        return cls(
            layer=config.get("layer", "counts"),
            condition_col=config.get("condition_col", "condition"),
            donor_col=config.get("donor_col", "patient_id"),
            case=config.get("case"),
            control=config.get("control"),
            covariates=list(config.get("covariates", [])),
            interactions=[(pair[0], pair[1]) for pair in config.get("interactions", [])],
            paired=bool(config.get("paired", False)),
            min_count=int(config.get("min_count", 10)),
            min_total_count=int(config.get("min_total_count", 15)),
            timeout=int(config.get("timeout_seconds", 1800)),
        )


def _validate_interactions(de_obs: pd.DataFrame, settings: _Settings) -> None:
    """Validate interaction terms up front.

    Each member must be the condition column or a declared covariate (otherwise the column
    is never carried into the pseudobulk meta), and must be categorical (the model matrix
    treatment-codes it; a numeric column belongs in `covariates` as a continuous term, not
    in an interaction). Fail loudly rather than build a design that silently drops or
    mis-codes the term.
    """
    covariates = settings.covariates
    allowed_interaction_cols = {settings.condition_col, *covariates}
    for a, b in settings.interactions:
        for member in (a, b):
            if member not in allowed_interaction_cols:
                raise CellQuorumConfigError(
                    f"DE interaction term references '{member}', which is neither "
                    f"the condition column ('{settings.condition_col}') nor a declared "
                    f"covariate ({covariates}). Add it to covariates first."
                )
            if pd.api.types.is_numeric_dtype(de_obs[member]):
                raise CellQuorumConfigError(
                    f"DE interaction member '{member}' is numeric; interactions "
                    "require categorical factors. Keep numeric terms as additive "
                    "covariates instead."
                )


def _validate_estimability(de_obs: pd.DataFrame, settings: _Settings, config: dict) -> None:
    """Halt loudly on a comparison that cannot yield valid statistics.

    Two gates. The pre-flight gate catches a missing condition arm, or an arm without
    independent donor replication — the class of silent-wrong DE that the inline
    paired-handling later cannot catch (an unreplicated arm produces confident-but-
    meaningless p-values). Paired completeness and donor-confounding are owned by the
    graceful auto-promote/restrict logic further down, so this validates with paired=False:
    it runs only the existence + per-arm replication checks and never pre-empts that
    recovery. The per-arm floor is config-overridable so a deliberate, documented pilot
    design can lower it rather than tripping a silent cap.

    The multi-factor gate follows. With covariates in the model the fixed-effects design
    ``~ [covariates +] condition`` can be rank-deficient in ways the two-level check cannot
    see -- most commonly a categorical covariate perfectly aliased with the tested
    condition (e.g. batch that coincides with case/control). edgeR would then either error
    cryptically or drop columns and return a meaningless condition coefficient. Numeric
    covariates enter edgeR as continuous terms and are not part of this categorical-aliasing
    check; donor blocking is resolved separately below. Interaction terms extend the design
    too: an empty factorial cell (e.g. no case sample in one batch) makes the interaction
    coefficient a zero column, i.e. rank-deficient. The interaction members are categorical
    by the check in :func:`_validate_interactions`; include them (and the condition) as
    design factors and pass the interaction pairs so the matrix rank reflects what edgeR
    fits. (A donor-nested interaction that survives here still fails loudly as an edgeR
    error -> recorded skip, never a silent-wrong result.)
    """
    condition_col, covariates, interactions = (
        settings.condition_col,
        settings.covariates,
        settings.interactions,
    )
    min_donors_per_arm = int(config.get("min_donors_per_arm", 2))
    validate_design_against_obs(
        de_obs,
        design=DesignConfig(
            donor_col=settings.donor_col,
            condition_col=condition_col,
            case=settings.case,
            control=settings.control,
            paired=False,
        ),
        min_donors_per_arm=min_donors_per_arm,
    )

    categorical_covariates = [c for c in covariates if not pd.api.types.is_numeric_dtype(de_obs[c])]
    interaction_factors = [m for pair in interactions for m in pair]
    design_factors = list(
        dict.fromkeys([*categorical_covariates, condition_col, *interaction_factors])
    )
    if categorical_covariates or interactions:
        comparison_obs = de_obs.loc[
            de_obs[condition_col].isin([settings.case, settings.control]),
            design_factors,
        ].drop_duplicates()
        validate_design_matrix(
            comparison_obs,
            factors=design_factors,
            interactions=list(interactions),
        )


def _apply_ambient_filter(adata: ad.AnnData, config: dict) -> tuple[ad.AnnData, dict | None, dict]:
    """Optional cross-compartment ambient-gene filter, applied before aggregation.

    A flagged gene never reaches edgeR at all (not merely flagged in the output). Off by
    default; see AmbientSpecificityFilterConfig. Returns the (possibly filtered) object, the
    filter's metrics (``None`` when disabled) and its config block.
    """
    ambient_cfg = config.get("ambient_filter") or {}
    if not ambient_cfg.get("enabled"):
        return adata, None, ambient_cfg
    reference_path = ambient_cfg.get("reference_path")
    target_group = ambient_cfg.get("target_group")
    if not reference_path or not target_group:
        raise CellQuorumConfigError(
            "differential_expression.ambient_filter.enabled requires both "
            "reference_path and target_group (the filter needs a reference "
            "profile built from a cohort wider than the one being tested)."
        )
    try:
        profile = load_expression_profile(reference_path)
        adata, ambient_metrics = filter_ambient_genes(
            adata,
            profile=profile,
            target_group=target_group,
            spec_threshold=float(ambient_cfg.get("spec_threshold", 0.25)),
            group_map=ambient_cfg.get("group_map") or None,
        )
    except Exception as exc:  # noqa: BLE001 — a broken reference must not
        # silently proceed unfiltered: that would reproduce the exact
        # false-positive class this filter exists to prevent.
        raise _Stop(f"ambient_filter failed: {str(exc)[:300]}") from exc
    return adata, ambient_metrics, ambient_cfg


class PseudobulkEdgeRMethod(RAnalysisMethod):
    """Donor-blocked pseudobulk DE via edgeR quasi-likelihood.

    Pseudobulk is the primary DE test (spec §6). Aggregates cells to donor x
    condition counts, fits ``~ [covariates +] donor + condition`` (paired) or
    ``~ [covariates +] condition`` (unpaired) in edgeR, and returns the DE table.
    """

    name = "pseudobulk_edger"
    stage_category = "differential_expression"
    r_package = "edgeR"

    def input_contract(self, config: dict) -> DataContract:
        """Require the raw-counts layer plus the design obs columns."""
        layer = config.get("layer", "counts")
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        covariates = list(config.get("covariates", []))
        return DataContract(
            required_layers=[layer],
            required_obs=[condition_col, donor_col, *covariates],
            expression_layer=layer,
            expected_kind="counts",
        )

    def requires_obs(self, config: dict) -> list[str]:
        """Return the design obs columns that must exist for DE to run."""

        # Read the design columns from config.
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        covariates = list(config.get("covariates", []))

        # Require all design columns to exist.
        return [condition_col, donor_col, *covariates]

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        """Aggregate pseudobulk, fit edgeR, and return the DE table."""
        settings = _Settings.from_config(config)
        case, control = settings.case, settings.control

        # A comparison needs both case and control labels.
        if not case or not control:
            return self._skip("case/control labels not set in config")

        # Only QC-eligible cells (DE fit AND inference permissions) may inform this fit,
        # including its replicate counts and estimability checks. ``None`` means QC never
        # ran, so every cell is eligible; QC state without its DE masks raises. The returned
        # object is never subset: eligibility scopes what is analysed, not what is kept.
        eligible = de_eligible_cells(adata.obs)
        de_obs = adata.obs if eligible is None else adata.obs.loc[eligible.to_numpy()]
        n_qc_ineligible = 0 if eligible is None else int((~eligible).sum())

        _validate_interactions(de_obs, settings)
        _validate_estimability(de_obs, settings, config)

        # Rscript + backend + package guards (hoisted to RAnalysisMethod).
        backend = self._resolve_rscript_backend(context, config)
        if isinstance(backend, MethodSkip):
            return backend

        try:
            adata, ambient_metrics, ambient_cfg = _apply_ambient_filter(adata, config)
        except _Stop as stop:
            return self._skip(stop.reason, **stop.details)

        # Aggregate to donor x condition pseudobulk counts.
        pb = aggregate_pseudobulk(
            adata if eligible is None else adata[eligible.to_numpy()],
            layer=settings.layer,
            donor_col=settings.donor_col,
            condition_col=settings.condition_col,
            extra_obs=settings.covariates,
        )

        # Resolve donor pairing so a matched design is never analysed unpaired by
        # accident. This is the safety net for the class of error where
        # `design.paired` is left False on data where every donor contributes both
        # a case and a control sample: blocking on donor removes inter-patient
        # baseline variance and is strictly more powerful. The decision (auto-
        # promotion + complete-pair restriction) is a pure, separately tested
        # helper — see resolve_donor_pairing.
        decision = resolve_donor_pairing(
            pb,
            donor_col=settings.donor_col,
            condition_col=settings.condition_col,
            case=case,
            control=control,
            paired=settings.paired,
        )
        paired = decision.paired
        counts_df = decision.counts

        # Build the design right-hand side and the coefficient to test. With no
        # interaction this is ~ covariates + [donor +] condition, testing the
        # case-vs-control condition coefficient. With interactions the RHS carries
        # the interaction terms and the tested effect becomes the interaction.
        design_rhs, test_coef = build_edger_design_rhs(
            covariates=settings.covariates,
            paired=paired,
            interactions=settings.interactions,
            condition_col=settings.condition_col,
            donor_col=settings.donor_col,
        )

        try:
            out_csv = self._fit_edger(
                backend, context, settings, counts_df, decision.sample_meta, design_rhs, test_coef
            )
        except _Stop as stop:
            return self._skip(stop.reason, **stop.details)

        # Describe the tested effect honestly: an interaction table is a
        # difference-of-differences (F-test over the interaction coefficients),
        # NOT the case-vs-control contrast, so it must never be read as one.
        interactions = settings.interactions
        if interactions:
            interaction_desc = ", ".join(f"{a} x {b}" for a, b in interactions)
            tested_effect = f"interaction {interaction_desc} (modifies {case} vs {control})"
        else:
            tested_effect = f"{case} vs {control}"

        # Return the DE table as an artifact plus provenance metrics.
        design_notes = decision.notes
        notes = [f"Pseudobulk edgeR DE: {tested_effect}, design ~ {design_rhs}.", *design_notes]
        if n_qc_ineligible:
            notes.append(
                f"{n_qc_ineligible} QC-ineligible cell(s) (DE fit/inference masks) excluded "
                "from the pseudobulk."
            )
        metrics = {
            "case": case,
            "control": control,
            "paired": paired,
            "design_rhs": design_rhs,
            "covariates": settings.covariates,
            "interactions": [list(pair) for pair in interactions],
            "tested_effect": tested_effect,
            "n_complete_donor_pairs": decision.n_complete_pairs,
            "design_notes": design_notes,
            "n_cells_qc_ineligible": n_qc_ineligible,
            "n_pseudosamples": int(counts_df.shape[0]),
            "n_genes_input": int(counts_df.shape[1]),
        }
        if ambient_metrics is not None:
            notes.append(
                f"ambient_filter: dropped {ambient_metrics['n_dropped']} of "
                f"{ambient_metrics['n_genes_before']} genes foreign+low-specificity "
                f"for '{ambient_cfg.get('target_group')}'."
            )
            metrics["ambient_filter"] = ambient_metrics
        return StageResult(
            adata=adata,
            artifacts=[
                StageArtifact(
                    name="de_results",
                    path=out_csv,
                    kind="csv",
                    description=f"Pseudobulk edgeR DE ({tested_effect}), ~ {design_rhs}.",
                )
            ],
            notes=notes,
            metrics=metrics,
            backend="rscript",
        )

    @staticmethod
    def _fit_edger(
        backend: Any,
        context: PipelineContext,
        settings: _Settings,
        counts_df: pd.DataFrame,
        sample_meta: pd.DataFrame,
        design_rhs: str,
        test_coef: str,
    ) -> Path:
        """Write the pseudobulk inputs, run the edgeR script, and return its table's path.

        A non-zero exit raises ``_Stop`` -> recorded skip (never crash).
        """
        # Write pseudobulk inputs to scratch.
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        counts_csv = scratch / "pb_counts.csv"
        meta_csv = scratch / "pb_meta.csv"
        counts_df.reset_index(names="sample").to_csv(counts_csv, index=False)
        # Rename design cols so the R script's fixed names (condition/donor) apply.
        meta = sample_meta.rename(
            columns={settings.condition_col: "condition", settings.donor_col: "donor"}
        )
        meta.to_csv(meta_csv)

        # Prepare the output path in the run results directory.
        results_dir = Path(context.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_csv = results_dir / "de_pseudobulk_edger.csv"

        args = [
            str(counts_csv),
            str(meta_csv),
            str(out_csv),
            "condition",
            str(settings.case),
            str(settings.control),
            design_rhs,
            str(settings.min_count),
            str(settings.min_total_count),
            test_coef,
        ]
        try:
            proc = backend.run_script(_EDGER_R, args, timeout=settings.timeout)
        except FileNotFoundError as exc:
            raise _Stop("R execution failed", error=str(exc)[:500]) from exc
        if proc.returncode != 0:
            raise _Stop("edgeR script failed", stderr=proc.stderr.strip()[:500])
        return out_csv


__all__ = ["PseudobulkEdgeRMethod"]
