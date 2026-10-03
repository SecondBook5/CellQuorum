"""scCODA Bayesian compositional differential-abundance method."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import pandas as pd

from cellquorum.backends.sccoda_backend import SCCODA_HELPER
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.comparative.differential_abundance.aggregation import (
    aggregate_celltype_counts,
)
from cellquorum.stages.comparative.differential_abundance.reference_selection import (
    DEFAULT_MIN_MEAN_ABUNDANCE,
    select_compositional_reference,
    split_reference_fits,
)
from cellquorum.stats.depth_confounding import MIN_PAIRED_BLOCKS
from cellquorum.stats.paired_concordance import (
    paired_abundance_concordance,
    qualify_abundance_calls,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from cellquorum.core.context import PipelineContext
    from cellquorum.stages.comparative.differential_abundance.aggregation import CelltypeCounts
    from cellquorum.stages.comparative.differential_abundance.reference_selection import (
        ReferenceChoice,
    )

# An acceptance rate outside this band means the sampler, not the data, decided the
# answer: near zero the chain never moved, and near one it is taking steps too small
# to have explored the posterior. Inside it, an empty result table is a real null.
# The band is the conventional one for Hamiltonian Monte Carlo rather than anything
# tuned here; scCODA's own healthy fits on this engine land at 40-48%.
HEALTHY_ACCEPTANCE_RANGE = (0.10, 0.95)


class SccodaMethod(AnalysisMethod):
    """scCODA Bayesian compositional DA test via sccoda_env subprocess.

    scCODA tests for cell-type compositional differences between conditions using
    a Bayesian model over the simplex (spec §DA). Aggregates cells to sample ×
    cell-type counts, fits a compositional model with an automatic or explicit
    reference cell type, and identifies credible abundance changes via hierarchical
    modeling with spike-and-slab priors.
    """

    name = "sccoda"
    stage_category = "differential_abundance"
    backend = "sccoda"

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

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        """Aggregate cell-type counts, fit scCODA, and return the DA table."""

        # Resolve config fields (all schema-driven; no hardcoded study assumptions).
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")
        case = config.get("case")
        control = config.get("control")
        reference_celltype = config.get("reference_celltype")
        seed = int(config.get("seed", 0))
        num_iterations = int(config.get("num_iterations", 20000))
        timeout = int(config.get("timeout_seconds", 3600))

        # A comparison needs both case and control labels.
        if not case or not control:
            return self._skip("case/control labels not set in config")

        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("sccoda")
            except Exception:
                backend = None
        if backend is None:
            return self._skip("sccoda backend unavailable")

        # Check backend availability (micromamba + sccoda_env import).
        status = backend.status()
        if not status.available:
            missing_str = ", ".join(status.missing) if status.missing else "unknown"
            return self._skip(f"sccoda backend unavailable ({missing_str})", missing=status.missing)

        # Aggregate to sample × cell-type counts.
        cc = aggregate_celltype_counts(
            adata,
            donor_col=donor_col,
            condition_col=condition_col,
            cell_type_col=cell_type_col,
        )

        # Resolve the compositional reference before fitting. scCODA's own
        # "automatic" selection minimises var(p)/mean(p), which equals cv**2 * mean
        # and therefore ranks cell types substantially by rarity -- it will happily
        # make a 0.3% population the denominator of every reported effect. The
        # engine picks on centred-log-ratio variance instead; see
        # ``reference_selection`` for the measurement behind that.
        resolved_reference, reference_reason, reference_choice = self._resolve_reference(
            cc.counts, config, reference_celltype
        )

        # Decide whether to model donor. On a matched cohort, omitting donor tests a
        # between-subject contrast on within-subject data -- the same design error
        # that turned a 1-gene differential-expression result into 645 on this
        # engine. Inferred from the data when nothing is declared, because whether a
        # cohort is paired is a property of the cohort; but a `design` block that
        # declares it outright is an instruction, not a hint, and overriding it from
        # the data would mean `paired: false` silently produced a donor-modelled fit.
        pair_mode = str(config.get("pair_by_donor", "auto")).lower()
        declared = config.get("paired")
        paired_donors = self._count_paired_donors(
            cc.sample_meta, donor_col, condition_col, case, control
        )
        use_pairing, pairing_reason = self._resolve_pairing(
            pair_mode, paired_donors, declared=None if declared is None else bool(declared)
        )

        # Write aggregated inputs to scratch.
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        counts_meta_csv = scratch / "da_counts_meta.csv"

        # counts_meta.csv: samples × cell types + condition column (+ donor when the
        # design is paired, since the helper reads its covariates from this file).
        counts_with_meta = cc.counts.copy()
        counts_with_meta[condition_col] = cc.sample_meta[condition_col]
        if use_pairing:
            counts_with_meta[donor_col] = cc.sample_meta[donor_col].astype(str)
        counts_with_meta.to_csv(counts_meta_csv, index=True)

        # Prepare the output path in the run results directory.
        results_dir = Path(context.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_csv = results_dir / "da_sccoda.csv"
        writer = StageArtifactWriter.from_context(context)
        diagnostics_json = scratch / "da_sccoda_diagnostics.json"

        args = [
            str(counts_meta_csv),
            str(out_csv),
            condition_col,
            case,
            control,
            str(seed),
            str(num_iterations),
        ]

        # Positional reference argument: the resolved reference, which is the config
        # value when set and the engine's pick otherwise.
        if resolved_reference:
            args.append(resolved_reference)

        if use_pairing:
            args += ["--covariates", donor_col]
        args += ["--diagnostics-json", str(diagnostics_json)]

        # Invoke the sccoda helper; non-zero exit -> recorded skip (never crash).
        try:
            proc = backend.run_helper(SCCODA_HELPER, args, timeout=timeout)
        except FileNotFoundError as exc:
            return self._skip("helper execution failed", error=str(exc)[:500])
        except subprocess.TimeoutExpired as exc:
            # A configured timeout must skip this method, not crash the stage
            # and abort the sibling methods still queued after it.
            return self._skip(f"helper execution timed out after {timeout}s", error=str(exc)[:500])
        if proc.returncode != 0:
            return self._skip("sccoda helper failed", stderr=proc.stderr.strip()[:500])

        # Read output CSV to compute metrics + back the composition figure (skip-not-crash).
        #
        # The helper returns TWO fits stacked whenever a reference was resolved --
        # which is the default -- one at the engine's reference and one at scCODA's
        # own automatic pick, distinguished only by the ``reference`` column. That is
        # a sensitivity analysis worth having, but it means the table holds every
        # cell type twice, so anything that reads it whole is counting each state
        # twice: `credible_effect.sum()` on the stacked frame reported 6 credible
        # effects for a cohort with 3, and it did so on every scCODA run this engine
        # has ever produced. Split the primary fit out before anything is measured.
        df, primary, sensitivity, n_credible, warnings = self._read_reported_fit(
            out_csv, resolved_reference
        )

        # Does the reported credible set survive scCODA's own reference choice? A
        # compositional model's effects are all relative to its denominator, so
        # "unchanged under a different reference" is the robustness statement a
        # reader of a compositional result actually needs, and the second fit is
        # already paid for.
        sensitivity_metrics = self._reference_sensitivity(primary, sensitivity)

        notes = [
            f"scCODA DA: {case} vs {control}, "
            f"reference={resolved_reference or 'auto'}, "
            f"iterations={num_iterations}.",
            *cc.notes,
        ]
        notes.append(f"Compositional reference: {reference_reason}.")
        notes.append(f"Design: {pairing_reason}.")

        # Say what the second fit found, in the run summary. A compositional result
        # whose credible set changes with the denominator is a different claim from
        # one that does not, and the difference should not need a CSV to see.
        stable = sensitivity_metrics["credible_set_reference_stable"]
        if stable is True:
            notes.append(
                "Reference sensitivity: the credible set is unchanged when the model is "
                "refitted against scCODA's own automatically selected reference."
            )
        elif stable is False:
            notes.append(
                f"Reference sensitivity: refitting against scCODA's own automatically "
                f"selected reference changes the credible set "
                f"({sensitivity_metrics['credible_set_reference_disagreement']} differ), so "
                f"these effects depend on the choice of denominator."
            )

        # Audit whether each called effect is a cohort-wide shift or a subset one.
        # scCODA reports a mean, and a mean cannot distinguish those; on this
        # engine's own cohort its single credible call turned out to move in the
        # reported direction in a minority of donors.
        concordance = paired_abundance_concordance(
            cc.counts,
            cc.sample_meta[donor_col],
            cc.sample_meta[condition_col],
            case=case,
            control=control,
        )
        extra_artifacts: list[StageArtifact] = []
        if primary is not None and not concordance.empty:
            # Annotate each fit separately. The concordance columns are properties of
            # the COUNTS rather than of the fit, so both blocks carry the same values;
            # what must not be duplicated is the notes, which are the run summary's
            # warning that a called effect is not donor-consistent. One warning per
            # call, from the fit being reported.
            primary, concordance_notes = qualify_abundance_calls(primary, concordance)
            if sensitivity is not None and not sensitivity.empty:
                sensitivity, _ = qualify_abundance_calls(sensitivity, concordance)
            df = self._restack(primary, sensitivity)
            warnings += self._write_back(df, out_csv, "donor-concordance annotations")
            notes.extend(concordance_notes)
            extra_artifacts.append(
                writer.table(
                    concordance,
                    "da_sccoda_donor_concordance.csv",
                    name="da_donor_concordance",
                    description=(
                        "Per-cell-type donor-level concordance for the scCODA calls: "
                        "sign test on the paired log-ratio change, leave-one-donor-out "
                        "stability, and the resulting pattern."
                    ),
                )
            )
            consistent = concordance[concordance["pattern"] == "consistent"]["cell_type"].tolist()
            if consistent:
                notes.append(
                    f"Donor-consistent abundance shifts (sign test p<0.05, stable to dropping "
                    f"any one donor): {', '.join(map(str, consistent))}."
                )

        # Record the reference criterion, so a reader can see what was rejected.
        if reference_choice is not None and not reference_choice.criterion.empty:
            extra_artifacts.append(
                writer.table(
                    reference_choice.criterion,
                    "da_sccoda_reference_criterion.csv",
                    name="da_reference_criterion",
                    description=(
                        "Compositional reference selection: per-cell-type presence, mean "
                        "abundance, centred-log-ratio variance and scCODA's own dispersion "
                        "criterion, with the selected row marked."
                    ),
                )
            )

        # Sampler diagnostics: the only thing that separates "healthy chain found
        # nothing" from "chain never explored the posterior", both of which produce
        # an empty result table.
        diagnostics, diagnostic_notes, diagnostic_warnings = self._read_diagnostics(
            diagnostics_json, n_credible
        )
        notes.extend(diagnostic_notes)
        warnings += diagnostic_warnings

        # Auto-emit the compositional DA figure (gated, skip-not-crash). Shows the
        # resolved reference's block rather than scCODA's automatic one, so the
        # figure and the reported effects share a denominator.
        figure_artifacts, figure_warnings = self._composition_artifacts(
            df,
            cc,
            condition_col=condition_col,
            donor_col=donor_col,
            case=case,
            control=control,
            config=config,
            context=context,
            reference=resolved_reference or "auto",
        )

        warnings += figure_warnings

        return StageResult(
            adata=adata,
            artifacts=[
                StageArtifact(
                    name="da_results",
                    path=out_csv,
                    kind="csv",
                    description=(
                        f"scCODA DA ({case} vs {control}), "
                        f"reference={resolved_reference or 'auto'}, "
                        f"iterations={num_iterations}."
                    ),
                ),
                *extra_artifacts,
                *figure_artifacts,
            ],
            notes=notes,
            warnings=warnings,
            metrics={
                "case": case,
                "control": control,
                "reference_celltype": resolved_reference,
                "reference_source": (
                    "config"
                    if reference_celltype
                    else ("engine" if reference_choice is not None else "sccoda_automatic")
                ),
                "reference_relaxed": (
                    bool(reference_choice.relaxed) if reference_choice is not None else None
                ),
                "paired_by_donor": use_pairing,
                "n_paired_donors": paired_donors,
                "seed": seed,
                "num_iterations": num_iterations,
                "n_samples": int(cc.counts.shape[0]),
                "n_celltypes": int(cc.counts.shape[1]),
                "n_unlabeled_cells": cc.n_unlabeled,
                "n_credible": n_credible,
                "n_donor_consistent": (
                    int((concordance["pattern"] == "consistent").sum())
                    if not concordance.empty
                    else None
                ),
                **sensitivity_metrics,
                **diagnostics,
            },
            backend="sccoda",
        )

    def _resolve_reference(
        self, counts: pd.DataFrame, config: dict, configured: str | None
    ) -> tuple[str | None, str, ReferenceChoice | None]:
        """Pick the compositional reference and explain the choice.

        Precedence is config, then the engine's criterion, then scCODA's own
        automatic selection. Config wins because reproducing a published table has
        to stay possible; scCODA's is last because it is the one that is wrong.
        """

        if configured:
            return str(configured), f"{configured}, set explicitly in config", None

        if not bool(config.get("select_reference", True)):
            return None, "left to scCODA's automatic selection (select_reference disabled)", None

        try:
            choice = select_compositional_reference(
                counts,
                min_mean_abundance=float(
                    config.get("min_reference_abundance", DEFAULT_MIN_MEAN_ABUNDANCE)
                ),
            )
        except Exception as exc:
            return None, f"left to scCODA (reference selection failed: {type(exc).__name__})", None

        if choice.cell_type is None:
            return None, f"left to scCODA ({choice.reason})", choice
        return str(choice.cell_type), choice.reason, choice

    def _read_reported_fit(
        self, out_csv: Path, resolved_reference: str | None
    ) -> tuple[
        pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, int | None, list[str]
    ]:
        """Read scCODA's table back and mark the fit being reported in the file itself.

        Without the marker, "which of these two rows for cell type 3 is the result?" is
        answerable only by knowing what the engine resolved, which is in a different file.

        Returns:
            ``(table, primary fit, sensitivity fit, number of credible effects, warnings)``.
            If the table cannot be read, the fits and count are ``None`` and the warning says
            which downstream figures are therefore unavailable, so the run does not look
            complete when it is not.
        """
        df = None
        try:
            df = pd.read_csv(out_csv)
            primary, sensitivity = split_reference_fits(df, resolved_reference)
            n_credible = int(primary["credible_effect"].sum())
            df = self._restack(primary, sensitivity)
            df.to_csv(out_csv, index=False)
        except Exception as exc:  # noqa: BLE001 -- reported below, never silent
            return (
                df,
                None,
                None,
                None,
                [
                    f"scCODA finished but its results table {Path(out_csv).name} could not be read "
                    f"back ({type(exc).__name__}: {exc}). The credible-effect count, the "
                    "reference-sensitivity check and the donor-concordance annotations are "
                    "unavailable for this run."
                ],
            )
        return df, primary, sensitivity, n_credible, []

    @staticmethod
    def _write_back(frame: pd.DataFrame, out_csv: Path, purpose: str) -> list[str]:
        """Rewrite the results table; return a warning, not silence, if that fails."""
        try:
            frame.to_csv(out_csv, index=False)
        except Exception as exc:  # noqa: BLE001 -- reported below, never silent
            return [
                f"scCODA's results table {Path(out_csv).name} could not be rewritten with its "
                f"{purpose} ({type(exc).__name__}: {exc}); the file on disk lacks them."
            ]
        return []

    @staticmethod
    def _restack(primary: pd.DataFrame, sensitivity: pd.DataFrame | None) -> pd.DataFrame:
        """Put the two fits back into one self-describing table.

        The ``is_primary`` marker appears only when there is a second fit to be
        confused with. A single-fit table needs no column saying every row is the
        result, and adding one would change the documented output schema of the
        auto-only path for nothing.
        """

        if sensitivity is None or sensitivity.empty:
            return primary
        return pd.concat(
            [primary.assign(is_primary=True), sensitivity.assign(is_primary=False)],
            ignore_index=True,
        )

    @staticmethod
    def _reference_sensitivity(
        primary: pd.DataFrame | None, sensitivity: pd.DataFrame | None
    ) -> dict:
        """
        Compare the reported credible set against the second fit's.

        Every effect a compositional model reports is relative to its reference, so
        the honest question about a compositional result is not only "is it credible"
        but "is it credible against a different denominator". The engine already pays
        for a second fit at scCODA's own automatic reference, so the answer is free —
        it just has to be recorded rather than left as two indistinguishable blocks
        in one CSV.

        Returns:
            Metrics describing the second fit, all ``None`` when only one fit ran, so
            the metric schema does not change shape between runs.
        """

        absent = {
            "n_credible_alternate_reference": None,
            "credible_set_reference_stable": None,
            "credible_set_reference_disagreement": None,
        }
        if primary is None or sensitivity is None or sensitivity.empty:
            return absent
        if "credible_effect" not in sensitivity.columns or "cell_type" not in sensitivity.columns:
            return absent

        def called(frame: pd.DataFrame) -> set[str]:
            hit = frame[frame["credible_effect"].astype(bool)]
            return {str(value) for value in hit["cell_type"]}

        primary_called = called(primary)
        alternate_called = called(sensitivity)
        disagreement = sorted(primary_called.symmetric_difference(alternate_called))
        return {
            "n_credible_alternate_reference": int(len(alternate_called)),
            "credible_set_reference_stable": not disagreement,
            "credible_set_reference_disagreement": ", ".join(disagreement) or None,
        }

    @staticmethod
    def _count_paired_donors(
        sample_meta: pd.DataFrame,
        donor_col: str,
        condition_col: str,
        case: str,
        control: str,
    ) -> int:
        """Count donors contributing a sample to both arms of the comparison."""

        if donor_col not in sample_meta.columns or condition_col not in sample_meta.columns:
            return 0
        arms = sample_meta[condition_col].astype(str)
        by_donor = sample_meta[donor_col].astype(str)
        has_case = set(by_donor[arms == str(case)])
        has_control = set(by_donor[arms == str(control)])
        return len(has_case & has_control)

    @staticmethod
    def _resolve_pairing(
        pair_mode: str, paired_donors: int, *, declared: bool | None = None
    ) -> tuple[bool, str]:
        """Decide whether donor enters the model formula, and say why.

        A donor term is only estimable alongside condition when at least one donor
        spans both arms; with none, donor is collinear with condition and the
        contrast of interest disappears. That is a hard constraint, so "always" is
        honoured only where the design permits it.

        Under ``auto``, an explicit ``design.paired`` declaration wins over the
        data-driven rule below it. Inferring pairing from the data is the right default
        when nothing has been declared, but it is the wrong answer when something has:
        a cohort whose donors happen to span both arms while the design says
        ``paired: false`` would otherwise be donor-modelled against instruction, and
        the fit would differ from every other method in the same run.

        The override is deliberately one-directional. ``declared=False`` turns pairing
        off, but ``declared=True`` does not force it on below the
        :data:`MIN_PAIRED_BLOCKS`-pair floor: declaring a cohort matched does not create
        the degrees of freedom to model it, and ``pair_by_donor: always`` already exists
        for a caller who wants the block regardless. So the two paths differ in what
        they can cost — turning pairing off can only remove parameters, while turning it
        on adds one per donor to a compositional fit that has few samples to spare.

        Args:
            pair_mode: ``auto``, ``always``, or ``never``.
            paired_donors: Donors contributing a sample to both arms.
            declared: ``design.paired`` if a design block declared it, else ``None``.
        """

        if pair_mode == "never":
            return False, "unpaired (pair_by_donor=never)"

        if paired_donors == 0:
            reason = "unpaired: no donor contributes both arms, so a donor term is collinear "
            reason += "with condition"
            if pair_mode == "always":
                reason += " (pair_by_donor=always could not be honoured)"
            return False, reason

        if pair_mode == "auto" and declared is False:
            return False, (
                f"unpaired: {paired_donors} donor(s) contribute both arms, but the design "
                f"declares paired=false and pair_by_donor is auto"
            )

        if pair_mode == "always":
            note = f"donor-paired on {paired_donors} donor(s) (pair_by_donor=always)"
            if paired_donors < MIN_PAIRED_BLOCKS:
                note += (
                    f"; below the {MIN_PAIRED_BLOCKS}-pair floor, so treat the condition effect "
                    f"as provisional"
                )
            return True, note

        if paired_donors >= MIN_PAIRED_BLOCKS:
            return True, (
                f"donor-paired: {paired_donors} donors contribute both arms, so condition is "
                f"estimated within donor"
            )

        return False, (
            f"unpaired: only {paired_donors} donor(s) contribute both arms, below the "
            f"{MIN_PAIRED_BLOCKS}-pair floor at which a within-donor contrast is worth the "
            f"degrees of freedom"
        )

    @staticmethod
    def _read_diagnostics(path: Path, n_credible: int | None) -> tuple[dict, list[str], list[str]]:
        """Summarise the sampler diagnostics and flag a fit that cannot be trusted.

        The distinction this exists for: an empty result table from a healthy chain
        is a finding, and an empty result table from a chain that never moved is a
        bug. The acceptance rate is what tells them apart, and scCODA prints it to
        stdout and then discards it.

        Returns:
            ``(metrics, notes, warnings)``. A diagnostics file that exists but cannot be read
            is a warning: without it an empty result table cannot be interpreted.
        """

        if not path.exists():
            return {}, [], []
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 -- reported below, never silent
            return (
                {},
                [],
                [
                    f"scCODA's sampler diagnostics {path.name} could not be read "
                    f"({type(exc).__name__}: {exc}), so an empty result cannot be told apart "
                    "from a chain that never explored the posterior."
                ],
            )

        fits = payload.get("fits") or []
        if not fits:
            return {}, [], []

        notes: list[str] = []
        low, high = HEALTHY_ACCEPTANCE_RANGE
        rates = []
        for fit in fits:
            rate = fit.get("acceptance_rate")
            if rate is None:
                continue
            rates.append(float(rate))
            if not (low <= float(rate) <= high):
                notes.append(
                    f"scCODA fit (reference={fit.get('reference')}) had an acceptance rate of "
                    f"{float(rate):.1%}, outside the healthy {low:.0%}-{high:.0%} band: the "
                    f"sampler did not explore the posterior, so this fit's effects -- including "
                    f"any absence of effects -- are not interpretable."
                )

        primary = fits[-1]
        threshold = primary.get("fdr_threshold_probability")
        if not n_credible and rates and all(low <= r <= high for r in rates):
            detail = (
                f" The inclusion-probability threshold needed to hold the target FDR was "
                f"{float(threshold):.3g}."
                if threshold is not None
                else ""
            )
            notes.append(
                f"scCODA found no credible compositional change, and the sampler was healthy "
                f"(acceptance {min(rates):.1%}-{max(rates):.1%}), so this is a null result rather "
                f"than a failed fit.{detail}"
            )

        metrics = {
            "sccoda_formula": payload.get("formula"),
            "sccoda_acceptance_rate_min": min(rates) if rates else None,
            "sccoda_acceptance_rate_max": max(rates) if rates else None,
            "sccoda_fdr_threshold_probability": threshold,
            "sccoda_max_inclusion_probability": primary.get("max_inclusion_probability"),
            "sccoda_mean_prior_distance": primary.get("mean_prior_distance"),
        }
        missing = payload.get("covariates_missing") or []
        if missing:
            notes.append(
                f"Requested covariate(s) {', '.join(map(str, missing))} were not present in the "
                f"aggregated sample table and were dropped from the model formula."
            )
        return metrics, notes, []

    def _composition_artifacts(
        self,
        effects: pd.DataFrame | None,
        cc: CelltypeCounts,
        *,
        condition_col: str,
        donor_col: str,
        case: str,
        control: str,
        config: dict,
        context: PipelineContext,
        reference: str = "auto",
    ) -> tuple[list[StageArtifact], list[str]]:
        """Build the two-panel scCODA composition figure (gated, default on).

        Pairs a per-condition proportion dumbbell with each cell type's posterior
        inclusion probability, driven by the scCODA effects table and the
        aggregated sample × cell-type counts. Study-agnostic: condition
        labels/colors come from config, no biology hardcoded. Emits nothing when
        disabled, when the effects table is unreadable/empty, or when the counts
        yield no proportions. Never raises: a plotting failure comes back as a warning.
        """

        if not config.get("write_da_figure", True) or effects is None or effects.empty:
            return [], []

        # Local imports keep matplotlib off the pure-method import path.
        from cellquorum.stages.comparative.differential_abundance.aggregation import (
            build_composition_proportions,
        )
        from cellquorum.stages.comparative.differential_abundance.da_figures import (
            plot_sccoda_composition,
            render_da_figure,
        )

        def build() -> Figure | None:
            proportions = build_composition_proportions(
                cc.counts,
                cc.sample_meta[condition_col],
                cc.sample_meta[donor_col],
                case=case,
                control=control,
            )
            if proportions.empty:
                return None
            return plot_sccoda_composition(
                effects,
                proportions,
                case=case,
                control=control,
                reference=reference,
                cluster_labels=config.get("cluster_labels") or {},
            )

        figures_dir = Path(getattr(context.paths, "figures", context.paths.results))
        return render_da_figure(
            build,
            figures_dir / "differential_abundance",
            "da_sccoda_composition",
            f"scCODA compositional DA ({case} vs {control}): per-condition proportion "
            "dumbbell + posterior inclusion probability.",
        )


__all__ = ["SccodaMethod"]
