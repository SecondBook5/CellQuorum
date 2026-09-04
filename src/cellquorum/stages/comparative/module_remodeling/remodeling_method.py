"""Module-remodeling method: condition effects on per-cell module activity.

Takes the cells x programs score matrix ``state_scoring`` already wrote, resolves a
group axis (a labelled partition or an existing obs column), and answers three
questions with the house statistical primitives:

* per module x group — how large is the condition shift (donor-random-intercept
  mixed model, BH-FDR across the whole family);
* per group — is the *multivariate* module profile different (seeded PERMANOVA on
  per-sample module vectors);
* across programs — how is module activity co-organized (pairwise correlation at
  the sample level, with the condition-adjusted partial correlation and the
  shared-gene count beside it, plus the descriptive per-cell matrix);
* per module — does the shift survive removing library depth (paired audit against
  ``log1p(depth)``, one row per program plus the contrast index).

Then it draws them. That last part is the reason the stage exists: the scoring
stage upstream produces tables and no figures, so a curated module set previously
ended a run as numbers in a CSV.

The depth audit is not optional and is not a diagnostic afterthought. Every score
here is a continuous per-cell quantity built by summing or ranking gene detections,
so deeper libraries score higher on all of them at once; if depth also differs
between the arms — which is a property of the cohort, not of the analysis — then a
condition effect on any of these programs has a second, uninteresting explanation
that no amount of donor pairing removes. A run that reports the effect and not the
audit cannot tell the two apart, so this stage writes both or says why it could not.
The audit is also not purely subtractive: because confounding has a direction, removing
depth can *expose* a fall it had been cancelling, and those rows come back as
``depth_masked`` leads rather than being filed as "nothing to audit".

Nothing here raises: every precondition failure is a recorded skip, and every
figure is drawn through ``render_figure`` so one bad panel cannot sink a stage.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.comparative.module_remodeling import mr_figures
from cellquorum.stages.comparative.module_remodeling.subtypes import UNASSIGNED, label_clusters
from cellquorum.stats import (
    depth_confound_audit,
    lmm_effect_sizes,
    permanova_by_group,
    program_correlation_tests,
    signed_program_contrast_index,
)
from cellquorum.stats.depth_confounding import MIN_PAIRED_BLOCKS
from cellquorum.visualization import figstyle
from cellquorum.visualization.figstyle import render_figure
from cellquorum.visualization.program_correlation import (
    program_correlation_heatmap,
    program_correlation_slopes,
)

# Subdirectory used under both the results and figures namespaces.
_STAGE_DIR = "module_remodeling"

# Depth covariates to look for, in preference order, when the config does not name one.
# Gene count first: it is the quantity a detection-based score is most nearly monotone
# in, and it saturates less than UMI count, so residualising on it removes more of the
# nuisance and less of the biology. Both are scanpy's standard ``calculate_qc_metrics``
# names, which is why no dataset has to declare anything for the audit to run.
_DEPTH_COLUMN_CANDIDATES: tuple[str, ...] = ("n_genes_by_counts", "total_counts")

# Verdicts from :func:`~cellquorum.stats.depth_confound_audit` that contradict a
# reported effect. ``depth_driven`` means the effect reversed or largely vanished under
# adjustment; ``attenuated`` means it kept its sign and most of its size but lost
# significance. The first is a warning because reporting the row would be wrong; the
# second is a note because reporting it *with the caveat* is legitimate.
_DEPTH_CONTRADICTED = "depth_driven"
_DEPTH_WEAKENED = "attenuated"

# The verdict that goes the other way: no unadjusted effect, a significant adjusted one,
# because depth was pushing the metric against its condition effect and cancelling it. A
# note rather than a warning — nothing reported is wrong — but it must be said, or the one
# row the audit *found* is the only one nobody reads.
_DEPTH_MASKED = "depth_masked"


def _resolve_depth_column(obs: pd.DataFrame, configured: object) -> str | None:
    """Pick the depth covariate: the configured one, else the first standard one present.

    A configured name that is absent from ``obs`` returns ``None`` rather than falling
    back silently. Falling back would run the audit against a *different* covariate from
    the one the manifest asked for and label the result as if it had honoured the
    request, which is worse than not auditing: the caller would read a passing audit as
    evidence about the column they named.
    """
    if configured:
        name = str(configured)
        return name if name in obs.columns else None
    for candidate in _DEPTH_COLUMN_CANDIDATES:
        if candidate in obs.columns:
            return candidate
    return None


def _group_labels(values: pd.Series) -> pd.Series:
    """Render a group column as labels, preserving "unassigned" as missing.

    Two things a plain ``astype(str)`` gets wrong on a real partition column.
    A clustering method writes a float column with ``NaN`` for the cells it could
    not place, and ``astype(str)`` turns those into the string ``"nan"`` — a group
    that then carries its own effect-table rows, its own PERMANOVA row and its own
    column on the flagship figure. And an integer-valued float renders as
    ``"1.0"``, so the figure's column headers read like measurements instead of
    cluster names.
    """
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric.dropna()
        integral = bool((finite % 1 == 0).all()) if not finite.empty else True
        rendered = numeric.map(
            lambda v: (pd.NA if pd.isna(v) else (f"{int(v)}" if integral else f"{v:g}".rstrip()))
        )
        return rendered.astype("string")
    labels = values.astype("string")
    # Categorical/object columns spell missing several ways; all of them mean the
    # same thing and none of them is a subtype.
    return labels.mask(labels.isin(["nan", "NaN", "None", "<NA>", ""]))


def _natural_key(label: str) -> tuple[int, float, str]:
    """Sort key that puts cluster 2 before cluster 10.

    Cluster labels arrive as numbers rendered to strings, and a plain sort gives
    1, 10, 2 — which reads as a mistake in a figure's column headers. Numeric
    labels sort numerically and ahead of names, so a partition that mixes
    ``"3"`` with ``"unassigned"`` still has one defined order.
    """
    text = str(label)
    try:
        return (0, float(text), "")
    except ValueError:
        return (1, 0.0, text)


def _resolve_group_order(labels: pd.Series, configured: list[str]) -> list[str]:
    """Canonical group order: the configured one, else natural order.

    Configured labels that no cell carries are dropped rather than drawn as empty
    columns, and observed labels the config forgot are appended rather than
    silently excluded from the figure.
    """
    observed = [str(label) for label in dict.fromkeys(labels.dropna().astype(str))]
    if configured:
        chosen = [str(label) for label in configured if str(label) in set(observed)]
        chosen += [label for label in sorted(observed, key=_natural_key) if label not in chosen]
        return chosen
    return sorted(observed, key=_natural_key)


def _apply_label_order(frame: pd.DataFrame, *, column: str, order: list[str]) -> pd.DataFrame:
    """Sort a table so ``column`` follows ``order``, leaving other columns' order alone.

    The sort is stable, so applying this twice — groups first, then programs —
    yields a program-major table whose groups run in the group order inside each
    program, without either call needing to know about the other.

    The tables are the replot source, so they carry the same order the panels are
    drawn in; a CSV ordered differently from the figure beside it is a small trap
    for whoever redraws it. Labels absent from ``order`` sort last rather than
    being dropped: an unranked row is still a result.
    """
    if frame.empty or column not in frame.columns:
        return frame
    rank = {label: index for index, label in enumerate(order)}
    keyed = frame.assign(
        _label_rank=frame[column].astype(str).map(lambda label: rank.get(label, len(rank)))
    )
    return keyed.sort_values("_label_rank", kind="stable").drop(columns="_label_rank")


class ModuleRemodelingMethod(AnalysisMethod):
    """Condition effects on module activity, per group, with figures."""

    name = "module_remodeling"
    stage_category = "module_remodeling"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract(required_obs=self.requires_obs(config))

    def requires_obs(self, config: dict) -> list[str]:
        keys = (
            config.get("donor_col"),
            config.get("condition_col"),
            config.get("sample_col"),
            config.get("group_col") or config.get("cluster_col"),
        )
        return [k for k in keys if k]

    # ------------------------------------------------------------------ #
    # input resolution                                                    #
    # ------------------------------------------------------------------ #
    def _score_frame(self, adata: ad.AnnData, config: dict) -> pd.DataFrame | MethodSkip:
        """Pull the score matrix out of ``obsm`` with its real column names.

        The column order comes from ``uns`` rather than being assumed, because the
        scored module set is config-driven: a matrix whose columns were silently
        mislabelled would put the right numbers under the wrong module names, and
        every figure downstream would look perfectly plausible.
        """
        score_key = config.get("score_key", "X_state_aucell")
        if score_key not in adata.obsm:
            return self._skip(
                f"score matrix '{score_key}' missing from adata.obsm — run state_scoring "
                "with the aucell method first",
                score_key=score_key,
            )

        matrix = adata.obsm[score_key]
        if isinstance(matrix, pd.DataFrame):
            frame = matrix.copy()
            frame.index = adata.obs_names
            return frame

        array = np.asarray(matrix, dtype=float)
        uns_key = config.get("programs_uns_key", "state_aucell")
        raw_names = (adata.uns.get(uns_key) or {}).get("programs") if uns_key in adata.uns else None
        # Coerce before testing. The names go in as a list and come back out of an
        # h5ad checkpoint as an object-dtype ndarray, and `if not <ndarray>` raises
        # rather than being falsy — so the truthiness test has to happen on a list,
        # after the coercion, or the stage dies on every real run while passing
        # every in-memory fixture.
        names = [str(name) for name in raw_names] if raw_names is not None else []
        if not names:
            return self._skip(
                f"program names not found at adata.uns['{uns_key}']['programs']; refusing to "
                "label score columns positionally",
                programs_uns_key=uns_key,
            )
        if len(names) != array.shape[1]:
            return self._skip(
                f"adata.uns['{uns_key}']['programs'] lists {len(names)} program(s) but "
                f"obsm['{config.get('score_key')}'] has {array.shape[1]} column(s)",
                n_names=len(names),
                n_columns=int(array.shape[1]),
            )
        return pd.DataFrame(array, index=adata.obs_names, columns=list(names))

    @staticmethod
    def _program_genes(adata: ad.AnnData, config: dict) -> dict[str, list[str]] | None:
        """The genes each program was scored on, when the scoring stage recorded them.

        Two programs that share genes correlate by construction, so the correlation
        table has to be able to say how many genes each pair shares. Returning ``None``
        when the lists are absent is deliberate: the table then marks the overlap
        *unknown* rather than zero, and "we did not check" never reads as "disjoint".
        """
        uns_key = config.get("programs_uns_key", "state_aucell")
        block = adata.uns.get(uns_key)
        genes = block.get("genes") if isinstance(block, dict) else None
        if not isinstance(genes, dict) or not genes:
            return None
        # h5ad round-trips these as object-dtype ndarrays of numpy str_, which compare
        # unequal to the plain Python strings the score columns carry.
        return {str(name): [str(gene) for gene in members] for name, members in genes.items()}

    # ------------------------------------------------------------------ #
    # run                                                                 #
    # ------------------------------------------------------------------ #
    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        donor_col = config.get("donor_col")
        condition_col = config.get("condition_col")
        sample_col = config.get("sample_col")
        case = config.get("case")
        control = config.get("control")
        cluster_col = config.get("cluster_col")
        group_col = config.get("group_col")
        key_added = config.get("key_added", "module_subtype")
        index_key = config.get("index_key", "contrast_index")
        seed = int(config.get("seed", 1337))

        if not case or not control:
            return self._skip("case/control condition labels not set in config")
        for label, column in (("donor_col", donor_col), ("condition_col", condition_col)):
            if not column or column not in adata.obs.columns:
                return self._skip(f"{label} '{column}' unresolved or absent from obs")

        scores = self._score_frame(adata, config)
        if isinstance(scores, MethodSkip):
            return scores

        requested = list(config.get("programs") or [])
        if requested:
            missing = [p for p in requested if p not in scores.columns]
            if missing:
                return self._skip(
                    f"configured program(s) not present in the score matrix: {sorted(missing)}"
                )
            scores = scores[requested]
        if scores.shape[1] == 0:
            return self._skip("the score matrix has no program columns")

        notes: list[str] = []
        warnings: list[str] = []

        # ---- Group axis: an existing obs column, or clusters we label. ----
        if group_col:
            if group_col not in adata.obs.columns:
                return self._skip(f"group_col '{group_col}' absent from obs")
            group_values = _group_labels(adata.obs[group_col])
            # No labelling happened, so there is no label table. This used to be a
            # zero-row frame with three of the real table's six columns, written out
            # under a description promising "cluster-level signature labels": a run
            # that grouped on an existing column was indistinguishable, from its own
            # output, from one whose labelling ran and named nothing. It also broke
            # any reader that reached for ``top_z``.
            subtype_table = None
            notes.append(
                f"group axis read directly from obs['{group_col}']; no signature "
                "labelling was run, so no module_subtypes.csv is written (the groups "
                "keep the column's own names)"
            )
        else:
            if not cluster_col or cluster_col not in adata.obs.columns:
                return self._skip(f"cluster_col '{cluster_col}' unresolved or absent from obs")
            try:
                group_values, subtype_table, subtype_notes = label_clusters(
                    scores,
                    adata.obs[cluster_col].astype(str),
                    signatures=config.get("subtype_signatures") or {},
                    min_margin=float(config.get("min_margin", 0.0)),
                )
            except ValueError as exc:
                return self._skip(f"subtype labelling failed: {exc}")
            notes.extend(subtype_notes)
            adata.obs[key_added] = pd.Categorical(group_values)
            notes.append(
                f"obs['{key_added}']: "
                + ", ".join(
                    f"{label} n={count}" for label, count in group_values.value_counts().items()
                )
            )

        metadata = adata.obs.copy()
        group_field = group_col or key_added
        # Overwrite rather than add: when the caller supplied ``group_col`` the
        # statistics read that column by name, so the rendered labels have to land
        # *there*. Adding them under ``key_added`` instead left every downstream
        # call reading the raw column, which is how "1.0" and "nan" reached the
        # figure even after the labels beside them were correct.
        metadata[group_field] = pd.Series(
            np.asarray(group_values, dtype=object), index=metadata.index
        )

        # ---- The signed identity axis, when the caller defined one. ----
        index_up = [p for p in (config.get("index_up") or []) if p in scores.columns]
        index_down = [p for p in (config.get("index_down") or []) if p in scores.columns]
        index_values = None
        index_frame: pd.DataFrame | None = None
        if index_up and index_down:
            index_values = signed_program_contrast_index(scores, up=index_up, down=index_down)
            adata.obs[index_key] = index_values.to_numpy(dtype=float)
            metadata[index_key] = adata.obs[index_key].to_numpy()
        elif config.get("index_up") or config.get("index_down"):
            # One side present and the other empty makes the "contrast" a plain
            # sum, which is a different quantity wearing the same name.
            warnings.append(
                f"contrast index skipped: index_up resolved to {len(index_up)} program(s) and "
                f"index_down to {len(index_down)}; both sides are required"
            )

        # ---- Cells with no group are excluded from the group statistics. ----
        # They keep their obs columns above (they are real cells with a real module
        # score and a real index value); what they cannot do is form a group.
        assigned = metadata[group_field].notna().to_numpy()
        if not assigned.all():
            n_unassigned = int((~assigned).sum())
            notes.append(
                f"{n_unassigned}/{len(assigned)} cell(s) have no value in "
                f"obs['{group_field}'] and are excluded from the per-group statistics; "
                "they are unassigned by the partition, not a group of their own"
            )
            scores = scores.loc[assigned]
            metadata = metadata.loc[assigned]
            if scores.empty:
                return self._skip(f"no cell has a value in obs['{group_field}']")

        # ---- Statistics. ----
        # The sample is the unit condition was assigned to, so it is what the mixed
        # model needs as its lowest random level. Without it ``lmm_effect_sizes``
        # reconstructs donor x condition, which is the same thing for a cohort with one
        # library per donor per arm and conservative for any other.
        effect_sample_col = sample_col if sample_col and sample_col in metadata.columns else None
        effects = lmm_effect_sizes(
            scores,
            metadata,
            donor_col=donor_col,
            condition_col=condition_col,
            group_col=group_field,
            sample_col=effect_sample_col,
            case=case,
            control=control,
            fdr_method=str(config.get("fdr_method", "fdr_bh")),
            min_donors_per_arm=int(config.get("min_donors_per_arm", 2)),
            seed=seed,
        )

        # The index gets the same test its own modules got. A panel of paired
        # violins with no test on it asks the reader to call an overlap a shift, and
        # deriving the answer from the module rows is not possible: the index is a
        # z-scored combination, so its effect is not any module's effect.
        index_effects = pd.DataFrame()
        if index_values is not None:
            index_effects = lmm_effect_sizes(
                pd.DataFrame({index_key: index_values.loc[metadata.index]}, index=metadata.index),
                metadata,
                donor_col=donor_col,
                condition_col=condition_col,
                group_col=group_field,
                sample_col=effect_sample_col,
                case=case,
                control=control,
                fdr_method=str(config.get("fdr_method", "fdr_bh")),
                min_donors_per_arm=int(config.get("min_donors_per_arm", 2)),
                seed=seed,
            )

        permanova = pd.DataFrame()
        if sample_col and sample_col in adata.obs.columns:
            permanova = permanova_by_group(
                scores,
                metadata,
                sample_col=sample_col,
                condition_col=condition_col,
                group_col=group_field,
                case=case,
                control=control,
                n_permutations=int(config.get("n_permutations", 999)),
                fdr_method=str(config.get("fdr_method", "fdr_bh")),
                seed=seed,
            )
        else:
            warnings.append(
                f"PERMANOVA skipped: sample_col '{sample_col}' unresolved or absent from obs"
            )

        # Two tables, not one. The matrix is the shape a heatmap wants and the shape
        # the reference published; the long-form tests are the ones that can be cited.
        # ``scores.corr()`` correlates cells, and 2,000 cells from 9 donors are not
        # 2,000 observations — so the matrix is written without any significance
        # attached to it, and the significance lives beside it at the donor level.
        correlation_method = str(config.get("correlation_method", "spearman"))
        correlation = scores.corr(method=correlation_method)
        correlation.index.name = "program"

        unit_col = sample_col if sample_col and sample_col in metadata.columns else None
        if unit_col is None:
            warnings.append(
                f"program correlations tested at the cell level: sample_col '{sample_col}' "
                "unresolved or absent from obs, so no independent unit could be formed. "
                "Cells within a donor are not independent; read module_correlation_tests.csv "
                "as descriptive and its p-values as anticonservative"
            )
        try:
            correlation_tests = program_correlation_tests(
                scores,
                metadata,
                sample_col=unit_col,
                condition_col=condition_col,
                method=correlation_method,
                fdr_method=str(config.get("fdr_method", "fdr_bh")),
                program_genes=self._program_genes(adata, config),
            )
        except ValueError as exc:
            # A sample straddling both arms is the case this catches. It is a design
            # fact about the cohort, not a reason to lose the stage's other four
            # tables, so the correlation is retried without the adjustment and the
            # reason is surfaced rather than swallowed.
            warnings.append(f"condition-adjusted program correlation unavailable: {exc}")
            correlation_tests = program_correlation_tests(
                scores,
                metadata,
                sample_col=unit_col,
                method=correlation_method,
                fdr_method=str(config.get("fdr_method", "fdr_bh")),
                program_genes=self._program_genes(adata, config),
            )

        # ---- One canonical axis order, applied to every table and panel. ----
        # ``lmm_effect_sizes`` returns groups in order of appearance, which on the
        # LEC arm was 1, 3, 6, 2, 4, 8, 5, 7 — and the dot-grid takes its columns
        # from the table, so that arbitrary order became the panel's column order.
        # Ordering the tables (rather than only the figures) keeps the CSV a
        # faithful replot source and gives the four panels one shared axis.
        group_order = _resolve_group_order(metadata[group_field], config.get("group_order") or [])
        program_order, _bands = mr_figures.ordered_programs(
            list(scores.columns), config.get("module_categories") or {}
        )
        effects = _apply_label_order(effects, column="group", order=group_order)
        effects = _apply_label_order(effects, column="program", order=program_order)
        permanova = _apply_label_order(permanova, column="group", order=group_order)
        index_effects = _apply_label_order(index_effects, column="group", order=group_order)
        correlation = correlation.reindex(index=program_order, columns=program_order)
        # b then a, relying on the sort being stable: the result is program_a-major with
        # program_b in the same order inside each block, matching the matrix's axes.
        correlation_tests = _apply_label_order(
            correlation_tests, column="program_b", order=program_order
        )
        correlation_tests = _apply_label_order(
            correlation_tests, column="program_a", order=program_order
        )

        n_untestable = int((~np.isfinite(effects["effect"])).sum())
        if n_untestable:
            # A blank cell in the flagship figure means "not testable", which is
            # not the same claim as "no effect" — say so where it is counted.
            warnings.append(
                f"{n_untestable}/{len(effects)} module x group cell(s) could not be tested; "
                "they are drawn as crosses, not as null effects"
            )
        n_fallback = int((effects["method"] != "lmm").sum())
        if n_fallback:
            notes.append(
                f"{n_fallback}/{len(effects)} cell(s) used a donor-level fallback rather than "
                "the mixed model (see the 'method' and 'reason' columns)"
            )

        # ---- Depth audit. ----
        # Run on the whole lineage rather than per group, deliberately. The first leg of
        # the audit is "does depth differ between the arms", which is a property of the
        # cohort and not of the partition, and it is the leg that gates the other two: on
        # depth-balanced arms no metric can be depth-driven however hard it tracks depth.
        # Splitting by group would divide the donors again, drop most groups below
        # ``min_pairs``, and answer the cohort question separately in each subset with less
        # power — so a program the lineage-level audit contradicts is contradicted for
        # every group's row on that program, and the note below says so.
        depth_col = _resolve_depth_column(metadata, config.get("depth_col"))
        depth_audit = pd.DataFrame()
        n_depth_driven = 0
        n_depth_attenuated = 0
        n_depth_masked = 0
        if depth_col is None:
            configured = config.get("depth_col")
            warnings.append(
                f"depth audit skipped: configured depth_col '{configured}' is absent from obs"
                if configured
                else "depth audit skipped: obs carries none of "
                f"{', '.join(_DEPTH_COLUMN_CANDIDATES)}; run scanpy's calculate_qc_metrics, or "
                "name a depth column with depth_col, and the module effects will be audited "
                "against it. Until then nothing here separates a condition effect from a "
                "library-size effect"
            )
        else:
            # The index is audited alongside the programs because it is the quantity the
            # manuscript sentence is usually stated on, and being a z-scored combination of
            # depth-sensitive scores does not make it less depth-sensitive.
            audit_metrics = scores.copy()
            if index_values is not None:
                audit_metrics[index_key] = index_values.loc[metadata.index].to_numpy(dtype=float)
            depth_audit = depth_confound_audit(
                audit_metrics,
                metadata,
                donor_col=donor_col,
                condition_col=condition_col,
                case=case,
                control=control,
                depth_col=depth_col,
                fdr_method=str(config.get("fdr_method", "fdr_bh")),
                min_pairs=int(config.get("min_depth_pairs") or MIN_PAIRED_BLOCKS),
            )
            depth_audit = _apply_label_order(
                depth_audit, column="metric", order=[*program_order, index_key]
            )
            verdicts = depth_audit["verdict"].astype(str)
            contradicted = depth_audit.loc[verdicts == _DEPTH_CONTRADICTED, "metric"].astype(str)
            weakened = depth_audit.loc[verdicts == _DEPTH_WEAKENED, "metric"].astype(str)
            masked = depth_audit.loc[verdicts == _DEPTH_MASKED, "metric"].astype(str)
            n_depth_driven = int(len(contradicted))
            n_depth_attenuated = int(len(weakened))
            n_depth_masked = int(len(masked))
            if n_depth_driven:
                warnings.append(
                    f"depth audit: {n_depth_driven} program(s) did not survive adjustment for "
                    f"obs['{depth_col}'] — {', '.join(contradicted)}. Their condition effect is a "
                    "library-depth readout; do not report the corresponding rows of "
                    "module_effect_sizes_lmm.csv, in any group, without the adjusted result "
                    "beside them (module_depth_audit.csv)"
                )
            if n_depth_attenuated:
                notes.append(
                    f"depth audit: {n_depth_attenuated} program(s) kept their sign and most of "
                    f"their size but lost significance under adjustment for obs['{depth_col}'] — "
                    f"{', '.join(weakened)}; reportable with that caveat stated"
                )
            if n_depth_masked:
                # The one verdict that hands back a result. Worth a note of its own because a
                # reader who has been told the audit protects claims will not look for the row
                # where it produced one, and the caveat travels with it either way.
                notes.append(
                    f"depth audit: {n_depth_masked} program(s) have no unadjusted condition "
                    f"effect and a significant one after adjustment for obs['{depth_col}'] — "
                    f"{', '.join(masked)}. Depth was moving them against their condition effect; "
                    "treat as leads, not findings, since the unadjusted test is the one this "
                    "table's FDR family was corrected over"
                )
            if not n_depth_driven and not n_depth_attenuated:
                # The informative negative. Saying which of the two reasons it holds for
                # matters: a balanced cohort is safe here and the same metric may not be in
                # the next one, whereas a robust metric survived the adjustment itself.
                balanced = int((verdicts == "depth_balanced").sum())
                notes.append(
                    f"depth audit: no program's condition effect is explained by "
                    f"obs['{depth_col}'] ({balanced}/{len(depth_audit)} row(s) on "
                    "depth-balanced arms, the rest tested and robust)"
                )

        # ---- Tables (the replot source). ----
        # Both namespaces get a stage subdirectory: a run that scores eleven
        # modules writes five tables and four figures, and dropping them loose
        # into results/ and figures/ makes the run directory unreadable.
        writer = StageArtifactWriter.from_context(context, default_subdir=_STAGE_DIR)
        artifacts = [
            writer.table(
                effects,
                "module_effect_sizes_lmm.csv",
                name="mr_effect_sizes",
                description=(
                    f"Module x {group_field} condition effects ({case} vs {control}): "
                    "mixed-model effect, CI, p, FDR."
                ),
                index=False,
            ),
            writer.table(
                correlation.reset_index(),
                "module_correlation.csv",
                name="mr_correlation",
                description=(
                    "Program x program correlation across cells (descriptive; the "
                    "testable version is module_correlation_tests.csv)."
                ),
                index=False,
            ),
            writer.table(
                correlation_tests,
                "module_correlation_tests.csv",
                name="mr_correlation_tests",
                description=(
                    "Program pair correlations at the sample level with FDR, the "
                    "condition-adjusted partial correlation, and the shared-gene count."
                ),
                index=False,
            ),
        ]
        if not depth_audit.empty:
            artifacts.append(
                writer.table(
                    depth_audit,
                    "module_depth_audit.csv",
                    name="mr_depth_audit",
                    description=(
                        f"Does each program's {case} vs {control} shift survive adjustment for "
                        f"obs['{depth_col}']: coupling rho, raw and adjusted effect, verdict."
                    ),
                    index=False,
                )
            )
        if subtype_table is not None:
            artifacts.append(
                writer.table(
                    subtype_table,
                    "module_subtypes.csv",
                    name="mr_subtypes",
                    description=("Cluster-level signature labels with the top-vs-second margin."),
                    index=False,
                )
            )
        if not permanova.empty:
            artifacts.append(
                writer.table(
                    permanova,
                    "module_permanova.csv",
                    name="mr_permanova",
                    description=f"Multivariate condition effect per {group_field} (PERMANOVA).",
                    index=False,
                )
            )
        if index_values is not None:
            # Built from ``metadata``, not from ``adata.obs`` + ``group_values``:
            # those are full-length and this frame has to line up with the cells the
            # statistics actually used, or the index figure describes a different
            # cohort from the effect table beside it.
            index_frame = pd.DataFrame(
                {
                    "cell": metadata.index,
                    group_field: metadata[group_field].to_numpy(),
                    condition_col: metadata[condition_col].astype(str).to_numpy(),
                    index_key: metadata[index_key].to_numpy(dtype=float),
                }
            )
            # ``plot_contrast_index`` reads its violin order off this frame, so the
            # index panel and the dot-grid share one column order instead of two.
            index_frame = _apply_label_order(
                index_frame, column=group_field, order=group_order
            ).reset_index(drop=True)
            artifacts.append(
                writer.table(
                    index_frame,
                    "module_contrast_index.csv",
                    name="mr_contrast_index",
                    description="Per-cell signed contrast index with group and condition.",
                    index=False,
                )
            )
            artifacts.append(
                writer.table(
                    index_effects,
                    "module_index_effects.csv",
                    name="mr_index_effects",
                    description=(
                        f"Condition effect on {index_key} per {group_field} "
                        f"({case} vs {control}): mixed-model effect, CI, p, FDR."
                    ),
                    index=False,
                )
            )

        # ---- Figures. ----
        figures_root = getattr(getattr(context, "paths", None), "figures", None)
        figures_dir = Path(figures_root) / _STAGE_DIR if figures_root is not None else None
        artifacts.extend(
            self._draw(
                effects=effects,
                permanova=permanova,
                correlation_tests=correlation_tests,
                index_frame=index_frame,
                index_effects=index_effects,
                group_field=group_field,
                condition_col=condition_col,
                index_key=index_key,
                config=config,
                case=case,
                control=control,
                out_dir=figures_dir,
                warnings=warnings,
            )
        )

        n_significant = int((effects["fdr"] < 0.05).sum())
        # The depth verdict belongs on the headline, not only in the table: a run summary
        # that reports significant programs and says nothing about the audit invites the
        # reader to quote the first number without ever reaching the second.
        headline = (
            f"module remodeling: {scores.shape[1]} program(s) x "
            f"{effects['group'].nunique()} group(s), {n_significant} significant at FDR<0.05"
        )
        if depth_audit.empty:
            headline += "; depth unaudited"
        elif n_depth_driven:
            headline += f"; {n_depth_driven} program(s) depth-driven"
        else:
            headline += "; depth-audited, none depth-driven"
        if n_depth_masked:
            headline += f", {n_depth_masked} depth-masked"
        notes.insert(0, headline + ".")

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            warnings=warnings,
            metrics={
                "case": case,
                "control": control,
                "n_programs": int(scores.shape[1]),
                "n_groups": int(effects["group"].nunique()),
                "n_significant": n_significant,
                "n_untestable": n_untestable,
                "n_fallback": n_fallback,
                "depth_col": depth_col,
                "n_depth_driven": n_depth_driven,
                "n_depth_attenuated": n_depth_attenuated,
                "n_depth_masked": n_depth_masked,
                "n_unassigned_clusters": int((subtype_table["label"] == UNASSIGNED).sum())
                if subtype_table is not None
                else 0,
                "seed": seed,
            },
            backend="python",
        )

    # ------------------------------------------------------------------ #
    # figures                                                             #
    # ------------------------------------------------------------------ #
    def _draw(
        self,
        *,
        effects: pd.DataFrame,
        permanova: pd.DataFrame,
        correlation_tests: pd.DataFrame,
        index_frame: pd.DataFrame | None,
        index_effects: pd.DataFrame,
        group_field: str,
        condition_col: str,
        index_key: str,
        config: dict,
        case: str,
        control: str,
        out_dir: Path | None,
        warnings: list[str],
    ) -> list[StageArtifact]:
        """Draw the panel suite; each figure fails independently."""
        if out_dir is None:
            warnings.append("figures skipped: no figures directory on the stage context")
            return []

        produced: list[tuple[str, str, list]] = []
        # Display wording, resolved once for the whole suite so the four panels
        # cannot label the same axis two different ways. The fallback is the column
        # name opened up, never an invented name.
        group_label = config.get("group_label") or group_field.replace("_", " ")
        program_labels = config.get("program_labels") or {}

        flagship: list = []
        render_figure(
            "module dot-grid",
            lambda: mr_figures.plot_module_dotgrid(
                effects,
                out_dir=out_dir,
                categories=config.get("module_categories") or {},
                group_label=group_label,
                program_labels=program_labels,
                case=case,
                control=control,
                effect_cap=config.get("effect_cap"),
                max_dot_fdr_exponent=float(config.get("max_dot_fdr_exponent", 6.0)),
            ),
            figures=flagship,
            warnings=warnings,
        )
        produced.append(
            (
                "mr_dotgrid",
                f"Module x {group_field} dot-grid: colour = {case} − {control} mixed-model "
                "effect, size = −log10(FDR).",
                flagship,
            )
        )

        if index_frame is not None:
            # Annotated from the table beside it, so a star on the panel and a row in
            # the CSV are the same number.
            significance = {
                str(row.group): float(row.fdr)
                for row in index_effects.itertuples()
                if np.isfinite(row.fdr)
            }
            paths: list = []
            render_figure(
                "contrast index",
                lambda: mr_figures.plot_contrast_index(
                    index_frame,
                    out_dir=out_dir,
                    group_col=group_field,
                    condition_col=condition_col,
                    index_col=index_key,
                    group_label=group_label,
                    index_label=config.get("index_label") or index_key.replace("_", " "),
                    significance=significance,
                    case=case,
                    control=control,
                ),
                figures=paths,
                warnings=warnings,
            )
            produced.append(
                ("mr_index_figure", "Signed contrast index by group and condition.", paths)
            )

        if not permanova.empty:
            paths = []
            render_figure(
                "PERMANOVA R²",
                lambda: mr_figures.plot_permanova(
                    permanova, out_dir=out_dir, group_label=group_label
                ),
                figures=paths,
                warnings=warnings,
            )
            produced.append(
                ("mr_permanova_figure", "Multivariate condition effect (R²) per group.", paths)
            )

        # The correlation panels are drawn from the *tested* table, not from the
        # descriptive per-cell matrix. A heatmap of ``scores.corr()`` shows a
        # coefficient with no unit, no significance and no disclosure of shared
        # genes, and it is the panel most likely to be read as evidence — so the
        # matrix stays a CSV and the figure carries what the CSV cannot.
        alpha = float(config.get("alpha", 0.05))
        program_order = list(dict.fromkeys(correlation_tests["program_a"].astype(str))) + list(
            dict.fromkeys(correlation_tests["program_b"].astype(str))
        )
        program_order = list(dict.fromkeys(program_order))
        paths = []
        render_figure(
            "program correlation",
            lambda: figstyle.save_figure(
                program_correlation_heatmap(
                    correlation_tests,
                    program_labels=program_labels,
                    program_order=program_order,
                    alpha=alpha,
                ),
                out_dir,
                "module_correlation",
            ),
            figures=paths,
            warnings=warnings,
        )
        produced.append(
            (
                "mr_correlation_figure",
                "Program-pair correlation: measured below the diagonal, "
                "condition-adjusted above, significance and shared genes in-cell.",
                paths,
            )
        )

        paths = []
        render_figure(
            "program correlation adjustment",
            lambda: figstyle.save_figure(
                program_correlation_slopes(
                    correlation_tests,
                    program_labels=program_labels,
                    alpha=alpha,
                    max_pairs=config.get("max_correlation_pairs"),
                ),
                out_dir,
                "module_correlation_adjustment",
            ),
            figures=paths,
            warnings=warnings,
        )
        produced.append(
            (
                "mr_correlation_adjustment_figure",
                "Each pair's coefficient before and after the condition is removed: "
                "a segment collapsing to zero is a correlation the condition explains.",
                paths,
            )
        )

        return [
            StageArtifact(name=name, path=path, kind="figure", description=description)
            for name, description, paths in produced
            for path in paths
        ]


__all__ = ["ModuleRemodelingMethod"]
