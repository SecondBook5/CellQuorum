"""Which clusters are populations and which are the library's debris wearing a cluster's name?

Every atlas grows a cluster that is not a cell state. Ambient RNA, stripped nuclei and
low-complexity debris do not distribute themselves evenly: they collect, they get their own
Leiden id, and downstream they are counted as cells. A composition figure then reports one
library's soup as a condition-specific population, and a differential test run inside the
affected lineages carries its genes.

Such a cluster is normally found by eye — someone notices a blob in the middle of the UMAP,
looks at it, and hardcodes its id into a mask. This module exists because that practice fails
in a specific and expensive way, and because the criteria behind the eyeballing are perfectly
statable.

**A debris cluster must be re-identified by criteria, never carried across partitions by id.**
Leiden numbering is a property of one clustering run, not of the cells. Re-cluster the same
data after changing QC, integration or resolution and every id is reassigned. A mask of
``{"18", "30", "40"}`` written against one partition, applied to the next, deletes whatever
now happens to be numbered 18 and leaves the actual debris in — two errors at once, both
silent, and the second one worse than doing nothing because the mask's presence reads as
having handled it. :func:`verify_declared_debris` refuses that move.

**No single signal identifies debris; the conjunction does.** Each of the individual marks is
routinely a real population:

- *Low complexity* on its own is a genuinely low-RNA cell type. Neutrophils, cornified
  keratinocytes and erythrocytes all sit at a fraction of an atlas's median gene count and are
  not debris. Deleting on complexity alone deletes real lineages, and it deletes the ones
  least likely to be missed.
- *Annotation-confidence collapse* on its own is a doublet cluster or a genuine transitional
  state, both of which one may want to keep and neither of which is ambient.
- *Lineage promiscuity* on its own is a cluster the annotation cut differently from the
  clustering, which is a labelling question rather than a data-quality one.
- *Single-library dominance* on its own is a donor-private population, which in a small cohort
  is common and sometimes is the finding.

What ambient debris is, mechanistically, is a shallow mixture of everything that was in the
droplet suspension. So it is simultaneously shallow, unassignable, and composed of every
lineage at once — and that conjunction is what :data:`AMBIENT_DEBRIS` requires. In the atlas
this module was calibrated on, the conjunction picked out exactly one cluster of thirty-nine,
while complexity alone picked out six, five of them real.

**A cluster that is almost entirely one condition is either the finding or the artifact, and
this audit cannot tell you which.** Condition dominance is therefore reported and deliberately
excluded from the verdict: an audit that downgraded clusters for being disease-specific would
delete the result. What disambiguates is whether the cluster is also one *library* — a
population that is 93% cases across nine donors is biology; one that is 93% cases because 77%
of it came from a single case library is that library.

**Embedding position is reported and not scored.** The blob in the middle of the UMAP is the
usual way debris gets noticed, but the embedding is downstream of the same PCA that the debris
distorts, so its position is a symptom of the artifact and of the pipeline's response to it in
unknown proportion. It goes in the table as description. It does not enter the verdict.

The verdicts are leads, not deletions. The columns are the evidence, and adjudication —
whether to mask, to exclude a library outright, or to accept the cluster — is a decision about
the study that a table cannot make.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A cluster median gene count below this fraction of the atlas median counts as shallow.
#: Set well below one because the point is to separate debris from the low-RNA tail of real
#: biology, not to rank clusters by depth: cornified keratinocytes land near 0.37 of the
#: atlas median and are a population.
DEFAULT_COLLAPSE_RATIO = 0.5

#: A cluster whose median annotation confidence falls below this has no lineage the
#: annotator will commit to. Real clusters sit near one; the interesting range is narrow and
#: the floor is deliberately permissive so that only a collapse trips it.
DEFAULT_CONFIDENCE_FLOOR = 0.7

#: Normalised Shannon entropy of a cluster's lineage composition, above which the cluster is
#: a mixture rather than a population. Normalised by ``log(n_lineages)`` over the whole
#: partition so the scale is comparable between atlases with different numbers of lineages.
DEFAULT_PROMISCUITY_ENTROPY = 0.25

#: Fraction of a cluster coming from one library, above which the cluster is that library's
#: property rather than the cohort's. A majority from one library cannot be a shared cell
#: state, whatever else is true of it.
DEFAULT_DOMINANCE_FRACTION = 0.5

#: Below this many cells a cluster's medians are not estimates of anything and the audit
#: declines rather than guesses.
DEFAULT_MIN_CELLS = 20

#: The verdict for the ambient-mixture conjunction: shallow, unassignable and every lineage
#: at once. The only verdict this module treats as debris by default.
AMBIENT_DEBRIS = "ambient_debris"

#: One library's oddity: it dominates the cluster *and* the cluster is marked some other way.
#: Distinct from :data:`AMBIENT_DEBRIS` because the remedy differs — dropping a library is a
#: cohort decision, masking a cluster is not.
LIBRARY_ARTIFACT = "library_artifact"

#: Verdicts a caller should treat as "do not count these cells as cells" unless it has
#: adjudicated otherwise. Deliberately short: everything else in the vocabulary is a lead.
DEFAULT_DEBRIS_VERDICTS = (AMBIENT_DEBRIS, LIBRARY_ARTIFACT)

#: The full verdict vocabulary, in the order :func:`cluster_artifact_audit` tests it. First
#: match wins, so the more specific conjunctions precede the single marks.
VERDICTS = (
    "insufficient_cells",
    AMBIENT_DEBRIS,
    LIBRARY_ARTIFACT,
    "ambiguous_annotation",
    "low_complexity",
    "promiscuous",
    "low_confidence",
    "single_library",
    "clean",
)

#: Columns of :func:`cluster_artifact_audit`, in order.
AUDIT_COLUMNS = [
    "cluster",
    "n_cells",
    "frac_of_atlas",
    "median_complexity",
    "complexity_ratio",
    "complexity_collapsed",
    "median_confidence",
    "confidence_collapsed",
    "n_lineages",
    "lineage_coverage",
    "modal_lineage",
    "modal_lineage_frac",
    "lineage_entropy",
    "lineage_promiscuous",
    "dominant_library",
    "dominant_library_frac",
    "library_dominated",
    "dominant_condition",
    "dominant_condition_frac",
    "embedding_radius",
    "embedding_radius_rank_pct",
    "n_marks",
    "marks",
    "verdict",
]

#: Columns of :func:`verify_declared_debris`.
DECLARED_COLUMNS = [
    "cluster",
    "declared",
    "audited",
    "verdict",
    "n_cells",
    "agrees",
]


def _as_labels(values: pd.Series | None, n: int) -> pd.Series | None:
    """A label vector as strings with the engine's absent markers as NA, or None."""
    if values is None:
        return None
    labels = pd.Series(np.asarray(values), dtype="object")
    if len(labels) != n:
        raise ValueError(f"label vector has {len(labels)} entries, expected {n}")
    labels = labels.where(~pd.isna(labels), other=None)
    labels = labels.map(lambda v: None if v is None else str(v))
    return labels.replace({"": None, "nan": None, "NaN": None, "None": None}).reset_index(drop=True)


def _normalised_entropy(counts: pd.Series, n_categories: int) -> float:
    """Shannon entropy of a composition, divided by the maximum a partition-wide mixture has.

    Dividing by ``log(n_categories)`` over the whole partition rather than over the cluster's
    own categories is what makes the number comparable between clusters: a cluster split
    evenly between two lineages and one split evenly between thirteen both have entropy one
    under self-normalisation, which is the opposite of the distinction wanted here.
    """
    if n_categories < 2 or counts.sum() == 0:
        return 0.0
    p = (counts / counts.sum()).to_numpy(dtype=float)
    p = p[p > 0]
    # ``max`` rather than the bare expression because a pure cluster's entropy is -0.0, which
    # prints as "-0.000" in a table and reads like a bug in a figure caption.
    return max(0.0, float(-(p * np.log(p)).sum() / np.log(n_categories)))


def _dominant(labels: pd.Series | None) -> tuple[str, float]:
    """The modal label and its share, or ``("", nan)`` when there is nothing to count."""
    if labels is None:
        return "", float("nan")
    counts = labels.dropna().value_counts()
    if counts.empty:
        return "", float("nan")
    return str(counts.index[0]), float(counts.iloc[0] / counts.sum())


def _verdict(
    *,
    enough_cells: bool,
    shallow: bool,
    unconfident: bool,
    promiscuous: bool,
    dominated: bool,
) -> str:
    """Map the marks onto :data:`VERDICTS`, most specific conjunction first.

    The ordering carries the argument of the module docstring. ``ambient_debris`` needs all
    three intrinsic marks because each alone is a real population. ``library_artifact`` needs
    dominance *plus* something else, because dominance alone is a donor-private population and
    in a nine-donor cohort that is unremarkable. Everything below those two is a single mark
    and therefore a lead: a name for what to go and look at, not a reason to delete cells.
    """
    if not enough_cells:
        return "insufficient_cells"
    if shallow and unconfident and promiscuous:
        return AMBIENT_DEBRIS
    if dominated and (shallow or unconfident or promiscuous):
        return LIBRARY_ARTIFACT
    if unconfident and promiscuous:
        return "ambiguous_annotation"
    if shallow:
        return "low_complexity"
    if promiscuous:
        return "promiscuous"
    if unconfident:
        return "low_confidence"
    if dominated:
        return "single_library"
    return "clean"


def cluster_artifact_audit(
    cluster_labels: pd.Series,
    *,
    complexity: pd.Series,
    confidence: pd.Series | None = None,
    lineage: pd.Series | None = None,
    library: pd.Series | None = None,
    condition: pd.Series | None = None,
    embedding: np.ndarray | None = None,
    collapse_ratio: float = DEFAULT_COLLAPSE_RATIO,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    promiscuity_entropy: float = DEFAULT_PROMISCUITY_ENTROPY,
    dominance_fraction: float = DEFAULT_DOMINANCE_FRACTION,
    min_cells: int = DEFAULT_MIN_CELLS,
) -> pd.DataFrame:
    """Score every cluster of a partition for the marks of a data artifact.

    Args:
        cluster_labels: Per-cell cluster id, one entry per cell. Cells with an absent label
            are dropped, since an unlabelled cell is not a cluster.
        complexity: Per-cell depth proxy, aligned to ``cluster_labels`` —
            ``n_genes_by_counts`` is the intended column. Genes detected is preferred over
            total counts because it saturates more slowly and is less sensitive to a handful
            of very high-expression transcripts.
        confidence: Per-cell annotation confidence, if the annotation produced one. Without
            it no cluster can be marked unconfident and therefore none can reach
            :data:`AMBIENT_DEBRIS`; the audit says so in ``marks`` rather than silently
            downgrading every verdict.
        lineage: Per-cell coarse cell-type label, for the promiscuity mark.
        library: Per-cell library / sample id, for the dominance mark. This is the sequencing
            library, not the donor: a donor with two libraries has two chances to produce
            debris and they are separate artifacts.
        condition: Per-cell condition, reported and never scored (see module docstring).
        embedding: ``(n_cells, n_dims)`` embedding, reported and never scored. Radius is
            measured from the coordinate-wise median rather than the mean so one distant
            cluster does not move the origin.
        collapse_ratio: See :data:`DEFAULT_COLLAPSE_RATIO`.
        confidence_floor: See :data:`DEFAULT_CONFIDENCE_FLOOR`.
        promiscuity_entropy: See :data:`DEFAULT_PROMISCUITY_ENTROPY`.
        dominance_fraction: See :data:`DEFAULT_DOMINANCE_FRACTION`.
        min_cells: See :data:`DEFAULT_MIN_CELLS`.

    Returns:
        One row per cluster with :data:`AUDIT_COLUMNS`, sorted by ``verdict`` severity then
        ascending ``complexity_ratio`` so the clusters worth looking at read first. Every
        threshold decision is exposed twice — as the measured value and as the boolean it
        produced — so a reader can move a threshold without re-running anything. ``marks`` is
        a semicolon-joined list of the marks that tripped, and is the column to quote: it is
        why, where ``verdict`` is only what.

    Raises:
        ValueError: If a vector's length does not match ``cluster_labels``, or if no cell
            carries a usable cluster label.
    """
    n = len(cluster_labels)
    clusters = _as_labels(cluster_labels, n)
    assert clusters is not None
    depth = pd.Series(np.asarray(complexity, dtype=float)).reset_index(drop=True)
    if len(depth) != n:
        raise ValueError(f"complexity has {len(depth)} entries, expected {n}")

    conf = None
    if confidence is not None:
        conf = pd.Series(np.asarray(confidence, dtype=float)).reset_index(drop=True)
        if len(conf) != n:
            raise ValueError(f"confidence has {len(conf)} entries, expected {n}")

    lineages = _as_labels(lineage, n)
    libraries = _as_labels(library, n)
    conditions = _as_labels(condition, n)

    radius = None
    if embedding is not None:
        coords = np.asarray(embedding, dtype=float)
        if coords.shape[0] != n:
            raise ValueError(f"embedding has {coords.shape[0]} rows, expected {n}")
        radius = pd.Series(np.linalg.norm(coords - np.median(coords, axis=0), axis=1)).reset_index(
            drop=True
        )

    keep = clusters.notna().to_numpy()
    if not keep.any():
        raise ValueError("no cell carries a usable cluster label")

    frame = pd.DataFrame({"cluster": clusters, "complexity": depth})
    if conf is not None:
        frame["confidence"] = conf
    if lineages is not None:
        frame["lineage"] = lineages
    if libraries is not None:
        frame["library"] = libraries
    if conditions is not None:
        frame["condition"] = conditions
    if radius is not None:
        frame["radius"] = radius
    frame = frame.loc[keep].reset_index(drop=True)

    atlas_median = float(frame["complexity"].median())
    n_lineages_total = int(frame["lineage"].dropna().nunique()) if lineages is not None else 0
    total_cells = int(len(frame))

    rows = []
    for cluster, block in frame.groupby("cluster", sort=True):
        n_cells = int(len(block))
        median_complexity = float(block["complexity"].median())
        ratio = median_complexity / atlas_median if atlas_median > 0 else float("nan")

        median_conf = float(block["confidence"].median()) if conf is not None else float("nan")
        if lineages is not None:
            counts = block["lineage"].dropna().value_counts()
            n_lin = int(len(counts))
            entropy = _normalised_entropy(counts, n_lineages_total)
            modal_lineage, modal_frac = _dominant(block["lineage"])
        else:
            n_lin, entropy, modal_lineage, modal_frac = 0, float("nan"), "", float("nan")

        top_library, library_frac = _dominant(block.get("library"))
        top_condition, condition_frac = _dominant(block.get("condition"))

        enough = n_cells >= min_cells
        shallow = bool(np.isfinite(ratio) and ratio < collapse_ratio)
        unconfident = bool(
            conf is not None and np.isfinite(median_conf) and median_conf < confidence_floor
        )
        promiscuous = bool(np.isfinite(entropy) and entropy > promiscuity_entropy)
        dominated = bool(np.isfinite(library_frac) and library_frac > dominance_fraction)

        marks = []
        if shallow:
            marks.append(f"shallow({ratio:.2f}x)")
        if unconfident:
            marks.append(f"unconfident({median_conf:.2f})")
        if promiscuous:
            marks.append(f"promiscuous({n_lin} lineages, H={entropy:.2f})")
        if dominated:
            marks.append(f"one library({top_library} {library_frac:.0%})")
        if conf is None:
            marks.append("no confidence column: unconfident could not be tested")
        if lineages is None:
            marks.append("no lineage column: promiscuous could not be tested")

        rows.append(
            {
                "cluster": cluster,
                "n_cells": n_cells,
                "frac_of_atlas": n_cells / total_cells,
                "median_complexity": median_complexity,
                "complexity_ratio": ratio,
                "complexity_collapsed": shallow,
                "median_confidence": median_conf,
                "confidence_collapsed": unconfident,
                "n_lineages": n_lin,
                "lineage_coverage": (
                    (n_lin / n_lineages_total) if n_lineages_total else float("nan")
                ),
                "modal_lineage": modal_lineage,
                "modal_lineage_frac": modal_frac,
                "lineage_entropy": entropy,
                "lineage_promiscuous": promiscuous,
                "dominant_library": top_library,
                "dominant_library_frac": library_frac,
                "library_dominated": dominated,
                "dominant_condition": top_condition,
                "dominant_condition_frac": condition_frac,
                "embedding_radius": (
                    float(block["radius"].median()) if radius is not None else float("nan")
                ),
                "embedding_radius_rank_pct": float("nan"),
                "n_marks": int(shallow) + int(unconfident) + int(promiscuous) + int(dominated),
                "marks": "; ".join(marks) if marks else "none",
                "verdict": _verdict(
                    enough_cells=enough,
                    shallow=shallow,
                    unconfident=unconfident,
                    promiscuous=promiscuous,
                    dominated=dominated,
                ),
            }
        )

    table = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    if radius is not None and len(table):
        table["embedding_radius_rank_pct"] = table["embedding_radius"].rank(pct=True)

    severity = {name: rank for rank, name in enumerate(VERDICTS)}
    table["_severity"] = table["verdict"].map(severity).fillna(len(VERDICTS))
    table = table.sort_values(
        ["_severity", "complexity_ratio"], ascending=[True, True], kind="stable"
    )
    return table.drop(columns="_severity").reset_index(drop=True)


def debris_clusters(
    audit: pd.DataFrame, *, verdicts: tuple[str, ...] = DEFAULT_DEBRIS_VERDICTS
) -> list[str]:
    """The cluster ids the audit calls debris, for a caller that wants to mask them.

    This is the function a driver should call instead of writing a literal set of ids. The
    difference is not stylistic: an id set is fixed at the moment it is typed and is wrong the
    next time the data is clustered, while this is recomputed from the partition in hand and
    cannot be stale.

    Args:
        audit: Output of :func:`cluster_artifact_audit`.
        verdicts: Which verdicts count as debris. The default is deliberately the two
            conjunctions; widening it to include the single-mark leads will remove real
            low-RNA lineages.

    Returns:
        Cluster ids in the audit's own order, i.e. worst first.
    """
    return [str(c) for c in audit.loc[audit["verdict"].isin(verdicts), "cluster"]]


def verify_declared_debris(
    audit: pd.DataFrame,
    declared: list[str] | tuple[str, ...] | set[str],
    *,
    verdicts: tuple[str, ...] = DEFAULT_DEBRIS_VERDICTS,
    strict: bool = True,
) -> pd.DataFrame:
    """Check a hand-written debris mask against the partition it is about to be applied to.

    For the case a mask already exists — reproducing a previous run, or honouring a curator's
    decision — and has to be shown to still mean what it meant. Three disagreements are
    possible and they are not equally visible:

    - A declared cluster **absent from this partition**. The mask was written against a
      different clustering. This is the one that must stop the run: continuing masks nothing,
      or worse, masks whatever inherited the number.
    - A declared cluster the audit calls **clean**. The mask deletes real cells. It may still
      be right — a curator can know something the columns do not — but it has to be a decision
      rather than an inheritance.
    - An audited debris cluster **not declared**. The artifact is still in the object, and the
      presence of a mask makes it look handled.

    Args:
        audit: Output of :func:`cluster_artifact_audit`.
        declared: Cluster ids the caller intends to mask.
        verdicts: Which verdicts count as debris, as in :func:`debris_clusters`.
        strict: Raise when a declared cluster is absent from the partition. Turn it off only
            to inspect the comparison frame, never to proceed past the error.

    Returns:
        One row per cluster mentioned by either side, with :data:`DECLARED_COLUMNS`.
        ``agrees`` is False on every row that needs a decision. Absent clusters appear with
        ``n_cells`` zero and verdict ``"absent"``.

    Raises:
        ValueError: Under ``strict``, if any declared cluster is not a cluster of this
            partition.
    """
    present = {str(c) for c in audit["cluster"]}
    wanted = {str(c) for c in declared}
    audited = set(debris_clusters(audit, verdicts=verdicts))

    missing = sorted(wanted - present)
    if missing and strict:
        raise ValueError(
            "declared debris clusters are not clusters of this partition: "
            f"{', '.join(missing)}. Leiden ids belong to one clustering run, so a mask "
            "written against another partition deletes whichever cells inherited the number "
            "and leaves the real artifact in. Re-identify the debris with "
            "cluster_artifact_audit on this object instead of carrying the ids across."
        )

    sizes = audit.set_index(audit["cluster"].astype(str))["n_cells"].to_dict()
    verdict_by_cluster = audit.set_index(audit["cluster"].astype(str))["verdict"].to_dict()

    rows = [
        {
            "cluster": cluster,
            "declared": cluster in wanted,
            "audited": cluster in audited,
            "verdict": verdict_by_cluster.get(cluster, "absent"),
            "n_cells": int(sizes.get(cluster, 0)),
            "agrees": (cluster in wanted) == (cluster in audited) and cluster in present,
        }
        for cluster in sorted(wanted | audited)
    ]
    return pd.DataFrame(rows, columns=DECLARED_COLUMNS)


__all__ = [
    "AMBIENT_DEBRIS",
    "AUDIT_COLUMNS",
    "DECLARED_COLUMNS",
    "DEFAULT_COLLAPSE_RATIO",
    "DEFAULT_CONFIDENCE_FLOOR",
    "DEFAULT_DEBRIS_VERDICTS",
    "DEFAULT_DOMINANCE_FRACTION",
    "DEFAULT_MIN_CELLS",
    "DEFAULT_PROMISCUITY_ENTROPY",
    "LIBRARY_ARTIFACT",
    "VERDICTS",
    "cluster_artifact_audit",
    "debris_clusters",
    "verify_declared_debris",
]
