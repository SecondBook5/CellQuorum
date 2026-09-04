"""Prior-knowledge (gene-set / net) loading — the only place sources are named.

Wraps decoupler's dedicated OmniPath accessors (``dc.op.hallmark``,
``dc.op.collectri``, ...), the MSigDB table for the pathway collections that have no
dedicated accessor (Reactome, KEGG, GO, WikiPathways, ...), the generic
``dc.op.resource`` fetch for anything else, and ``.gmt`` loading for user gene sets. Any
fetch or import failure is normalized to :class:`PriorFetchError` so a method can record a
clean skip instead of crashing.

One naming asymmetry to know about, because it will otherwise surprise someone who unions
two collections into one figure: ``dc.op.hallmark`` strips the ``HALLMARK_`` prefix from
its set names, so hallmark arrives as ``TNFA_SIGNALING_VIA_NFKB`` while the MSigDB-backed
collections keep MSigDB's official identifiers (``REACTOME_HEMOSTASIS``, ``GOBP_...``).
Nothing here re-strips them: the official identifier is what a reader pastes back into
MSigDB to check a set, and silently rewriting 1,787 Reactome names to look like hallmark's
would trade that for nine characters of axis label.
"""

from __future__ import annotations

import pandas as pd

#: MSigDB collections, keyed by the name a config uses. decoupler ships a dedicated
#: accessor for hallmark only; every other MSigDB collection lives in one large table with
#: a ``collection`` column, so they are fetched from there and filtered.
#:
#: These used to be routed through ``dc.op.resource`` under names like ``"Reactome"``, and
#: that never worked: ``dc.op.resource`` serves OmniPath's *annotation* resources, whose
#: list contains no Reactome and no GO_Biological_Process at all. Every request for one
#: raised, the method recorded a skipped collection, and a run configured for
#: ``[hallmark, reactome]`` quietly produced hallmark only.
MSIGDB_COLLECTIONS = {
    "reactome": "reactome_pathways",
    "kegg": "kegg_pathways",
    "kegg_medicus": "kegg_medicus_pathways",
    "go": "go_biological_process",
    "go_bp": "go_biological_process",
    "go_mf": "go_molecular_function",
    "go_cc": "go_cellular_component",
    "wikipathways": "wikipathways",
    "biocarta": "biocarta_pathways",
    "pid": "pid_pathways",
    "oncogenic": "oncogenic_signatures",
    "immunesigdb": "immunesigdb",
    "cell_type_signatures": "cell_type_signatures",
    "positional": "positional",
}

#: The whole MSigDB table, per (organism, license), held for the life of the process.
#:
#: The fetch is ~36 s and ~5.9 M rows over the wire with no on-disk cache in decoupler, and
#: a pipeline asking for three MSigDB collections would otherwise pay it three times. Held
#: as categoricals, which is what makes this affordable at all: 1,228 MB as object dtype
#: versus 62 MB as categories, for the identical frame.
_MSIGDB_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


class PriorFetchError(Exception):
    """Raised when a gene-set / net cannot be fetched or parsed."""


def _msigdb_table(organism: str, license: str) -> pd.DataFrame:
    """The full MSigDB long table, fetched at most once per process per organism."""
    import decoupler as dc

    key = (organism, license)
    cached = _MSIGDB_CACHE.get(key)
    if cached is None:
        table = dc.op.resource("MSigDB", organism=organism, license=license)
        cached = table.astype({column: "category" for column in table.columns})
        _MSIGDB_CACHE[key] = cached
    return cached


def _msigdb_collection(collection: str, organism: str, license: str) -> pd.DataFrame:
    """One MSigDB collection as a ``source``/``target`` net."""
    table = _msigdb_table(organism, license)
    subset = table[table["collection"] == collection]
    if subset.empty:
        available = sorted(map(str, table["collection"].cat.categories))
        raise PriorFetchError(f"MSigDB has no collection '{collection}'; available: {available}")
    net = subset[["geneset", "genesymbol"]].astype(str)
    net = net.rename(columns={"geneset": "source", "genesymbol": "target"})
    return net.drop_duplicates(["source", "target"]).reset_index(drop=True)


def get_net(
    collection: str,
    organism: str = "human",
    gmt_path: str | None = None,
    license: str = "academic",
) -> pd.DataFrame:
    """Return a long-format ``source/target[/weight]`` net for a collection.

    Args:
        collection: A key like ``"hallmark"``, ``"reactome"``, ``"go_bp"``,
            ``"collectri"``, ``"progeny"``, ``"dorothea"``, any other key in
            :data:`MSIGDB_COLLECTIONS`, or an arbitrary OmniPath annotation resource
            name resolved via ``dc.op.resource``. Ignored when ``gmt_path`` is set.
        organism: Species passed through to decoupler (``config.organism``).
        gmt_path: If set, load a user ``.gmt`` instead of fetching.
        license: decoupler license mode.

    Returns:
        Long-format net DataFrame.

    Raises:
        PriorFetchError: On import failure, fetch failure, or parse failure.
    """

    try:
        import decoupler as dc
    except Exception as exc:  # ImportError or a shimmed-None module
        raise PriorFetchError(f"decoupler unavailable: {exc}") from exc
    if dc is None:
        raise PriorFetchError("decoupler unavailable")

    try:
        if gmt_path is not None:
            return dc.pp.read_gmt(gmt_path)

        key = collection.lower()
        if key == "hallmark":
            return dc.op.hallmark(organism=organism, license=license, verbose=False)
        if key == "collectri":
            return dc.op.collectri(organism=organism, license=license, verbose=False)
        if key == "progeny":
            return dc.op.progeny(organism=organism, top=500, license=license, verbose=False)
        if key == "dorothea":
            return dc.op.dorothea(
                organism=organism, levels=["A", "B", "C"], license=license, verbose=False
            )
        if key in MSIGDB_COLLECTIONS:
            return _msigdb_collection(MSIGDB_COLLECTIONS[key], organism, license)
        return dc.op.resource(collection, organism=organism, license=license)
    except PriorFetchError:
        raise
    except Exception as exc:
        raise PriorFetchError(f"failed to fetch '{collection}': {exc}") from exc


__all__ = ["MSIGDB_COLLECTIONS", "PriorFetchError", "get_net"]
