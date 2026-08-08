"""Prior-knowledge (gene-set / net) loading — the only place sources are named.

Wraps decoupler's OmniPath accessors (``dc.op.*``), the generic
``dc.op.resource`` fetch for collections without a dedicated accessor
(Reactome, KEGG, GO, ...), and ``.gmt`` loading for user gene sets. Any fetch
or import failure is normalized to ``PriorFetchError`` so a method can record a
clean skip instead of crashing.
"""

from __future__ import annotations

import pandas as pd

# Collections with a dedicated dc.op accessor. Everything else is fetched via
# dc.op.resource(<omnipath name>). The mapping is config-facing only — no study
# assumptions, just decoupler's own resource names.
_DEDICATED = {"hallmark", "collectri", "progeny", "dorothea"}
_RESOURCE_NAMES = {
    "reactome": "Reactome",
    "kegg": "KEGG",
    "go_bp": "GO_Biological_Process",
    "go": "GO_Biological_Process",
}


class PriorFetchError(Exception):
    """Raised when a gene-set / net cannot be fetched or parsed."""


def get_net(
    collection: str,
    organism: str = "human",
    gmt_path: str | None = None,
    license: str = "academic",
) -> pd.DataFrame:
    """Return a long-format ``source/target[/weight]`` net for a collection.

    Args:
        collection: A key like ``"hallmark"``, ``"reactome"``, ``"collectri"``,
            ``"progeny"``, ``"dorothea"``, or an arbitrary name resolved via
            ``dc.op.resource``. Ignored when ``gmt_path`` is set.
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
        name = _RESOURCE_NAMES.get(key, collection)
        return dc.op.resource(name, organism=organism, license=license)
    except PriorFetchError:
        raise
    except Exception as exc:
        raise PriorFetchError(f"failed to fetch '{collection}': {exc}") from exc


__all__ = ["PriorFetchError", "get_net"]
