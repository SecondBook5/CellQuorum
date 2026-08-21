"""Curated cell-state gene programs and the program-resolution helper.

``STATE_PROGRAMS`` holds a small set of well-established, organism-agnostic-in-
spirit (but symbol-wise **human**) cell-state signatures. They are a convenience
default, never a study assumption: the scoring methods gate every program on how
many of its genes are actually present, so a program whose symbols do not match
the dataset skips rather than scoring noise.

``resolve_programs`` merges, in order, the curated programs, any user ``programs``
dict, named ``config.markers`` panels, and a user ``.gmt`` — a later source
overrides an earlier one on name collision — into one ``name -> [gene, ...]`` map.
"""

from __future__ import annotations

from pathlib import Path

# Curated human cell-state signatures. Compact, canonical marker sets — not
# exhaustive gene sets. Cell-cycle is deliberately absent: it is scored upstream
# by the QC stage (Tirosh S/G2M) and duplicating it here would double-write.
STATE_PROGRAMS: dict[str, list[str]] = {
    # Heat-shock / proteotoxic stress.
    "stress_hsp": [
        "HSPA1A",
        "HSPA1B",
        "HSPA6",
        "HSPB1",
        "HSPH1",
        "DNAJB1",
        "DNAJA1",
        "HSPD1",
        "HSP90AA1",
        "HSP90AB1",
        "BAG3",
    ],
    # Hypoxia / HIF response (Hallmark-hypoxia core).
    "hypoxia_hif": [
        "VEGFA",
        "SLC2A1",
        "CA9",
        "LDHA",
        "PGK1",
        "HK2",
        "BNIP3",
        "NDRG1",
        "ENO1",
        "PDK1",
        "ALDOA",
        "ADM",
        "HILPDA",
    ],
    # Type-I interferon-stimulated genes.
    "interferon_isg": [
        "ISG15",
        "IFI6",
        "IFI27",
        "IFI44",
        "IFI44L",
        "IFIT1",
        "IFIT3",
        "MX1",
        "MX2",
        "OAS1",
        "OAS2",
        "OAS3",
        "RSAD2",
        "STAT1",
        "IRF7",
        "XAF1",
        "BST2",
    ],
    # Senescence + senescence-associated secretory phenotype.
    "senescence_sasp": [
        "CDKN1A",
        "CDKN2A",
        "SERPINE1",
        "GLB1",
        "IL6",
        "IL1B",
        "CXCL8",
        "CXCL1",
        "CCL2",
        "MMP3",
        "MMP1",
        "IGFBP3",
        "IGFBP7",
        "TNFRSF10C",
    ],
    # Fibrosis / ECM / myofibroblast.
    "fibrosis_ecm": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL5A1",
        "FN1",
        "ACTA2",
        "TAGLN",
        "POSTN",
        "FBN1",
        "LOX",
        "TIMP1",
        "SPARC",
        "BGN",
        "DCN",
        "THBS1",
    ],
}


def read_gmt(gmt_path: str) -> dict[str, list[str]]:
    """Parse a ``.gmt`` file into ``name -> [gene, ...]``.

    A GMT line is ``set_name<TAB>description<TAB>gene1<TAB>gene2<...>``. The
    description column is discarded; blank lines are ignored. Duplicate genes
    within a set are de-duplicated while preserving first-seen order.

    Args:
        gmt_path: Path to a ``.gmt`` file.

    Returns:
        Mapping of gene-set name to its ordered, de-duplicated gene list.

    Raises:
        FileNotFoundError: If the path does not exist.
    """

    programs: dict[str, list[str]] = {}
    text = Path(gmt_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        fields = [f.strip() for f in line.split("\t") if f.strip()]
        # Need at least a name and one gene (description column is optional/dropped).
        if len(fields) < 3:
            continue
        name, genes = fields[0], fields[2:]
        # De-duplicate while preserving order.
        seen: dict[str, None] = {}
        for gene in genes:
            seen.setdefault(gene, None)
        programs[name] = list(seen)
    return programs


def resolve_programs(config: dict, context: object | None = None) -> dict[str, list[str]]:
    """Merge every configured program source into one ``name -> [gene, ...]`` map.

    Sources are applied in order so a later one overrides an earlier one on a
    name collision: curated ``STATE_PROGRAMS`` → user ``programs`` → named
    ``config.markers`` panels → user ``.gmt``.

    Args:
        config: Resolved state-scoring config sub-block.
        context: Pipeline context, used only to resolve ``marker_panels`` against
            ``context.config.markers`` (skipped when unavailable).

    Returns:
        The merged program map (may be empty when no source contributes).
    """

    programs: dict[str, list[str]] = {}

    # 1. Curated defaults (optionally restricted to a named subset).
    if config.get("use_builtin_programs", True):
        subset = config.get("builtin_programs") or list(STATE_PROGRAMS)
        for name in subset:
            if name in STATE_PROGRAMS:
                programs[name] = list(STATE_PROGRAMS[name])

    # 2. User-supplied program gene lists.
    for name, genes in (config.get("programs") or {}).items():
        programs[name] = list(genes)

    # 3. Named marker panels from the project's markers block.
    panel_names = config.get("marker_panels") or []
    if panel_names and context is not None:
        markers = getattr(getattr(context, "config", None), "markers", None)
        if markers is not None:
            for panel_name in panel_names:
                # markers.panel() fails loud on an unknown panel name.
                programs[panel_name] = list(markers.panel(panel_name))

    # 4. User .gmt gene sets.
    gmt_path = config.get("gmt_path")
    if gmt_path:
        for name, genes in read_gmt(gmt_path).items():
            programs[name] = list(genes)

    return programs


__all__ = ["STATE_PROGRAMS", "read_gmt", "resolve_programs"]
