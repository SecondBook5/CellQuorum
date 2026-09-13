# Rebuilding the lymphedema global atlas as a standalone, tracked project

## Problem

The current global atlas (`cellquorum/runs/le_global/objects/le_global_shared.h5ad`,
handed to every lineage project as their shared input) carries three independent
per-cell classifications that were never fully reconciled with each other:

- `cell_type` — consensus of marker-vote + CellTypist + `ref_state`, via the
  `annotation_consensus` stage (tracked, in `provenance/resolved_config.json`).
- `ref_state` — a coarse scArches reference-mapping pass against an
  atopic-dermatitis atlas (`label_key: Cell_type`), tracked in the same config.
- `cell_type_granular` — a finer reference-mapping pass, same style of output
  (confidence/entropy/agreement columns identical in shape to `ref_state`'s), but
  **produced by a script that no longer exists anywhere in the run directory or
  the engine repo.** No config, no notebook, nothing. It cannot be reproduced or
  audited.

Because `require_agreement` (used by every lineage manifest to build its slice)
only ever checks `cell_type` against `ref_state`, cells where the untracked
granular pass disagreed with the coarse consensus were never caught. Concretely,
in the fibroblast slice: 731/63,663 cells (1.15%) carry non-fibroblast granular
labels (DC, macrophage, T cell, mast, pericyte, SMC, keratinocyte, endothelial,
Schwann, etc.) despite both coarse systems agreeing "Fibroblasts." That
contamination then surfaces downstream as visual noise in every figure that
colors by `cell_type_granular` (RNA velocity legend, embeddings).

This was found while debugging the fibroblast pipeline run and is not
lineage-specific — every current and future lineage project inherits it.

## Decision

Rebuild the global atlas from raw CellRanger counts, as its own standalone
project, running one stage at a time with real inspection of each stage's
output before advancing — not a blind full run, and not a patch on top of the
existing (partly unauditable) object.

Scope is deliberately narrow: **this covers only the global atlas project.**
Restructuring the four lineage projects (`le_fibroblasts`, `le_lec`,
`le_keratinocytes`, `le_vec`) to match, or defining a shared parent-directory
convention across all five, is explicitly deferred to a later pass.

## Project layout

New standalone directory, sibling to the lineage projects, same convention:

```
/home/ajbook/projects/le_global/
  manifests/
    global.yaml              # the pipeline manifest (stages, params, cohort ref)
    le_global_cohort.csv      # moved from cellquorum/configs/manifests/ — real
                              # experimental design (9 patients, paired
                              # Normal/Lymphedema, real CellRanger paths) —
                              # content unchanged, only relocated
  template.yaml
  scripts/
    run.sh
  configs/                   # gen-configs output
  runs/
    <run_name>/
      objects/ figures/ results/ provenance/ reports/ logs/ scratch/
```

The existing `cellquorum/runs/le_global/` (with its accumulated `da_step6`,
`de_step7`, `figures_organized`, `qc_recheck_fixed10/15`, `mechano_screen`
folders) is left exactly as-is — historical record, not deleted, not migrated.
The new build produces a separate, clean run under the new project. Nothing
downstream repoints at the new object until it's verified.

## Build plan

Raw input: `/mnt/e/lymphedema_cellranger` (9 sample-set directories, confirmed
present) + the relocated cohort CSV (9 patients, paired Normal/Lymphedema,
`include=TRUE` rows as before — no redesign of the experimental design itself).

Run stage-by-stage (`--from-stage X --until-stage X`, or the narrowest window
that makes sense per stage), inspecting real output — a table, a figure, a
row count — before extending to the next stage. Order:

1. `bootstrap` → `ambient_correction` (soupx) → `qc` → `preprocessing` →
   `feature_selection` → `dimensionality` → `integration` (scVI/Harmony).
   Mechanically the same as before; the difference this time is actually
   looking at each stage's output.
2. `clustering` → `annotation` (marker genes + CellTypist) → `reference_mapping`
   (scArches vs the atopic-dermatitis atlas, `label_key: Cell_type` → `ref_state`,
   same config as the original: `atlas_h5ad:
   /mnt/e/atopic_dermatitis_singlecell_atlas/data/Broad/SCP2613/anndata/scp.h5ad`,
   `reference_filters: [{column: disease_lesional, keep: [Healthy, Non-lesional]}]`,
   seeds `[0,1,2]`) → `annotation_consensus` (marker + CellTypist + `ref_state` →
   `cell_type`, `annotation_confidence`, `needs_review` — same
   `backbone_aliases` as before).
3. **New, tracked stage**: a second reference-mapping pass, same atlas file,
   same reference filters, but `label_key: Cell_type_granular` (confirmed
   present in the atlas, 86 categories, verbatim match to what's currently in
   `cell_type_granular` — e.g. "Fibroblast SLPI+", "DC MMP9+CCR7+") →
   `key_added: cell_type_granular`. This replaces the lost, unrecoverable
   script with a real, reproducible, provenance-tracked run.
   (Note: the atlas also has a `Cell_type_granular_v2` variant with abbreviated
   names — using the non-`_v2` column since it matches the existing object;
   flag if `_v2` was actually intended.)
4. **New cross-check**: before the object is treated as final, verify
   `cell_type` / `ref_state` / `cell_type_granular` agree, using an explicit
   alias table (granular value → coarse category, e.g. `"Fibroblast SLPI+" →
   "Fibroblasts"`) in the same style as `annotation_consensus`'s existing
   `backbone_aliases`. Cells where the granular call disagrees with the coarse
   consensus get flagged (and excluded from lineage slices), the same
   `needs_review` mechanism extended to cover the third classifier instead of
   just two.
5. `adjudication` → `population_identity` → final object.

## Verification approach

Per stage: run it, then actually look — row/cell counts match expectations,
a spot-check figure renders sensibly, no silent stage skips. Move to the next
stage only after confirming the current one's output is right. No multi-hour
blind run followed by post-hoc forensics, which is what produced this problem
in the first place.

## Out of scope

- Restructuring `le_fibroblasts` / `le_lec` / `le_keratinocytes` / `le_vec` to
  match this convention — later, separate pass.
- A shared parent directory or naming scheme across all five projects — later.
- Re-deriving the fibroblast/LEC/keratinocyte/VEC slices from the new global
  object — happens only after this rebuild is verified end-to-end.
- Fixing the other confirmed bugs found this session (dead GSEA renderer, ORA
  dotplot collision, ccc_network float node IDs, near-zero differential Ricci
  curvature) — independent of this rebuild, tracked separately.

## Success criteria

- Every stage's config, including the granular reference-mapping pass, is
  present in `provenance/resolved_config.json` — nothing about how the atlas
  was built is unrecoverable, unlike today.
- `cell_type` / `ref_state` / `cell_type_granular` agreement is checked and
  enforced once, at the global level, before any lineage ever sees the object.
- The new object lives in its own project directory with a real manifest,
  not inside the engine repo.
