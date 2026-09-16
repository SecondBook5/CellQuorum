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

1. `bootstrap` → `ambient_correction` (soupx) → `qc_evidence` → `preprocessing` →
   `feature_selection` → `dimensionality` → `integration` (scVI/Harmony).
   Mechanically the same as before; the difference this time is actually
   looking at each stage's output.
2. `clustering` → `annotation` (marker genes + CellTypist).
3. **One reference-mapping pass, not two** — `label_key: Cell_type_granular`,
   `coarse_label_key: Cell_type` (both confirmed present in
   `/mnt/e/atopic_dermatitis_singlecell_atlas/data/Broad/SCP2613/anndata/scp.h5ad`;
   granular has 86 categories, verbatim match to what's currently in
   `cell_type_granular` — e.g. "Fibroblast SLPI+", "DC MMP9+CCR7+"),
   `key_added: cell_type_granular`, same `reference_filters:
   [{column: disease_lesional, keep: [Healthy, Non-lesional]}]` and seeds
   `[0,1,2]` as the original run. `reference_mapping` trains scArches once on
   the fine (`Cell_type_granular`) column and *deterministically collapses* it
   to `cell_type_granular_coarse` via a fine→coarse hierarchy it derives from
   the atlas itself (`_label_hierarchy` in `scarches.py`) — refusing to run if
   that hierarchy is ambiguous. This is the engine's existing, correct answer
   to "map twice and reconcile": don't map twice, map once and read the same
   prediction at two depths, so the two readouts cannot contradict each other.
   Replaces both the untracked, unrecoverable granular script *and* the
   original separate `ref_state` mapping pass with one tracked run.
   (Note: the atlas also has a `Cell_type_granular_v2` variant with abbreviated
   names — using the non-`_v2` column since it matches the existing object;
   flag if `_v2` was actually intended.)
4. `annotation_consensus` — `method_label_keys: [cell_type_markers,
   cell_type_celltypist, cell_type_granular_coarse]` (the reference-mapping
   vote is now the coarse collapse, not a separately-trained `ref_state`),
   `granular_source_key: null` (do **not** let consensus overwrite
   `cell_type_granular` with a copy of the coarse vote — it already holds the
   real fine-grained prediction from step 3), same `backbone_aliases` as
   before → produces `cell_type`, `annotation_confidence`, `needs_review`.
   Because all three votes (2 direct classifiers + 1 hierarchy-collapsed
   reference-mapping vote) go through this one existing mechanism,
   `needs_review`/`annotation_confidence` already *is* the three-way
   agreement check — no new alias-table script needed. Lineage manifests
   should gate inclusion on `needs_review == False` (or
   `annotation_confidence == "high"`) going forward, not a bespoke
   `require_agreement` column pair.
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

- Every stage's config, including the reference-mapping pass, is present in
  `provenance/resolved_config.json` — nothing about how the atlas was built
  is unrecoverable, unlike today.
- `cell_type_granular` and its coarse collapse `cell_type_granular_coarse`
  cannot disagree with each other (enforced by construction via
  `coarse_label_key`), and `cell_type`'s consensus already incorporates that
  coarse vote alongside marker-genes and CellTypist — so `needs_review` is a
  real three-way check, not a two-way one, with no separate script required.
- The new object lives in its own project directory with a real manifest,
  not inside the engine repo.
