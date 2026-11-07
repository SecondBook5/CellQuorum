# Consolidation audit

Date: 2026-11-06. Scope: `src/cellquorum` (114,000 lines, 2,156 functions, 17 stage packages). Method: AST scan for
function length, `pylint` duplicate-code scan (12 or more similar lines, comments, docstrings and imports ignored),
`vulture` dead-code scan, and pattern counts. This is a measurement, not a change; nothing here alters behaviour.

The approved direction (2 October) stands: keep modules consolidated, keep the internal stage executor and the
Snakemake workflow, one theme function. This note says where consolidation is still worth doing and where it is not.

## What the numbers say

| Measure | Result | Reading |
|---|---|---|
| Dead code (`vulture`, 80% confidence) | 0 findings | Nothing certain to be unused. The 251 findings at 60% are registered stage and method hooks that static analysis cannot see. |
| Duplicated code | 19 groups, about 700 duplicated lines (0.6% of the code) | The code is not copy-paste heavy. The duplication is concentrated in one place (below). |
| Functions over 200 lines | 45 (was 37 on 3 October) | The main structural debt. It grows as stages are added. |
| Functions over 100 lines | 206 of 2,156 | |
| Largest files | `visualization/qc/panels.py` 2,698 lines; `figstyle.py` 1,771; `cell_cell_communication/viz/_plots.py` 1,654 | |
| Largest functions | `scarches._run` 616; `remodeling_method._run` 525; `plot_mnn_sample_activity` 495; `pyscenic_method._run` 392 | |

## Where the duplication is

Eight external-environment backends (`metacells`, `pyscenic`, `pyucell`, `celloracle`, `scclr`, `partipy`, `sccoda`,
`hdwgcna`) each carry the same 35 to 80 line skeleton: check the launcher is on the path, build
`micromamba run -n <env> python <script> ...`, run it under a timeout, probe a module inside the environment. The
largest pairs repeat 79, 77 and 54 lines. `backends/r.py` and `backends/rscript.py` share 47 lines. Two stage pairs
(`adjudication` and `population_identity`; `propeller` and `proportion_ttest`) share about 30 lines each.

One change removes most of it: an `ExternalEnvBackend` base class holding the launcher check, command build, timeout
and module probe, with each backend supplying its environment name and script. About 400 lines go, and a fix to
launcher handling or timeouts is made once. Risk is low because every backend already has tests.

## Ranked recommendations

1. **`ExternalEnvBackend` base class** for the eight backends. Highest value, lowest risk. One commit per backend.
2. **Break up the 45 functions over 200 lines**, worst first, as phases inside the same module (the approach already
   used for `subclustering`: 628 to 131 lines). Start with `scarches._run` (needs a settings object first),
   `remodeling_method._run`, `plot_mnn_sample_activity`. Behaviour must not change, so each needs its existing tests
   green and a figure or output comparison before and after.
3. **One figure-output config.** Ten stage configs declare `figure_formats` and 24 declare `dpi`. A shared
   `FigureOutputConfig` fragment removes the repetition and makes "all figures at 300 dpi, PNG and PDF" a single
   setting.
4. **Figure plumbing.** 108 `plt.subplots` calls and 12 direct `savefig` calls beside 22 `save_cellquorum_figure`
   calls. Route every save through `save_cellquorum_figure` so companion formats, fonts and metadata cannot differ.
   Splitting `visualization/qc/panels.py` by panel family would make it reviewable.
5. **Stage skeleton.** 38 `*_method.py` files define a `_run` with the same opening (resolve config, check
   availability, skip with a reason, write artifacts). A small set of helpers in `methods/stage_base.py` would shorten
   each; do this last, after 1 to 4, because it touches every stage.

## What not to consolidate

Stage packages stay separate. They are the unit users reason about, and merging them would trade duplication for a
harder-to-read tree. The 17 stage directories and 29 stage classes match the 43 configurable stages.

## Defects this audit found along the way

These are separate from consolidation and are tracked here so they are not lost:

- A config that names some stages still runs every stage it does not name (33 of 43 default to on). The run header now
  says so. Whether the default should change is a design decision.
- The QC archetype audit runs for up to its 900-second limit and is abandoned on a CPU; on a 15,000-cell run that was 15
  of 24 minutes. Candidates: off by default, or a much smaller cap.
- `seurat_v3` highly variable gene selection can fail with a loess singularity on a small cell subset. See the
  feature-selection note in the changelog once fixed.
- The `cellquorum-core` recipe does not pin `scverse-misc`, so a future re-lock could pick a release that breaks
  `mudata`, as the GPU re-lock did.
