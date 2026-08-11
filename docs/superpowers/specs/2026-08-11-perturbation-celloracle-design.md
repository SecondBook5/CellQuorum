# Perturbation Stage (in-silico KO via CellOracle) — Design

**Date:** 2026-08-11
**Status:** approved (brainstorm; standing delegation)
**Checklist item:** item 4, remaining half — in-silico KO / perturbation (the HARD MANDATE for the lymphedema case study). GRN co-expression (hdWGCNA `coexpression`) and regulon inference (`grn`/pySCENIC) already shipped.
**Generalize-from source:** none — CellOracle is a fresh integration (no reusable KO code exists in AJ's repos; crrt-ude's perturbation is model-based counterfactual on a trained field, not a GRN simulator).

## Goal

Add a generalizable `perturbation` stage to the CellQuorum engine that performs
**in-silico transcription-factor knockouts with CellOracle**: infer a
simulation-ready gene-regulatory network from observational scRNA + a built-in
promoter base GRN, simulate each KO by zeroing the TF and propagating the signal,
and rank knockouts by how strongly they shift disease cells toward the healthy
state — running end-to-end from one command and skipping cleanly (never crashing)
when its isolated environment is absent.

The flagship deliverable is a **ranked therapeutic-target table**: "knock out TF X
→ strongest disease→healthy shift," with a supporting KO shift-field UMAP.

## Non-goals

- **Not** scGen / GEARS / Perturb-seq methods — the lymphedema study is observational
  case/control with **no perturbation dataset**; those need perturbation-labeled
  training data. CellOracle fits on observational scRNA + a motif base GRN, which is
  exactly this constraint.
- **No UDE-based perturbation method here.** A novel UDE (universal differential
  equation) perturbation method is a separate research track with its own spec/plan.
  This stage carves a clean extension seam for it (a second `method` under the same
  stage/config/artifact contract) but does not implement it.
- **No PerturbODE here.** Deferred with the UDE research work (data-requirement
  unresolved; not the mandate-satisfier).
- **No dependency on SEACells metacells.** CellOracle performs its own kNN imputation
  internally and its validated workflow runs at cell level; metacell aggregation in
  front of it is non-standard and could interfere with its imputation / cell-count
  assumptions. SEACells stays deferred and decoupled (it would more naturally pair
  with a future pySCENIC-on-metacells variant, not this stage).
- **No consumption of the pySCENIC `grn` output.** CellOracle builds its own GRN from
  the promoter base prior; the two GRN stages stand as **independent convergent
  evidence** (pySCENIC = descriptive regulons; CellOracle = simulation-ready GRN), not
  a pipe. No cross-stage coupling.
- **No scATAC-derived custom base GRN.** No ATAC data for the lymphedema study; use
  CellOracle's built-in promoter base GRN (hg38 / mm10), selected by config `organism`.
- **No study-specific biology in `src`.** Disease/healthy is a generic config-supplied
  label (`condition_key` / `healthy_label`), never hardcoded; TF targets come from the
  fitted GRN, not a curated lymphedema list.

## Engine invariants this stage must honor

(Carried from the `grn`/pySCENIC stage — the direct precedent.)

1. **Zero study-specific biology in `src`.** Generic obs-key fallbacks only
   (`cluster_key`: config → `cell_type` → `leiden` → `"all"`; `rep_key`: config →
   `X_pca` → `X_pca_harmony`; `embedding_key`: config → `X_umap`). No hardcoded gene
   lists, no "lymphedema"/"disease" strings, no organism baked into code.
2. **Skip-not-crash.** Every failure path returns `MethodSkip` (never raises out of
   `_run`). The in-env script writes a sentinel + `sys.exit(0)` on a harmless skip and
   exits non-zero only on a *real* CLI failure (which the method converts to `MethodSkip`).
   Absence of `condition_key`/`healthy_label` is **not** a skip — the stage degrades to
   direction-agnostic outputs.
3. **Determinism.** Seeded throughout (kNN imputation, signal propagation, any sampling).
4. **Dual-format figures** via `figstyle.save_figure` (PNG+PDF, 300 dpi, `pdf.fonttype=42`),
   palette from `figstyle.CATEGORICAL_PALETTE`.
5. **TDD**, one quartet + backend + 4 wiring seams, mirroring `grn`.
6. **Isolated-env subprocess backend** (`micromamba run -n <env> …`), `shutil.which`
   launcher probe, `subprocess.run(check=False)`.

## Architecture

Mirrors the `grn`/pySCENIC stage one-for-one, substituting the CellOracle CLI script
for the pySCENIC one.

### Backend — `src/cellquorum/backends/celloracle_backend.py`

`CellOracleBackend(BaseBackend)`, modeled on `PyscenicBackend`:

- `name = "celloracle"`, `kind = "external"`, `env_name = "celloracle_env"`,
  `launcher = "micromamba"`, `script_timeout_seconds = 10800` (GRN fit + screen is slow),
  `timeout_seconds = 60` (availability check).
- `run_script(script_path, args, *, timeout=None)` →
  `micromamba run -n celloracle_env python <script> [args...]`, `check=False`,
  `capture_output=True`, `text=True`. Raises `FileNotFoundError` if launcher or script
  missing. (Identical control flow to `PyscenicBackend.run_script`.)
- `status()` → available iff launcher on PATH **and** `celloracle` module importable
  in-env (`_py_module_available`, regex-guarded module name, mirrors pySCENIC).
- Requirements: `micromamba` (executable) + `celloracle` (other), each with an
  `install_hint` (frozen env; built-in base GRNs bundled with CellOracle).
- `build_celloracle_backend(*, env_name="celloracle_env", launcher="micromamba", timeout_seconds=60)`.
- Module exports the bundled in-env script path: `CELLORACLE_KO_PY`.

### Bundled in-env script — `src/cellquorum/backends/celloracle_scripts/celloracle_ko.py`

Runs **inside** `celloracle_env`. Reads the h5ad directly (like pySCENIC's h5py reader
if the frozen env's AnnData is older; otherwise standard `anndata.read_h5ad` —
implementer confirms against the frozen env at build time). Runs CellOracle's standard
three-phase workflow:

**Args:** `--h5ad`, `--out-dir`, `--tag`, `--organism` (default `human`), `--cluster-key`,
`--rep-key`, `--embedding-key`, `--condition-key` (optional), `--healthy-label` (optional),
`--tf-list` (optional space/comma list; absent → systematic screen), `--n-top-targets`,
`--knn-n-neighbors`, `--n-propagation`, `--seed`.

1. **GRN inference.** Load built-in promoter base GRN for `--organism`; `oracle.import_anndata_as_raw_count`
   (or the CellOracle-current equivalent), `oracle.perform_PCA()`, `oracle.knn_imputation(seed)`;
   fit cluster-specific GRNs (`Links`) on `--cluster-key`. Outputs `base_grn.parquet`,
   `links_<cluster>.parquet` (one per cluster), `grn_summary.csv` (per-cluster top regulators).
2. **KO simulation.** TF set = `--tf-list` if given, else every TF present in both the
   fitted GRN and `adata.var`. For each TF: `oracle.simulate_shift(perturb_condition={TF: 0.0}, n_propagation)`,
   then transition-probability shift + per-cell shift vector on `--embedding-key`. Outputs
   `shift_vectors_<TF>.parquet` (cells × 2, indexed by obs_names), one per-TF summary row.
3. **Scoring & ranking.** If `--condition-key` **and** `--healthy-label` present: compute
   the disease→healthy axis as the centroid difference (diseased-cell centroid → healthy-cell
   centroid) in the embedding; each TF's directional score = mean projection of its per-cell
   shift vectors onto that unit axis; rank TFs descending. Else: score = mean shift magnitude
   per TF (direction-agnostic). Outputs `perturbation_ranking.csv` (**flagship**: columns
   `tf`, `score`, `n_cells`, `direction` ∈ {directional, magnitude}).

Graceful-skip (missing deps / no base GRN / empty GRN fit → empty schemas +
`perturbation_SKIPPED_{tag}.txt` + exit 0); fail-loud (real CellOracle failure →
`perturbation_FAILED_{tag}.txt` + persistent log + non-zero).

### Quartet — `src/cellquorum/perturbation/`

**`config.py`** — `PerturbationConfig(StrictBaseModel)`:
```
enabled: bool = True
method: str = "celloracle"
layer: str = "counts"
organism: str = "human"              # built-in base GRN (human/mouse)
cluster_key: str | None = None       # GRN cluster grouping; falls back cell_type -> leiden -> "all"
embedding_key: str | None = None     # shift-vector space; falls back X_umap
rep_key: str | None = None           # PCA/kNN rep; falls back X_pca -> X_pca_harmony
condition_key: str | None = None     # disease->healthy axis; absent -> direction-agnostic
healthy_label: str | None = None     # target condition value; absent -> skip directional
tf_list: list[str] | None = None     # None -> systematic screen of all fitted TFs
n_top_targets: int = 20              # ranked-table / figure cutoff
knn_n_neighbors: int = 200
n_propagation: int = 3
min_cells_total: int = 200
seed: int = 0
env_name: str = "celloracle_env"
launcher: str = "micromamba"
timeout_seconds: int = 10800
```

**`celloracle_method.py`** — `CellOracleMethod(AnalysisMethod)`:
- `name = "celloracle"`, `stage_category = "perturbation"`, `backend = "celloracle"`.
- `input_contract`: `required_layers=[layer]`, `required_obs=[]` (`condition_key`/`cluster_key`
  fall back — do **not** hard-require, same lesson as grn/hdWGCNA), `expected_kind="counts"`.
- `requires_obs` → `[]`.
- `_run` orchestration:
  1. Resolve config; resolve `cluster_key` (config → `cell_type` → `leiden` → `"all"`),
     `rep_key` (config → `X_pca` → `X_pca_harmony`), `embedding_key` (config → `X_umap`).
  2. Guards → `MethodSkip`, in this order **before** any subprocess/h5ad write:
     `n_obs < min_cells_total`; launcher not on PATH; backend unavailable from
     `context.backend_registry.get("celloracle")`; `celloracle` module unavailable in-env.
     (No cisTarget-style resource gate — the base GRN is built into CellOracle; the in-env
     script skips gracefully if the organism base GRN is missing.)
  3. Write counts h5ad to scratch (X ← `layers[layer]` if needed), including the resolved
     `rep_key`/`embedding_key` in `obsm` and `cluster_key`/`condition_key` in `obs`.
  4. Run `celloracle_ko.py` via backend; on `FileNotFoundError/TimeoutExpired/OSError` →
     `MethodSkip`; on non-zero return or `_SKIPPED`/`_FAILED` sentinel → `MethodSkip`.
  5. If `perturbation_ranking.csv` non-empty, render figures **in-process** (cellquorum env)
     via `perturbation_figures.py`, passing the real per-cell obs (`cluster_key`, and
     `condition_key` when set) and the `embedding_key` coords, aligned by `obs_names`.
  6. Build `StageArtifacts` for every file that exists (base GRN, per-cluster links, GRN
     summary, ranking CSV, per-TF shift parquets, each figure PNG+PDF). `metrics`:
     `n_tfs_screened`, `n_top_targets`, `condition_scored` (bool), `cluster_key`, `n_obs`.
     Return `StageResult(adata=adata, …, backend="celloracle")` (adata unchanged — the stage
     writes tables + figures, not obs/var, like `grn`/`coexpression`).

**`perturbation_figures.py`** — house-styled on `figstyle` (built fresh — no port source):
- `plot_ko_shift_field(shift_df, embedding_df, groups, out_dir, tf, ...)` — quiver/streamline
  of the KO perturbation vector field on the embedding (CellOracle's signature plot).
- `plot_target_ranking(ranking_df, out_dir, n_top, ...)` — lollipop/bar of top-N TFs by score
  (**flagship**, when directional).
- `plot_ko_fate_summary(shift_summary_df, out_dir, tf, ...)` — how a KO redistributes cells
  across clusters (transition summary).
- `plot_grn_connectivity(grn_summary_df, out_dir, ...)` — top regulators / degree summary of
  the fitted network.
- Every function: `figstyle.set_style()`; returns `list[Path]` (PNG+PDF); returns `[]` (or
  writes matching empty sentinels) on empty input — never raises to the caller. Palette from
  `figstyle.CATEGORICAL_PALETTE`.

**`stage.py`** — `PerturbationStage(MethodDispatchStage)`, `name="perturbation"`,
`stage_category="perturbation"`, `_select_method_name` → `config.get("method","celloracle")`.

**`__init__.py`** — register `CellOracleMethod` under `("perturbation", "celloracle")` as an
import side-effect (guarded by `METHOD_REGISTRY.has`), mirroring `grn/__init__.py`. Exports
`PerturbationConfig`, `CellOracleMethod`.

### Extension seam — the UDE method (documented, not built)

A future `method: ude` registers under `("perturbation", "ude")` and implements the same
`AnalysisMethod` contract: consumes counts + embedding + optional `condition_key`, produces
a `perturbation_ranking.csv` + shift artifacts in the same schema, reuses
`perturbation_figures.py` unchanged. No stage/planner/config-field changes required — only a
new method module + registration + (optionally) a `method`-selected env. This spec does not
implement it.

### The four wiring seams (all mandatory — the 3rd bit hdWGCNA)

1. **`config/models.py`**: import `PerturbationConfig`; add `perturbation: PerturbationConfig`
   field to `CellQuorumConfig`; add `perturbation: bool = True` to `StageSelectionConfig` with
   docstring.
2. **`core/executor.py`**: import `PerturbationStage`; add `"perturbation": PerturbationStage()`
   to `build_default_stage_registry` (after `grn`).
3. **`core/planner.py`**: add `("perturbation", self.config.stages.perturbation)` to the
   canonical `stage_flags` list — **slot it after `grn`, before `trajectory`** (implementer
   confirms the exact tuple index against the current list at build time).
   (This is the seam silently missed for hdWGCNA; a planner test guards it.)
4. **`backends/registry.py`**: import + `registry.register(build_celloracle_backend())` in
   `build_default_backend_registry`.

## Data flow

```
adata (counts + X_pca/rep + cluster labels + [condition label] + X_umap)
  ──▶ scratch h5ad ──▶ [celloracle_env] celloracle_ko.py
        [1] base GRN (built-in hg38) + PCA + kNN imputation + fit cluster GRNs
              └─▶ base_grn.parquet, links_<cluster>.parquet, grn_summary.csv
        [2] for each TF (tf_list OR systematic): simulate_shift(TF->0) + propagate
              └─▶ shift_vectors_<TF>.parquet, per-TF summary row
        [3] score: if condition_key+healthy_label -> project shift onto disease->healthy axis
                   else -> shift magnitude; rank TFs
              └─▶ perturbation_ranking.csv  (flagship)
  ranking + shifts + obs[cluster_key/condition_key] + obsm[embedding_key]
  ──▶ [cellquorum_env] perturbation_figures.py
        └─▶ ko_shift_field, target_ranking, ko_fate_summary, grn_connectivity  (PNG+PDF)
```
Shift parquets are indexed by `adata.obs_names`; figures align by index intersection — no
filename parsing.

## Error handling / skip matrix

| Condition | Where caught | Result |
|---|---|---|
| `n_obs < min_cells_total` | method guard | `MethodSkip` |
| launcher not on PATH | method guard (`shutil.which`) | `MethodSkip` |
| `celloracle` backend missing / module not in env | method guard | `MethodSkip` |
| built-in base GRN missing for organism | `celloracle_ko.py` | sentinel + exit 0 → `MethodSkip` |
| in-env import / GRN fit fails | `celloracle_ko.py` | sentinel + exit 0 → `MethodSkip` |
| real CellOracle CLI failure | in-env script | `_FAILED` + non-zero → `MethodSkip` |
| subprocess timeout / OSError | method (`except`) | `MethodSkip` |
| `condition_key`/`healthy_label` absent | method / script | **not** a skip → direction-agnostic outputs |
| empty ranking / no TFs fitted | method / figures | skip figures, still return `StageResult` |
| a single figure raises | method (`try` per figure) | note appended, others continue |

## Testing

Mirror the `grn` test set (fakes, no real CellOracle / base GRN in CI):
- **Backend** (`test_celloracle_backend.py`): `run_script` builds the right
  `micromamba run -n celloracle_env python …` argv; missing launcher raises
  `FileNotFoundError`; `status()`/`_py_module_available` probe shape.
- **Method** (`test_celloracle_method.py`) with a `FakeBackend`: skips on too-few-cells;
  skips when env/module unavailable; skips on non-zero return; skips on `TimeoutExpired`;
  `input_contract` does **not** require `condition_key`/`cluster_key`; on a faked successful
  run (backend writes stub ranking + shift parquets) builds artifacts + metrics; **both**
  directional (condition_key set) and direction-agnostic (unset) paths covered, asserting
  `metrics["condition_scored"]` accordingly.
- **Figures** (`test_perturbation_figures.py`): each `plot_*` writes PNG+PDF on synthetic
  input and returns `[]`/sentinels on empty; palette from `figstyle.CATEGORICAL_PALETTE`
  (no `theme` import).
- **Config** (`test_perturbation_config.py`): defaults; `StrictBaseModel` rejects unknown keys.
- **In-env script** (`test_celloracle_ko_script.py`): arg-parse + sentinel/skip behavior on
  synthetic input without a real CellOracle install (guard the heavy import, exercise the
  graceful-skip path + ranking-schema writer).
- **Wiring**:
  - planner: `test_perturbation_stage_is_planned_in_canonical_order` — `"perturbation" in order`,
    `order.index("grn") < order.index("perturbation")`, and
    `order.index("perturbation") < order.index("trajectory")` (matches the seam-3 placement:
    after `grn`, before `trajectory`).
  - executor registry: `"perturbation"` in `registered_stage_names()` (`tests/test_pipeline_executor.py`).
  - backend registry (`tests/test_backend_registry.py`): add `"celloracle"` to the
    `registry.names()` list and `row_names` set assertions.
  - config: `CellQuorumConfig().stages.perturbation is True`; `.perturbation` sub-block is a
    `PerturbationConfig`.
- **Trajectory e2e fixture** (`tests/test_trajectory_track_e2e.py`): add `"perturbation": False`
  to the disabled-stages dict (same cascade guard as `coexpression`/`grn`; its counts-layer
  contract would otherwise halt the trajectory-only run).

**Metrics:** `n_tfs_screened`, `n_top_targets`, `condition_scored` (bool), `cluster_key`, `n_obs`.

## README / docs updates

- Stages badge 26 → 27; "twenty-six" → "twenty-seven"; add `→ perturbation` to the backbone
  diagram after `grn`.
- Analysis-stages table: add
  `| perturbation | in-silico TF-knockout with CellOracle (own GRN + KO simulation, isolated env) — ranked therapeutic-target table + KO shift-field UMAP, fate-redistribution + GRN-connectivity figures | Implemented |`.
- Workflow spine: change the GRN line to
  `◐ gene-regulatory networks — ✅ co-expression modules (hdWGCNA); ✅ regulon/GRN inference (pySCENIC); ✅ in-silico perturbation (CellOracle)`
  and add a `✅ in-silico KO (CellOracle; ranked-target + KO shift-field figures)` spine line.
