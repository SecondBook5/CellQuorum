# Architecture

CellQuorum separates an **execution spine** — the machinery that turns a config into
a validated, planned, provenance-tracked run — from a **config-driven analysis
backbone** of stages that do the science. This document describes how a run flows
through the engine and how the source is organized.

## Execution model

```mermaid
flowchart TD
    CFG([YAML config / dict / CellQuorumConfig]) --> VAL[Pydantic validation<br/>strict, extra=forbid]
    VAL --> REG[Backend registry<br/>Python / R / Rscript / GPU / RAPIDS]
    REG --> PLAN[Execution planner<br/>enabled · skipped · unimplemented]
    PLAN --> BOOT[Run bootstrapper<br/>directories + provenance]
    BOOT --> EXEC[Registry-driven executor]

    subgraph loop["for each planned stage, in canonical order"]
        direction TB
        DC{Input data<br/>contract valid?}
        DC -->|no| SK[Record skip with reason]
        DC -->|yes| DISP[Dispatch config-selected method]
        DISP --> UPD[Update AnnData]
        UPD --> REC[Record success / skip / failure]
    end

    EXEC --> loop
    loop -->|thread AnnData onward| EXEC
    REG -.-> PROV[(provenance/)]
    PLAN -.-> PROV
    EXEC -.-> PROV
```

1. **Validation.** A config (path, `dict`, or `CellQuorumConfig`) is parsed by a
   strict Pydantic schema. Unknown keys are rejected (`extra = "forbid"`), so typos
   fail loudly before any compute. The whole-config validator also rejects
   contradictory settings (e.g. `compute.backend: auto` with `fallback_to_cpu: false`).

2. **Backend detection.** The backend registry probes for Python, R/Rscript, GPU,
   and RAPIDS availability and produces a structured report. Detection gates on real
   capability, not merely a declared preference.

3. **Planning.** The planner walks the canonical stage order and, for each stage,
   marks it enabled (its `stages.*` flag is on and it is implemented), skipped
   (disabled), or unimplemented (a reserved slot). The plan is written to provenance
   before execution.

4. **Bootstrapping.** The run bootstrapper creates the standardized run directory
   and writes the resolved config, plan, backend status, run metadata (including an
   environment/version stamp), and an artifact manifest.

5. **Execution.** The registry-driven executor runs each enabled stage in order.
   For every stage it validates the input data contract, dispatches to the
   config-selected method, threads the updated `AnnData` to the next stage, and
   records a structured success, skip, or failure. A skipped or contract-failing
   stage leaves the `AnnData` unchanged and never aborts the run for a downstream
   stage that does not depend on it.

## Data contracts (fail-loud)

Every stage boundary is guarded by a data contract with three layers of checks:

- **Structural** — required layers, `obs` columns, and embeddings are present.
- **Layer-provenance** — layers carry semantic tags (e.g. "log-normalized"), so a
  stage cannot silently consume the wrong representation.
- **Statistical** — sanity checks such as rejecting raw counts that are mislabeled
  as log-normalized.

When a contract is not satisfied, the stage **skips with a recorded reason** rather
than crashing or producing wrong output. This is the core "no silent wrong answers"
guarantee.

## Method dispatch

Most analysis stages are `MethodDispatchStage`s: the stage reads a `method:` (or
`methods:`) key from its config block and dispatches to a registered
`AnalysisMethod`. Methods self-gate — an R method checks for Rscript and its R
packages; a GPU method checks for a capable device — and skip with a reason when
their backend is absent. This keeps method selection declarative (in YAML) and makes
backend availability a runtime concern rather than an install-time hard dependency.

R/Bioconductor methods share one abstraction, `cellquorum.methods.r_method.RAnalysisMethod`,
which centralizes Rscript resolution and R-package availability checks for the
edgeR, Milo, propeller, NicheNet, MultiNicheNet, and scDiagnostics adapters.

## Compute routing

A shared compute router selects CPU or GPU execution for routed operations
(normalization, PCA, neighbors, Leiden). It gates on real capability
(`rapids-singlecell` + `cupy` importable and a CUDA device present), honors
`compute.backend`/`prefer_gpu`/`fallback_to_cpu`, and guarantees that routed methods
emit identical output keys on both paths, so downstream contracts and results are
path-independent.

## Resume

With `run.resume: true`, the executor computes a deterministic input fingerprint for
each stage (stage config + input `AnnData` signature + seed). A side-effect-only
stage whose prior completion marker matches the current fingerprint — and whose
recorded artifacts still exist — is skipped as "resumed". Resume is best-effort: any
failure in fingerprinting or the resume decision degrades to normal execution, never
to a broken run.

## Source layout

The package is organized into 19 top-level packages under `src/cellquorum/`:

| Package | Responsibility |
|---|---|
| `core` | pipeline context, planner, executor, data contracts, provenance, run reporting, resume/fingerprint |
| `config` | Pydantic models, loader, validation, cohort/design/markers schemas |
| `methods` | method dispatch base classes, the method registry, and shared abstractions (incl. `RAnalysisMethod`) |
| `backends` | backend registry + subprocess adapters (pySCENIC, scCODA, CellOracle, scclr) and bundled R scripts |
| `qc` | quality control and ambient correction (`qc.ambient`) |
| `preprocessing` | normalization, feature selection, dimensionality reduction |
| `clustering` | clustering and recursive subclustering |
| `integration` | batch integration, embeddings, integration benchmark |
| `annotation` | annotation, consensus, diagnostics, adjudication, reference mapping, population identity |
| `differential_expression` | donor-aware pseudobulk DE and its visualization |
| `differential_abundance` | compositional/abundance testing |
| `enrichment` | GSEA/ORA/GSVA/decoupler and enrichment visualization |
| `gene_regulation` | co-expression (hdWGCNA), GRN (pySCENIC), perturbation (CellOracle) |
| `cell_cell_communication` | LIANA/Tensor-c2c/NicheNet, network topology + curvature, CCC visualization |
| `multicellular_programs` | cross-cell-type coordinated programs via DIALOGUE |
| `trajectory` | RNA velocity and trajectory visualization |
| `io` | input/output helpers |
| `visualization` | shared figure styling and plotting |
| `cli` | the `cellquorum`/`cq` Typer app and the `gen-configs` workflow commands |

Each stage package follows a uniform layout — `stage.py` (the stage implementation),
`config.py` (its Pydantic block), and one method module per method — which keeps the
dispatch surface predictable across stages.
