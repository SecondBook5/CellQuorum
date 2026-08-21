# Python API

CellQuorum has two front doors: the `cellquorum` / `cq` command-line interface
(see the [README](https://github.com/SecondBook5/cellquorum#cli-reference)) and the
Python API documented here. The Python surface is deliberately small — one
entry point plus four notebook namespaces — and re-exported from the top-level
`cellquorum` package, so `cq.run_pipeline`, `cq.tl`, `cq.pp`, `cq.diag`, and
`cq.evidence` are the canonical import paths.

```python
import cellquorum as cq

result = cq.run_pipeline(config="configs/config.yaml")
```

## Pipeline entry point

::: cellquorum.run_pipeline

::: cellquorum.api.pipeline.PipelineRunResult

## Notebook namespaces

These namespaces expose the engine's stages as ergonomic functions for
interactive (notebook) use. Each is a thin adapter over the same stage classes
the CLI runs, so a notebook exploration and a config-driven run stay consistent.

### Tools — `cellquorum.tl`

::: cellquorum.api.tl

### Preprocessing — `cellquorum.pp`

::: cellquorum.api.pp

### Diagnostics — `cellquorum.diag`

::: cellquorum.api.diag

### Evidence — `cellquorum.evidence`

::: cellquorum.api.evidence
