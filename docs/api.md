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

## Reusable utilities — `cellquorum.utils`

A few of the engine's internal helpers are useful on their own in analysis
scripts, independent of a full pipeline run. They are exposed here as a stable,
versioned surface (`cq.utils.*`) — re-exports of the canonical implementations in
`cellquorum.comparative`, so a fix to the engine is a fix here. Importing this
module pulls in no heavy optional dependency (`get_net` lazy-imports `decoupler`
only when called).

```python
import cellquorum as cq

ranked = cq.utils.de_table_to_ranking(de_table)        # DE table -> preranked GSEA input
net = cq.utils.get_net("hallmark", organism="human")   # long-format prior-knowledge net
pb = cq.utils.aggregate_pseudobulk(                     # cells -> donor x condition pseudobulk
    adata, layer="counts", donor_col="donor_id", condition_col="condition"
)
```

::: cellquorum.utils.de_table_to_ranking

::: cellquorum.utils.get_net

::: cellquorum.utils.aggregate_pseudobulk

::: cellquorum.utils.PseudobulkResult

::: cellquorum.utils.PriorFetchError
