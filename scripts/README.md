# Maintained command-line helpers

Reusable analysis belongs in `src/cellquorum`, with its public commands exposed
through the `cellquorum` CLI. This directory holds development checks and narrow
inspection/export utilities. Keep helper scripts thin; move reusable computation
and rendering into the package rather than copying it between scripts.

| Helper | Purpose |
| --- | --- |
| `audit_gse140819.py` | Audit benchmark archive provenance, barcode joins, and reference-label coverage. |
| `verify_gpu.py` | Probe RAPIDS or PyTorch/scVI CUDA execution capabilities. |
| `lock_environments.py` | Run pinned conda-lock with its missing manylinux tags restored. |
| `build_figure_gallery.py` | Generate publication examples and synthetic source tables. |
| `smoke_real_h5ad.py` | Exercise the pipeline against an existing H5AD file. |
| `inspect_reference_atlas.py` | Inspect SCP2613 reference metadata and query gene overlap. |
| `inspect_celltypist_labels.py` | Inspect CellTypist model labels. |
| `export_le_global.py` | Export the existing LE study's global analysis outputs. |
| `validate_le_global.py` | Validate the LE global analysis outputs. |
| `verify_le_manifest.py` | Check the LE study's manifest. |

The LE helpers are study-specific maintained tooling, not generalized analysis
APIs. Local `le_*.py` experiment scripts are ignored by Git and are not shipped
as package capabilities. Do not make the installed pipeline depend on them.

CI checks tracked scripts with Ruff. From the repository root:

```bash
ruff check scripts
ruff format --check scripts
```

GPU probes verify environment capabilities; the installed-workflow acceptance
suite separately verifies pipeline outputs and checkpoint continuation.
