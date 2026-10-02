# Configurations

| File or folder | What it is |
| --- | --- |
| `generic_pbmc_example.yaml` | A runnable example for any single-object dataset. Start here. |
| `config.yaml` | The reference configuration the CLI loads when no `--config` is given. |
| `qc/`, `preprocessing/`, `integration/`, `annotation/`, `markers/`, `backends/`, `profiles/`, `project/`, `manifests/` | Reusable building blocks for composing a configuration. |
| `le_*.yaml`, `mast_cell.yaml` | The lab's own lymphedema and mast-cell studies, which are the real-data validation cases for the engine. They are examples of a fully worked project, not templates. |

To try CellQuorum without any data, run `cellquorum demo my_demo`, which writes a small
synthetic dataset and a commented config you can copy.
