# Configurations

| File or folder | What it is |
| --- | --- |
| `generic_pbmc_example.yaml` | A runnable example for any single-object dataset. Start here. |
| `config.yaml` | The reference configuration the CLI loads when no `--config` is given. |
| `qc/`, `preprocessing/`, `integration/`, `annotation/`, `markers/`, `backends/`, `profiles/`, `project/`, `manifests/` | Reusable building blocks for composing a configuration. |

To try CellQuorum without any data, run `cellquorum demo my_demo`, which writes a small
synthetic dataset and a commented config you can copy.
