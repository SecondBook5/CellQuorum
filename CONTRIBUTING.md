# Contributing to CellQuorum

Thanks for looking at the code. This file covers the mechanics: how to get a working
environment, what CI enforces, and the conventions the codebase follows. For *what*
the engine does and how a run flows, read [`docs/reference/how-it-works.md`](docs/reference/how-it-works.md)
and [`docs/reference/architecture.md`](docs/reference/architecture.md) first.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). Changes to the files
listed in [`.github/CODEOWNERS`](.github/CODEOWNERS) are reviewed by their owners.

## Table of contents

- [Getting set up](#getting-set-up)
- [The test tiers](#the-test-tiers)
- [What CI enforces](#what-ci-enforces)
- [Code conventions](#code-conventions)
- [Adding a stage](#adding-a-stage)
- [Figures](#figures)
- [Adding a config option](#adding-a-config-option)
- [Public API and deprecation](#public-api-and-deprecation)
- [Commit and PR conventions](#commit-and-pr-conventions)
- [Releasing](#releasing)

## Getting set up

```bash
git clone https://github.com/SecondBook5/cellquorum.git
cd cellquorum

mamba create -n cellquorum-dev python=3.12 -y
mamba activate cellquorum-dev

# Editable install with the pinned dev toolchain.
python -m pip install -e ".[dev]"

# Install the git hooks (ruff lint + format, YAML/TOML checks, large-file guard).
pre-commit install
```

The `[dev]` extra pins **exact** versions of ruff, mypy, pytest, and the docs
toolchain, while the runtime dependencies stay as lower bounds. That asymmetry is
deliberate and explained in `pyproject.toml` — the short version is that a linter
which disagrees between your laptop and CI is worse than no linter. **Do not
loosen those pins in a feature PR.** Bumping a tool is its own commit so the
resulting churn is reviewable on its own.

Heavyweight backends (RAPIDS, pySCENIC, scCODA, CellOracle, hdWGCNA, R/Bioconductor)
have mutually incompatible dependency graphs and therefore live in separate conda
environments under [`envs/`](envs/). You do **not** need them to develop the engine:
tests that require them skip themselves. See [`docs/guide/backends.md`](docs/guide/backends.md).

## The test tiers

The full suite is slow, mostly because ~294 test modules each import the
single-cell stack. Use the tiers during development and let CI run everything:

```bash
make test-fast     # deselect slow/gpu/r/integration; the loop you use while coding
make test          # everything the current machine can run, in parallel
make test-cov      # everything, with a coverage report
make typecheck     # mypy on the gated packages (the list in pyproject.toml)
make lint          # ruff check + ruff format --check
```

Two mechanisms control what runs, and they are **not** interchangeable:

| Mechanism | Question it answers | Use it when |
|---|---|---|
| `@pytest.mark.skipif(...)` | "Is this dependency present?" | A test needs Rscript, a GPU, or an optional env. It must skip cleanly, never fail, when absent. |
| `@pytest.mark.slow` / `gpu` / `r` / `integration` | "Do I want to run this now?" | Always, on top of `skipif`, so the test can be deselected with `-m`. |

A test that needs R gets **both**: `skipif` so it degrades gracefully on a machine
without R, and `@pytest.mark.r` so `-m "not r"` can deselect it even where R exists.

Markers are declared in `pyproject.toml` under `[tool.pytest.ini_options]`. Because
`--strict-markers` is on, a typo'd marker is an error rather than a silent no-op.

### External datasets — never hardcode a path

**No tracked file may name a specific machine's filesystem.** `tests/config/test_no_hardcoded_machine_paths.py`
enforces this: it scans every git-tracked file under `configs/`, `src/`, `tests/`, and
`scripts/` for absolute paths rooted at `/mnt/`, `/home/`, `/Users/`, `/media/`,
`/Volumes/`, or a Windows drive letter, and fails with the full list.

This is not tidiness. `configs/config.yaml` is what the CLI loads when no `--config` is
given, and it used to point at the author's external drive — so a new user's first
`cellquorum run` tried to write to a path that did not exist on their machine. Several
integration tests hardcoded data paths too, which meant they were permanently skipped for
everyone else, and the skip reason was indistinguishable from a misconfiguration.

Point at data through the environment instead:

**In YAML** — the loader resolves OmegaConf interpolations, so use `${oc.env:VAR}`, or
`${oc.env:VAR,fallback}` for a default. An unset variable with no fallback fails at
config-load time with a message naming both the variable and the config key, which is the
right moment to find out.

**In tests** — use the helpers in `tests/_external_data.py`, which skip with an actionable
message rather than failing:

```python
from _external_data import require_cellranger_library, require_r_package

def test_something_on_real_data(tmp_path):
    library = require_cellranger_library("Set1_norm_LE", "LE1_v8", "outs",
                                         needs=("raw_feature_bc_matrix.h5",))
    require_r_package("SoupX")
```

Resolve external data **inside the test or a fixture**, not at module scope. A
module-level `skipif` runs its filesystem stats and `Rscript` subprocesses during
collection, for every session, even when the test is deselected.

Variables the shipped configs and tests read:

| Variable | Points at |
|---|---|
| `CELLQUORUM_CELLRANGER_ROOT` | Cell Ranger output root (the dir holding per-run `outs/`) |
| `CELLQUORUM_KC_H5AD` | SoupX-corrected keratinocyte object, for `configs/le_kc.yaml` |
| `CELLQUORUM_AD_ATLAS_H5AD` | Atopic-dermatitis reference atlas (Broad SCP2613) |
| `CELLQUORUM_KC_ATLAS_H5AD` | Fiskin keratinocyte reference atlas |
| `CELLQUORUM_SMOKE_H5AD` | Subsampled smoke-test cohort (has a `/tmp` default) |
| `CELLQUORUM_TEST_CELLRANGER_ROOT` | Cell Ranger root **for tests** |
| `CELLQUORUM_TEST_KC_H5AD` | Reference-mapped keratinocyte object **for tests** |

The `TEST_` variables are separate on purpose: configuring a run should not silently
switch on multi-minute integration tests.

A test that only checks a config *validates* should stub these rather than depend on your
environment — use `stub_config_env(monkeypatch, tmp_path)` from `tests/_external_data.py`.
`tests/config/test_no_hardcoded_machine_paths.py` also walks every shipped config and asserts it
still validates, so a field renamed in `config/models.py` fails there instead of in a
user's first run.

If a literal absolute path is genuinely required, append a `machine-path-ok` comment to
that line to opt out.

### The lazy-import invariant

`tests/repo/test_import_cost.py` asserts that a bare `import cellquorum` does not pull in
`torch`, `scvi`, `lightning`, `celltypist`, or `decoupler`, and stays under a module
ceiling. This is load-bearing, not cosmetic: a single module-scope
`import scvi` in a stage package once made `cellquorum --version` take 4.7 seconds
and added a multi-second tax to every test module.

**Probe for an optional dependency with `importlib.util.find_spec("name") is not
None`, never with `import name` inside `try/except ImportError`.** The `try/except`
form stops the crash but still executes the module when it *is* installed, which is
the expensive case. Import optional backends inside the method body that uses them.

## What CI enforces

Every push and PR runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Job | Enforces |
|---|---|
| **Lint** | `ruff check` and `ruff format --check` on `src` and `tests`, at the exact pinned version. |
| **Typecheck** | `mypy` on the packages listed under `files` in `[tool.mypy]` in `pyproject.toml`, with no module exempted. |
| **Test** | The suite on Python 3.12 and 3.13, with a coverage floor (`fail_under` in `pyproject.toml`). Optional-backend tests self-skip. |
| **Figures** | `tests/visualization/test_figure_gallery.py`: every gallery figure must match its stored baseline pixel for pixel. |
| **Docs** | `mkdocs build --strict` — a broken link or unresolved API reference fails the build. |
| **Build** | Builds the wheel and sdist, asserts the wheel bundles every `.R` script, a non-empty `LICENSE` and `py.typed`, then installs the wheel into a clean environment and runs `tests/acceptance` there: the Python API and the command line a user types. |

A separate workflow, [`.github/workflows/security.yml`](.github/workflows/security.yml), runs weekly
and on changes to `pyproject.toml`: **CodeQL** reads our own code for injection and unsafe-file-handling
bugs, and **pip-audit** checks the packages a clean `pip install` brings in against public advisory
databases. It is kept apart from CI so that a newly published advisory for someone else's package
cannot turn an unrelated pull request red. A red audit means a dependency needs a newer floor in
`pyproject.toml` (or, if no fix exists, a note in `SECURITY.md`).

### The mypy ratchet

mypy gates the packages listed under `files` in `[tool.mypy]` in `pyproject.toml`, and **none of
them has an exemption**. That list is the only place the gated packages are named: CI and
`make typecheck` both just run `mypy`, so there is no second copy to forget to update.
The rest of the package (`visualization/`, `backends/` and most stages) still has type errors,
mostly untyped-third-party fallout; gating all of it at once would be a permanently red job
that everyone learns to ignore.

To widen the gate, pick one subpackage, run `mypy` on it, fix its errors, then add its path to
`files`. Never add a package with errors still in it, and never add an `ignore_errors` override
to make a PR pass.

## Code conventions

**Comments explain *why*, not *what*.** The single most useful thing you can write is
the reason a non-obvious choice was made — especially a rejected alternative. The
module docstring in `src/cellquorum/core/stages.py` is the model: it records why
`pkgutil.walk_packages` auto-discovery was rejected, which stops a future
contributor from "simplifying" it back into a bug.

Conversely, do not narrate the next line. `# Print the version string.` above
`console.print(...)` adds no information and silently becomes wrong when the code
moves. Some older modules are written this way; new code should not be, and
touching such a block is a fine time to fix it.

**Annotate honestly.** Do not write `: object` to sidestep an import — it disables
type checking exactly where a reader most needs the type. Use a real type, with
`if TYPE_CHECKING:` when the import would be circular. The file already has
`from __future__ import annotations`, so the annotation costs nothing at runtime:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellquorum.core.planner import PipelinePlan


def _restrict_plan_from_stage(plan: PipelinePlan, from_stage: str) -> PipelinePlan: ...
```

**No silent wrong answers.** This is the project's central design rule. A stage
whose inputs are missing must *skip with a recorded reason*, never guess and never
emit a plausible-looking wrong result. Validate at boundaries via
`core/contracts/`, and record skips, failures, and never-run stages distinctly in
provenance.

**Google-style docstrings** on public functions, with `Args:`, `Returns:`, and
`Raises:`. `mkdocstrings` renders these into the API reference, and the docs build
is strict, so a malformed docstring fails CI.

## Adding a stage

1. Create `src/cellquorum/stages/<name>/` with `config.py` (a `StrictBaseModel`
   subclass) and `stage.py` (a `PipelineStage` subclass decorated with
   `@register_stage`).
2. Add the config to `CellQuorumConfig` in `src/cellquorum/config/models.py`.
3. Add the import to `src/cellquorum/core/stages.py`, positioned to match pipeline
   order. That block is bracketed by `# isort: off` / `# isort: on` because the
   ordering is narrative — do not alphabetize it.
4. Keep heavy or optional imports **inside** the method body.
5. Add tests, including a skip-path test proving the stage records a reason rather
   than crashing when its inputs are absent.

## Adding a config option

The config surface is already large (~640 fields across 30 stage modules), and it
is the main thing standing between a wet-lab user and a successful run. Before
adding a field, ask whether the engine can infer it.

If it must exist: give it a **safe default** so existing configs keep working,
document it in `docs/guide/configuration.md`, and add a validator that fails at load time
rather than mid-run. Config errors should be impossible to discover late.

## Figures

Figures are part of the product, so a code change must not alter how one looks by accident.

- **One theme.** Every figure applies its look with `figstyle.apply_theme(style)`, where `style`
  is `"compact"`, `"notebook"` or `"publication"`. The argument is required. Do not call
  `matplotlib.rcParams` setters from stage code.
- **Every figure has a baseline.** Fixtures live in `scripts/gallery_<stage>.py`: small synthetic
  inputs and the call to the renderer. `scripts/build_figure_gallery.py` discovers them, and
  `tests/visualization/test_figure_gallery.py` compares each output to `docs/gallery/assets/<name>.png`.
  Run the comparison with `CELLQUORUM_CHECK_FIGURE_BASELINES=1 pytest tests/visualization/test_figure_gallery.py`.
- **Adding a figure:** add a function to the stage's `gallery_<stage>.py` registry, build to a
  scratch folder, look at the result, copy only that PNG into `docs/gallery/assets/`, then run
  `python scripts/update_gallery_page.py`.
- **Changing a figure on purpose:** the same recipe. Build, inspect before and after, copy just
  the PNG you intended to change, and say in the commit message what changed visually.

```bash
python scripts/build_figure_gallery.py --output /tmp/gallery_new
cp /tmp/gallery_new/<name>.png docs/gallery/assets/<name>.png
CELLQUORUM_CHECK_FIGURE_BASELINES=1 pytest tests/visualization/test_figure_gallery.py
```

Do not regenerate everything into `docs/gallery/assets`: the PDF and SVG files embed timestamps
and would show up as changes you did not make.

**Beware libraries that change matplotlib on import.** `pycirclize` and `decoupler` rewrite global
settings the first time they are imported, so the same figure can look different depending on what
ran before it. Import such a library inside `with matplotlib.rc_context():` and test it in a
fresh subprocess; see `tests/stages/gene_regulation/grn/test_grn_radial_figures.py` for an example.

## Public API and deprecation

Three things count as the public interface, and changes to them are treated differently from
changes to internals:

1. **Python names** listed in the `__all__` of the top-level packages. The list is frozen in
   `tests/api/test_public_api_contract.py`; a rename or removal fails that test, so it cannot
   happen by accident. Anything not in an `__all__` (including everything under `stages/`
   unless re-exported) is internal and may change in any release.
2. **The configuration file.** Settings are validated strictly, so a renamed or removed key
   makes an existing config fail with an "unknown setting" error rather than being ignored.
3. **The `cellquorum` command and the layout of a run folder** (`results/`, `figures/`,
   `reports/`, `objects/`, `provenance/`).

CellQuorum is pre-1.0, so a minor release (0.2 to 0.3) is allowed to break any of these. The rules
below are about doing it deliberately rather than quietly:

- **Say it in the CHANGELOG.** Every break goes under `Changed` or `Removed` in the release's
  entry, and for a renamed config key it states the old name and the new one, because that is
  the sentence a lab member will search for when their config stops loading.
- **Deprecate first where it is cheap.** A function that is superseded stays for at least one
  minor release and calls `warnings.warn(..., DeprecationWarning, stacklevel=2)` with the name
  of the replacement and the reason (`module_gene_overlap` in `stats/module_remodeling.py` is
  the model). A config key is the exception: because configs are strict, keep accepting the old
  key only if it can be mapped to the new one without ambiguity, and warn when it is used.
- **Never change what a result means under the same name.** If the statistics of a table or
  figure change, change its name or add a note to its provenance, so that two runs of different
  versions are not silently compared as if they were the same analysis.
- **From 1.0 on,** the rule tightens to semantic versioning: breaking changes only in a major
  release, and a deprecation lasts at least one minor release.

## Commit and PR conventions

Commits follow a `type(scope): summary` convention, matching the existing history:

```
feat(qc): add per-library ambient contamination estimate
fix(trajectory): stop the kernel plot from re-normalizing velocities
docs(configuration): document the subclustering donor gate
refactor(core): replace `: object` annotations in pipeline.py
```

Keep a PR to one logical change. Specifically, do not mix a dev-tool version bump
(which reformats many files) with a behavior change — the review becomes impossible.

Before opening a PR:

```bash
make lint && make typecheck && make test-fast
```

## Releasing

Releases are cut by pushing a tag. The
[`release.yml`](.github/workflows/release.yml) workflow verifies that the tag
matches `src/cellquorum/version.py`, requires a `CHANGELOG.md` heading for the version, runs
the whole CI workflow (including the clean-install acceptance tests), rebuilds and re-checks the
distributions, and publishes to PyPI via Trusted Publishing (no API token stored in the repo).

1. Update `__version__` in `src/cellquorum/version.py`.
2. Move the `CHANGELOG.md` "Unreleased" entries under the new version with a date.
3. Commit, then tag and push:

```bash
git tag -a v0.2.0 -m "v0.2.0"
git push origin v0.2.0
```

The tag/version guard exists because shipping a wheel whose metadata disagrees with
its tag is unfixable after the fact — PyPI does not allow re-uploading a version.
