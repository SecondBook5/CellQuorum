# Maintaining CellQuorum

This page is for whoever looks after the repository: keeping it healthy, reading a red build,
cutting a release, and working out what went wrong in a run. It assumes you can use git and a
terminal, and nothing more. The longer rules (code style, how to add a stage, how figures are
pinned) are in [CONTRIBUTING.md](https://github.com/SecondBook5/cellquorum/blob/main/CONTRIBUTING.md); this page
says *when* to reach for them.

## The five-minute health check

From a checkout with the dev environment active:

```bash
make check
```

That runs the linter, the type checker and the fast tests, which is what a pull request has to
pass first. If it is green, the repository is in the state its own rules describe.

## What each CI job means

Every push runs `.github/workflows/ci.yml`. When one job is red, this is what it is telling you.

| Job | What red means | What to do |
|---|---|---|
| CPU and GPU lock integrity | The pinned environment files disagree with each other. | Someone edited an `envs/*.yml` recipe without re-locking. Run `make lock` and commit the lock files. |
| Lint | Formatting or style. | `make format`, review the diff, commit. Never hand-wave a lint failure with `# noqa` unless the reason is written next to it. |
| Typecheck | A type error in a gated package. | Fix the code. The gated list is `files` in `[tool.mypy]` in `pyproject.toml`. Do not shrink it to make a build pass. |
| Test (3.12, 3.13) | A test failed, or coverage fell under `fail_under` in `pyproject.toml`. | Read the failing test's name: the suite mirrors the source layout, so `tests/stages/qc/` tests `stages/qc/`. A coverage failure means new code arrived without tests. |
| Publication figure comparisons | A figure no longer matches its stored image pixel for pixel. | Download the `figure-comparisons` artifact from the run and look at the difference. If you meant to change the figure, follow *Changing a figure on purpose* in CONTRIBUTING. If you did not, the code change altered a figure by accident: fix the code, not the image. |
| Docs build (strict) | A broken link or an unresolvable API reference. | `make docs` reproduces it locally and names the page. |
| Build | The wheel is missing a bundled R script, the license, or `py.typed`, or the installed package fails its end-to-end tests. | This catches mistakes that only appear when the package is *installed* rather than run from the checkout. Reproduce by building the wheel into a clean environment. |

A separate workflow, `.github/workflows/security.yml`, runs CodeQL on our code and `pip-audit` on a
clean install every Monday. A red audit means a dependency has a published advisory: raise its
minimum version in `pyproject.toml`. It is deliberately not part of the pull-request checks, so an
advisory published today cannot block an unrelated change.

## Routine jobs

**Dependabot pull requests (monthly).** They bump the pinned development tools and GitHub Actions.
Merge them one at a time and let CI judge each. A ruff bump may reformat files; that belongs in its
own commit, apart from any behaviour change.

**Updating the analysis environments.** The recipes are `envs/*.yml` and the resolved locks are
`envs/*.conda-lock.yml`. After changing a recipe:

```bash
make lock
```

Commit the recipe and the lock together. The *CPU and GPU lock integrity* job fails if they
disagree.

**Cutting a release.** Follow *Releasing* in CONTRIBUTING: bump `version.py`, move the changelog's
"Unreleased" entries under the new version, then tag and push. The release workflow refuses a tag
that does not match the version, refuses a version with no changelog entry, and runs the whole CI
suite before it publishes. PyPI does not allow re-uploading a version, so a wrong release is fixed
by a new one.

**Changing what is public.** Read *Public API and deprecation* in CONTRIBUTING before renaming a
function or a configuration setting. A lab member's saved config is a public interface.

## Finding out what happened in a run

A run writes everything under its output folder. The parts you will open:

| Path | What it holds |
|---|---|
| `reports/report.html` | The run report: stages, warnings, figures. Start here. |
| `reports/methods.txt` | A draft methods paragraph for the stages that ran. |
| `results/<stage>/` | The analysis tables. |
| `figures/<stage>/` | The figures, as image and vector files. |
| `objects/final_annotated.h5ad` | The final data object. |
| `provenance/resolved_config.json` | The exact configuration used, after defaults were applied. |
| `provenance/run_metadata.json` | The random seed, run id and software environment. |
| `provenance/stage_execution_records.json` | One record per stage: whether it ran, was skipped (with the reason) or failed (with the error), its warnings, its duration and its inputs and outputs. |
| `provenance/artifact_manifest.csv` | Every file the run produced. |

To diagnose a stage that did not do what was expected, open its record in
`stage_execution_records.json` (or the `.csv` beside it): `status`, `skip_reason`, `error` and
`warnings` say why. A stage that is switched off in the configuration is recorded as skipped too, with the reason
"Disabled by configuration"; the terminal summary counts those and lists by name only the stages
that were wanted but could not run.

Two things about a run folder are easy to be surprised by. The `logs/` folder is created but
nothing writes to it yet, so the provenance records and the terminal output are the log. And
setting `run.resume` lets a re-run into the same folder skip stages that already finished with
unchanged inputs; `--from-stage` and `--until-stage` (see `cellquorum run --help`) re-run only part
of a pipeline, and `--from-stage` needs a first run made with `run.checkpoint` on.

## Where to look when you want to change something

| You want to | Start at |
|---|---|
| Add an analysis step | *Adding a stage* in CONTRIBUTING |
| Add a setting | *Adding a config option* in CONTRIBUTING, then `docs/guide/configuration.md` |
| Add or change a figure | *Figures* in CONTRIBUTING and `scripts/gallery_<stage>.py` |
| Understand how the pieces fit | [Architecture](reference/architecture.md) and [How it works](reference/how-it-works.md) |
| Check a licence or citation | [Third-party software](reference/third-party-software.md) |

## Known gaps

These are open on purpose and written down so nobody assumes they are covered.

- The R and GPU tests are skipped on the standard CI runners, because the runners have neither.
  They pass on a machine that has them; run the full suite there (`make test`) before a release.
- The type checker covers part of the package, not all of it. The ratchet in CONTRIBUTING describes
  how to widen it.
- Runs have no log file of their own; see above.
