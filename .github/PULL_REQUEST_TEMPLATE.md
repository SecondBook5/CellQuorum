# Summary

<!-- What changes and why. If this fixes an issue, write "Fixes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New method for an existing stage
- [ ] New stage
- [ ] Engine / CLI / provenance change
- [ ] Documentation
- [ ] Dev tooling or CI (should not be mixed with any of the above)

## Checks

```bash
make lint && make typecheck && make test-fast
```

- [ ] `make lint` passes
- [ ] `make typecheck` passes
- [ ] Tests pass, and I added tests covering the change
- [ ] Docstrings updated (the docs build is `--strict`; a malformed docstring fails CI)

## Project-specific review points

<!-- Delete any that don't apply. -->

- [ ] **No silent wrong answers.** New failure modes either raise or record a skip
      with a reason. Nothing guesses a value and continues.
- [ ] **Lazy imports preserved.** No new module-scope import of an optional heavy
      backend (`torch`, `scvi`, `celltypist`, `decoupler`, …). Optional deps are
      probed with `importlib.util.find_spec`, not `import` inside `try/except`, and
      imported inside the method body that uses them.
      `tests/test_import_cost.py` enforces this.
- [ ] **Honest annotations.** No new `: object` used to sidestep an import; real
      types with `if TYPE_CHECKING:` where needed.
- [ ] **Markers, not just skipif.** Any test needing R, a GPU, or an external tool
      carries both a `skipif` (so it degrades gracefully) and the matching
      `slow`/`gpu`/`r`/`integration` marker (so it can be deselected).
- [ ] **No mypy backlog growth.** I did not add a module to the
      `[[tool.mypy.overrides]]` `ignore_errors` list to make this pass.
- [ ] **Config additions justified.** A new config field has a safe default, a
      load-time validator, and an entry in `docs/configuration.md` — and the engine
      genuinely cannot infer it.

## Provenance / output impact

<!-- Does this change what lands in the run directory, the artifact manifest, or the
     resolved config schema? Anything that changes an existing config's meaning is a
     breaking change and needs a CHANGELOG note. -->

- [ ] No change to existing outputs, or the change is noted in `CHANGELOG.md`
