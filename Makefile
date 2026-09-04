VERSION := $(shell python -c "import cellquorum; print(cellquorum.__version__)")
IMAGE := cellquorum:$(VERSION)
IMAGE_GPU := cellquorum:$(VERSION)-gpu
REQUIRED_ENVS := cellquorum-core celloracle_env pyscenic_env hdwgcna_env scclr sccoda_env cellquorum-r

.PHONY: image image-gpu lock smoke matrix docs docs-serve \
        help lint format typecheck test test-fast test-cov check

# Default target: list what is available rather than doing something surprising.
.DEFAULT_GOAL := help

help:
	@echo "Development:"
	@echo "  make check      - lint + typecheck + test-fast (run before opening a PR)"
	@echo "  make lint       - ruff check + ruff format --check"
	@echo "  make format     - ruff format + ruff check --fix (rewrites files)"
	@echo "  make typecheck  - mypy on the engine spine"
	@echo "  make test-fast  - fast tier only; the loop to use while coding"
	@echo "  make test       - everything this machine can run, sharded across cores"
	@echo "  make test-cov   - everything, with a coverage report"
	@echo "  make docs       - build the docs site (strict)"
	@echo "  make docs-serve - serve the docs with live reload"
	@echo ""
	@echo "Packaging and environments:"
	@echo "  make image      - build the CPU Docker image"
	@echo "  make image-gpu  - build the GPU Docker image"
	@echo "  make lock       - regenerate conda-lock files for envs/"
	@echo "  make smoke      - smoke-test the built image"
	@echo "  make matrix     - run the Snakemake matrix inside the image"

# Mirror the CI lint job exactly, so a green `make lint` means a green CI lint.
lint:
	ruff check src tests
	ruff format --check src tests

# The write side of `make lint`. Separate target because a formatter that runs
# implicitly during a check is how unreviewed diffs appear in a PR.
format:
	ruff format src tests
	ruff check --fix src tests

# Mirror the CI typecheck job: the engine spine only. The mypy backlog lives in
# pyproject.toml under [[tool.mypy.overrides]]; shrinking it widens this gate.
typecheck:
	mypy src/cellquorum/core src/cellquorum/config src/cellquorum/io \
	     src/cellquorum/methods src/cellquorum/cli

# The tight development loop. Deselects the slow, GPU, R, and integration tiers so
# the run is dominated by actual assertions rather than backend startup.
test-fast:
	pytest -m "not slow and not gpu and not r and not integration"

# Everything this machine can run. Tests needing an absent backend self-skip.
# -n auto shards across cores: the suite spends most of its time importing the
# single-cell stack, so sharding trades memory for a large wall-clock win.
test:
	pytest -n auto

# Full suite plus coverage. Coverage is not in the default addopts because the
# tracer slows every local run; ask for it explicitly or let CI do it.
test-cov:
	pytest -n auto --cov --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

# What CI will check, minus the slow tiers. Run this before opening a PR.
check: lint typecheck test-fast

# Build the documentation site (strict: warnings — broken links, unresolved
# API references — fail the build, matching the CI docs job).
docs:
	mkdocs build --strict

# Serve the docs with live reload for local authoring.
docs-serve:
	mkdocs serve

image:
	docker build --target cpu -t $(IMAGE) -f docker/Dockerfile .

image-gpu:
	docker build --target gpu -t $(IMAGE_GPU) -f docker/Dockerfile .

# Generate reproducible conda-lock files for every environment recipe.
# Platform is pinned to linux-64 on purpose: the image builds on the linux-64
# micromamba base, and the GPU env carries linux-only CUDA packages
# (pytorch-cuda, the nvidia channel) that have no osx-64/win-64 build — an
# unpinned multi-platform solve would fail on those envs. linux-64 is the only
# platform we ship, so it is the only platform we lock.
lock:
	@command -v conda-lock >/dev/null || { echo "install conda-lock first"; exit 1; }
	for f in envs/*.yml; do \
	  [ -s "$$f" ] || continue; \
	  conda-lock lock -p linux-64 -f "$$f" --lockfile "$${f%.yml}.conda-lock.yml" || exit 1; \
	done

smoke:
	docker run --rm $(IMAGE) --version
	docker run --rm --entrypoint micromamba $(IMAGE) run -n cellquorum-core cellquorum plan --config docker/smoke/smoke.yaml --json
	@for env in $(REQUIRED_ENVS); do \
	  docker run --rm --entrypoint micromamba $(IMAGE) env list | grep -qw $$env \
	    || { echo "MISSING ENV: $$env"; exit 1; }; \
	done
	@echo "smoke OK"

matrix:
	docker run --rm -v $(PWD):/work -w /work --entrypoint micromamba $(IMAGE) \
	  run -n cellquorum-core snakemake --snakefile workflow/Snakefile --cores $(or $(CORES),4) --keep-going
