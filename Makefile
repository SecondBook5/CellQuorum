VERSION := $(shell python -c "import cellquorum; print(cellquorum.__version__)")
IMAGE := cellquorum:$(VERSION)
IMAGE_GPU := cellquorum:$(VERSION)-gpu
REQUIRED_ENVS := cellquorum-core celloracle_env pyscenic_env hdwgcna_env scclr sccoda_env cellquorum-r

.PHONY: image image-gpu lock smoke matrix

image:
	docker build --target cpu -t $(IMAGE) -f docker/Dockerfile .

image-gpu:
	docker build --target gpu -t $(IMAGE_GPU) -f docker/Dockerfile .

lock:
	@command -v conda-lock >/dev/null || { echo "install conda-lock first"; exit 1; }
	for f in envs/*.yml; do \
	  [ -s "$$f" ] || continue; \
	  conda-lock lock -f "$$f" --lockfile "$${f%.yml}.conda-lock.yml" || exit 1; \
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
