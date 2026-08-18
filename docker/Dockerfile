# syntax=docker/dockerfile:1

########## base: core env + package ##########
FROM mambaorg/micromamba:1.5.8 AS base
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /opt/cellquorum
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/cellquorum-core.yml /tmp/core.yml
RUN micromamba create -y -n cellquorum-core -f /tmp/core.yml && micromamba clean -a -y
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/cellquorum
RUN micromamba run -n cellquorum-core pip install -e .

########## cpu: base + R env ##########
# The R/Bioconductor env powers pseudobulk DE (edgeR/limma/DESeq2), SingleR, and
# the other R-dispatch adapters called over the rpy2 bridge. It is on the
# critical path because the standard pipeline (incl. the fibroblast 7/7) uses
# pseudobulk_edger for differential expression.
FROM base AS cpu
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/cellquorum-r.yml /tmp/r.yml
RUN micromamba create -y -n cellquorum-r -f /tmp/r.yml && \
    micromamba run -n cellquorum-r pip install -e . && \
    micromamba clean -a -y
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "micromamba", "run", "-n", "cellquorum-core", "cellquorum"]
CMD ["--help"]

########## gpu: cpu + GPU env (RAPIDS + CUDA torch) ##########
FROM cpu AS gpu
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/cellquorum-gpu.yml /tmp/gpu.yml
RUN micromamba create -y -n cellquorum-gpu -f /tmp/gpu.yml && micromamba clean -a -y
# Install the package editable into the GPU env. Done here (not via `-e ..` in
# the env yml) because the source is present at /opt/cellquorum by now and the
# yml's relative path would resolve against /tmp inside the image.
RUN micromamba run -n cellquorum-gpu pip install -e .
# Run the CLI IN the GPU env so runtime capability-detection
# (compute/router.py: cupy + rapids_singlecell import + CUDA device) can see the
# GPU. cellquorum-gpu is a superset of cellquorum-core, so every in-process
# stage still works here. Without this override the GPU image would inherit the
# cpu ENTRYPOINT and run in cellquorum-core, leaving all GPU acceleration dead.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "micromamba", "run", "-n", "cellquorum-gpu", "cellquorum"]
CMD ["--help"]

########## backends: OPTIONAL external subprocess backends, OFF the cpu/gpu path ##########
# pySCENIC (GRN), hdWGCNA (co-expression), scCODA (differential abundance), and
# scclr are heavyweight isolated envs dispatched as subprocesses. The router
# skips any that are absent at runtime, and none is needed for the standard 7/7
# pipeline. They are deliberately NOT in the cpu/gpu lineage so a solve/build
# failure in one never blocks the core engine image. Build with `--target
# backends` (this layers them onto the full cpu image).
# TODO: hdWGCNA env currently fails to solve on this base image — harden before
# relying on --target backends.
FROM cpu AS backends
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/pyscenic_env.yml /tmp/pyscenic_env.yml
RUN micromamba create -y -n pyscenic_env -f /tmp/pyscenic_env.yml && micromamba clean -a -y
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/hdwgcna_env.yml /tmp/hdwgcna_env.yml
RUN micromamba create -y -n hdwgcna_env -f /tmp/hdwgcna_env.yml && micromamba clean -a -y
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/sccoda_env.yml /tmp/sccoda_env.yml
RUN micromamba create -y -n sccoda_env -f /tmp/sccoda_env.yml && micromamba clean -a -y
# scclr: bake toolchain from recipe; install the package only if SCCLR_SRC is provided.
ARG SCCLR_SRC=""
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/scclr.yml /tmp/scclr.yml
RUN micromamba create -y -n scclr -f /tmp/scclr.yml && micromamba clean -a -y
RUN if [ -n "$SCCLR_SRC" ]; then micromamba run -n scclr pip install "$SCCLR_SRC"; \
    else echo "SCCLR_SRC not set: scclr toolchain baked, package NOT installed"; fi

########## celloracle: in-silico-KO backend, isolated off the cpu/gpu path ##########
# Fragile to build: celloracle pulls gimmemotifs, whose native (C) wheel build
# fails on the compiler in this base image. Isolated here so it never blocks the
# gpu image. Build explicitly with `--target celloracle`.
# TODO(#160): install gimmemotifs from bioconda (precompiled) instead of letting
# pip build it, then expose a `full` target that layers this env onto `gpu`.
FROM base AS celloracle
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/celloracle_env.yml /tmp/celloracle_env.yml
RUN micromamba create -y -n celloracle_env -f /tmp/celloracle_env.yml && \
    micromamba run -n celloracle_env pip install --no-build-isolation velocyto celloracle && \
    micromamba clean -a -y
