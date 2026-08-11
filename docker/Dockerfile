# syntax=docker/dockerfile:1

########## base: primary env + package ##########
FROM mambaorg/micromamba:1.5.8 AS base
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /opt/cellquorum
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/cellquorum-core.yml /tmp/core.yml
RUN micromamba create -y -n cellquorum-core -f /tmp/core.yml && micromamba clean -a -y
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/cellquorum
RUN micromamba run -n cellquorum-core pip install -e .

########## backends: five isolated envs ##########
FROM base AS backends
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/celloracle_env.yml /tmp/celloracle_env.yml
RUN micromamba create -y -n celloracle_env -f /tmp/celloracle_env.yml && micromamba clean -a -y
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

########## cpu: base + backends + R env ##########
FROM backends AS cpu
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/cellquorum-r.yml /tmp/r.yml
RUN micromamba create -y -n cellquorum-r -f /tmp/r.yml && micromamba clean -a -y
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "micromamba", "run", "-n", "cellquorum-core", "cellquorum"]
CMD ["--help"]

########## gpu: cpu + gpu env ##########
FROM cpu AS gpu
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/cellquorum-gpu.yml /tmp/gpu.yml
RUN micromamba create -y -n cellquorum-gpu -f /tmp/gpu.yml && micromamba clean -a -y
