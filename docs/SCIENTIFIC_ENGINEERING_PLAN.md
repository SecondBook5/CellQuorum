# CellQuorum — Finalized Scientific and Engineering Plan

## Final decision

CellQuorum should **not** be defined as a workflow organizer, and it should **not** be rewritten wholesale in Rust.

It should be:

> **A Python-native, GPU-capable, continuously validated scientific framework for single-cell RNA-seq analysis, with original methods for evidence adjudication, dynamic regulatory systems analysis, and mechanistic evidence integration, plus an optional config-driven execution engine for complete reproducible analyses.**

The distinction is important:

```text
CellQuorum scientific package
│
├── reusable analysis APIs
├── native scientific methods
├── standardized result objects
├── continuous diagnostics
├── evidence-gated biological interpretation
├── GPU-capable computational primitives
├── adapters to mature external methods
│
└── optional workflow engine
      ├── YAML configuration
      ├── automatic planning
      ├── dependency routing
      ├── end-to-end execution
      ├── checkpointing
      └── provenance
```

The workflow engine is a **feature of the package**. It is not the package’s scientific identity.

The existing repository already provides the right execution foundation: strict configuration validation, backend discovery, strategy-based method dispatch, fail-loud AnnData contracts, standardized run outputs, provenance, R support, GPU routing, and executable analysis stages.

CellQuorum now needs to grow upward from that infrastructure into a scientific framework.

---

# Part I — What CellQuorum uniquely owns

CellQuorum becomes its own thing by owning capabilities that no individual wrapped package provides.

## 1. Continuous analytical validation

Quality control must occur throughout the analysis, not only before normalization.

Every module follows:

```text
Validate inputs
       ↓
Check scientific eligibility
       ↓
Run primary method
       ↓
Generate quantitative diagnostics
       ↓
Run configured robustness analyses
       ↓
Apply validity gate
       ↓
PASS / WARN / FAIL / SKIP
       ↓
Release validated output downstream
```

Every result carries:

```python
result.status
result.metrics
result.diagnostics
result.warnings
result.failures
result.artifacts
result.provenance
result.input_fingerprint
result.method_version
result.backend
result.device
```

A visually attractive output never substitutes for diagnostics.

---

## 2. Evidence-gated cellular taxonomy

CellQuorum will not equate a Leiden cluster with a biological cell state.

It will combine:

```text
candidate partition
        ↓
technical validity
        ↓
hierarchical structural support
        ↓
algorithmic stability
        ↓
donor replication
        ↓
held-out donor generalization
        ↓
molecular coherence
        ↓
discrete-versus-continuous assessment
        ↓
external reference support
        ↓
CellQuorum biological classification
```

Final classifications:

```text
validated_identity

reproducible_state

condition_restricted_state

rare_replicated_population

trajectory_associated_state

continuous_program

donor_restricted_population

ambiguous_population

unsupported_split

technical_population
```

That decision framework is CellQuorum-native even when CHOIR, scICE, scDiagnostics, or other packages generate evidence.

---

## 3. Unified biological evidence objects

External methods return incompatible outputs.

CellQuorum should convert them into standardized biological evidence types:

```text
CellIdentityEvidence

CellStateEvidence

AbundanceEvidence

GeneEffectEvidence

ProgramEvidence

PathwayEvidence

TFActivityEvidence

RegulonEvidence

GRNEdgeEvidence

TrajectoryEvidence

VelocityEvidence

FateEvidence

LigandReceptorEvidence

LigandTargetEvidence

MulticellularProgramEvidence

PhenotypeAssociationEvidence

PerturbationEvidence
```

Every evidence object records:

```text
source method

biological entity

direction

effect magnitude

uncertainty

statistical support

donor support

condition specificity

robustness status

observed versus inferred status

provenance
```

---

## 4. Dynamic regulatory systems analysis

CellQuorum should own the integration of:

```text
developmental potential
        ↓
trajectory
        ↓
velocity
        ↓
fate
        ↓
context-specific GRN
        ↓
dynamic GRN rewiring
        ↓
network topology
        ↓
dynamic curvature
        ↓
in silico perturbation
        ↓
predicted state and fate consequences
```

This is not currently one coherent package.

---

## 5. Mechanistic evidence graphs

The signature CellQuorum output should connect:

```text
sender identity
        ↓
sender state
        ↓
ligand
        ↓
receptor
        ↓
receiver pathway
        ↓
transcription factor
        ↓
regulon
        ↓
context-specific GRN
        ↓
target program
        ↓
receiver state
        ↓
trajectory and fate
        ↓
multicellular phenotype
        ↓
clinical or experimental phenotype
```

This is where CellQuorum becomes more than a collection of analyses.

---

# Part II — Product architecture

## Layer 1: scientific library

Public Python API:

```python
import cellquorum as cq

cq.pp.qc(...)
cq.pp.normalize(...)
cq.pp.select_features(...)

cq.tl.integrate(...)
cq.tl.annotate(...)
cq.tl.taxonomy(...)
cq.tl.differential_state(...)
cq.tl.composition(...)
cq.tl.programs(...)
cq.tl.trajectory(...)
cq.tl.velocity(...)
cq.tl.fate(...)

cq.grn.infer(...)
cq.grn.dynamic(...)
cq.grn.perturb(...)

cq.ccc.infer(...)
cq.ccc.programs(...)

cq.diag.evaluate(...)

cq.evidence.build(...)
```

Users can call individual modules in notebooks without running an entire pipeline.

---

## Layer 2: standardized method adapters

Every method implements a shared interface.

Conceptually:

```python
class AnalysisMethod:

    def validate_inputs(self, context):
        ...

    def check_eligibility(self, context):
        ...

    def estimate_resources(self, context):
        ...

    def run(self, context):
        ...

    def diagnose(self, result):
        ...

    def export_evidence(self, result):
        ...
```

External methods become interchangeable through standardized result objects.

Example:

```python
palantir_result = cq.tl.trajectory(
    adata,
    method="palantir",
)
```

and:

```python
slingshot_result = cq.tl.trajectory(
    adata,
    method="slingshot",
)
```

both return a CellQuorum `TrajectoryResult`.

---

## Layer 3: continuous diagnostics plane

Diagnostics are independent of method execution.

```text
Analysis layer
      │
      ├── input QC
      ├── numerical QC
      ├── model QC
      ├── biological QC
      ├── robustness QC
      └── downstream eligibility
```

---

## Layer 4: optional execution engine

The existing planner remains useful.

```bash
cellquorum plan --config project.yaml

cellquorum run --config project.yaml
```

The current repository already exposes both CLI and Python execution and uses strict configuration models.

The engine should:

* compose modules;
* resolve dependencies;
* select backends;
* estimate resources;
* checkpoint outputs;
* skip ineligible methods with a reason;
* resume interrupted analyses;
* record complete provenance.

---

# Part III — Canonical end-to-end workflow

```text
0. Experimental design
          ↓
1. Input ingestion and manifest validation
          ↓
2. Cell calling
          ↓
3. Ambient RNA assessment and correction
          ↓
4. Sample-aware QC
          ↓
5. Doublet detection
          ↓
6. PFlog1pPF normalization
          ↓
7. Feature selection
          ↓
8. Unintegrated representation
          ↓
9. Conditional integration
          ↓
10. Integration diagnostics and validity gate
          ↓
11. Broad cell discovery
          ↓
12. Generalized reference mapping
          ↓
13. Annotation
          ↓
14. Annotation diagnostics
          ↓
15. Lineage-specific reanalysis
          ↓
16. Principled cellular taxonomy
          ↓
17. Rare-state analysis
          ↓
18. Discrete-versus-continuous determination
          ↓
19. Metacell construction
          ↓
20. Differential abundance
          ↓
21. Donor-aware differential state
          ↓
22. Phenotype-linked discovery
          ↓
23. Gene programs
          ↓
24. Pathway activity
          ↓
25. Developmental potential
          ↓
26. Optional topology-aware signal decomposition
          ↓
27. Trajectory topology and pseudotime
          ↓
28. RNA velocity
          ↓
29. Fate inference
          ↓
30. Dynamic expression analysis
          ↓
31. TF activity
          ↓
32. Regulon inference
          ↓
33. Context-specific GRNs
          ↓
34. Dynamic GRNs
          ↓
35. Network topology and curvature
          ↓
36. In silico perturbation
          ↓
37. LIANA+ → Tensor-cell2cell
          ↓
38. MultiNicheNet
          ↓
39. DIALOGUE
          ↓
40. FlowSig
          ↓
41. Mechanistic evidence graph
          ↓
42. Publication report and provenance archive
```

Many downstream modules branch and run in parallel after validated cellular taxonomy.

---

# Part IV — Detailed scientific modules

# 0. Experimental-design intelligence

The framework begins with metadata.

Required concepts:

```text
patient

donor

animal

biological replicate

sample

library

capture

condition

treatment

time point

paired structure

longitudinal structure

tissue

center

sequencing run

chemistry

technical replicate

biological covariates

technical covariates

primary contrast

secondary contrasts
```

Validation detects:

* duplicated identifiers;
* missing values;
* impossible pairs;
* missing repeated measurements;
* batch–condition confounding;
* treatment–center confounding;
* insufficient biological replication;
* invalid contrasts;
* group imbalance;
* non-estimable model terms;
* cell types occurring in too few donors;
* pseudo-replication risk.

Outputs:

```text
inferential

descriptive

exploratory

not_estimable

confounded
```

No inferential module runs before the design gate passes.

---

# 1. Input ingestion and data contracts

Supported inputs:

* `.h5ad`;
* Cell Ranger outputs;
* count matrices;
* loom where needed;
* compatible AnnData objects;
* future additional formats through adapters.

Canonical layers:

```text
raw_counts

ambient_corrected_counts

pflog1ppf

conventional_log1p

analytic_pearson_residuals

spliced

unspliced
```

CellQuorum validates:

* count integerness;
* non-negativity;
* sparsity;
* duplicate genes;
* duplicate barcodes;
* gene identifiers;
* species;
* layer semantics;
* matrix orientation;
* missing metadata;
* NaN and infinity values.

Raw counts are immutable.

---

# 2. Cell calling

Activation:

```text
raw droplet matrix available
```

Primary:

* CellBender.

Sensitivity:

* emptyDrops.

Baseline:

* Cell Ranger calls.

Preserve:

* original calls;
* revised calls;
* rescued cells;
* rejected droplets;
* confidence;
* method disagreement.

Diagnostics:

* barcode-rank curve;
* UMI distribution;
* ambient profile;
* recovered low-RNA populations;
* sample-level retention.

---

# 3. Ambient RNA

Primary production method:

* SoupX.

Alternatives:

* CellBender;
* decontX.

Assessment should always be available.

Correction should be evidence-driven.

Diagnostics:

* inferred soup profile;
* contamination fraction;
* genes most removed;
* cell types most affected;
* pre/post marker specificity;
* pre/post cross-lineage contamination;
* overcorrection risk;
* preservation of true biological expression.

The existing repository already implements SoupX through the staged execution system.

---

# 4. Sample-aware QC

Metrics:

```text
total counts

detected genes

mitochondrial fraction

ribosomal fraction

hemoglobin fraction

complexity

ambient contamination

doublet score

stress score

dissociation score

sample-specific outlier score
```

Methods:

* MAD-based adaptive thresholds;
* configurable hard thresholds;
* sample-specific modeling;
* biological exception rules.

Outputs:

* retention waterfall;
* threshold table;
* pre/post distributions;
* exclusion table;
* exclusion reasons;
* sample-level summaries.

No universal mitochondrial threshold should be treated as biologically correct for every tissue.

---

# 5. Doublet analysis

Primary:

* scDblFinder.

Consensus:

* Scrublet.

Optional sensitivity:

* SOLO.

Optional compatibility:

* DoubletFinder.

Run per capture or library.

Integrate:

* score;
* expected rate;
* mixed-lineage markers;
* neighborhood structure;
* annotation conflict;
* genotype information when present.

Classes:

```text
high_confidence_doublet

probable_doublet

possible_homotypic_doublet

discordant_prediction

likely_singlet
```

Automatic removal only for high-confidence cases by default.

---

# 6. PFlog1pPF normalization

Canonical cell-level representation:

```text
PFlog1pPF
```

Implementation:

* CPU reference;
* CuPy GPU implementation;
* sparse execution.

Sensitivity representations:

* conventional library-size log1p;
* analytic Pearson residuals.

Required parity:

* absolute error;
* relative error;
* rank preservation;
* PCA concordance;
* neighborhood concordance;
* marker concordance;
* cluster concordance.

Raw counts remain the basis for count-based inference.

The current repository already routes PFlog1pPF through a CuPy implementation when GPU capability is present.

---

# 7. Feature selection

Methods:

* batch-aware HVG;
* deviance-based selection;
* PFlog1pPF variance selection after validation;
* Pearson-residual feature selection.

Separate feature sets:

| Feature set        | Purpose                     |
| ------------------ | --------------------------- |
| Global             | broad cellular structure    |
| Batch-aware global | multi-sample representation |
| Lineage-specific   | subtype discovery           |
| State-sensitive    | activation programs         |
| Regulatory         | GRN inference               |
| Full gene set      | pseudobulk and pathways     |

Features are relearned after lineage subsetting.

Do not automatically remove:

* cell-cycle genes;
* interferon genes;
* stress genes;
* mitochondrial biology;
* ribosomal biology.

---

# 8. Unintegrated representation

Always retain:

```text
X_pca_unintegrated
```

The unintegrated representation is the biological reference against which corrected representations are judged.

GPU operations:

* PCA;
* neighbors;
* UMAP;
* repeated parameter runs.

---

# 9. Conditional integration

Supported:

* Harmony;
* scVI;
* scPoli.

Reference-oriented:

* scANVI;
* scArches.

Recommended routing:

| Dataset                         | Preferred method |
| ------------------------------- | ---------------- |
| modest simple technical batches | Harmony          |
| large complex cohort            | scVI             |
| population-aware integration    | scPoli           |
| query/reference mapping         | scANVI/scArches  |

Biological condition must not be removed automatically.

---

# 10. Integration diagnostics and gate

Batch-removal metrics:

* kBET;
* iLISI;
* batch silhouette;
* graph connectivity;
* PC regression.

Biological-conservation metrics:

* cLISI;
* cell-identity silhouette;
* marker preservation;
* HVG preservation;
* neighborhood preservation;
* isolated-label preservation;
* rare-state preservation;
* condition preservation;
* trajectory preservation.

Evaluate:

```text
global

per lineage

per cell type

per condition

per donor

across seeds

across subsamples
```

Store both raw and standardized metrics.

Example:

```text
raw cLISI

standardized identity-conservation score
```

Use Pareto selection.

No single integration score selects the winner.

---

# 11. Broad cellular discovery

GPU-accelerated:

* PCA;
* neighbors;
* Leiden;
* UMAP.

Leiden creates candidate structure.

UMAP is visualization.

Neither defines biological truth.

---

# 12. Generalized scArches reference mapping

scArches must be atlas-agnostic.

Modes:

```text
load reference

train reference

update reference

map query
```

Reference types:

* user-provided;
* public atlas;
* laboratory atlas;
* CellQuorum model registry.

Validate:

* species;
* gene identifiers;
* shared features;
* feature ordering;
* normalization;
* count layers;
* model class;
* labels;
* category mappings.

Allow:

```text
known label

uncertain label

unknown label

novel candidate
```

---

# 13. Deep-model diagnostics

For:

* scVI;
* scANVI;
* scArches;
* scPoli;
* SOLO;
* future neural models.

Record:

```text
training loss

validation loss

ELBO

reconstruction loss

KL divergence

classifier loss

adversarial loss

learning rate

epoch

best epoch

early stopping
```

Diagnostics:

* convergence;
* divergence;
* plateau;
* overfitting;
* undertraining;
* train–validation gap;
* latent collapse;
* seed stability;
* latent-dimension sensitivity;
* runtime;
* GPU memory.

Loss curves are mandatory report artifacts.

---

# 14. Hierarchical annotation

Evidence sources:

* marker vote;
* positive markers;
* negative markers;
* CellTypist;
* SingleR;
* scANVI/scArches;
* tissue-specific references;
* expert review;
* de novo structure.

Statuses:

```text
validated

confident

probable

ambiguous

mixed

novel_candidate

unresolved
```

The current marker-vote implementation remains a transparent baseline.

---

# 15. scDiagnostics annotation validation

scDiagnostics becomes a primary validation backend rather than merely a plotting package.

It supports assessment of annotation reliability, query/reference alignment, marker behavior, anomaly detection, and QC relationships.

Use:

* PCA comparison;
* PCA-subspace comparison;
* MDS;
* discriminant spaces;
* graph integration;
* Wasserstein distance;
* MMD;
* Hotelling (T^2);
* Cramér tests;
* pairwise correlation;
* marker overlap;
* HVG overlap;
* variable-importance overlap;
* gene shifts;
* anomaly detection;
* categorization entropy;
* QC-versus-annotation plots.

Outputs:

```text
reference_aligned

reference_shifted

anomalous_query_state

ambiguous

novel_candidate

unsupported
```

---

# 16. Lineage-specific reanalysis

For every broad lineage:

1. select high-confidence cells;
2. recalculate features;
3. recompute PCA or latent representation;
4. rebuild neighbors;
5. generate candidate partitions;
6. rerun annotation evidence;
7. evaluate discrete versus continuous structure.

Global PCA should not be the sole basis for fine subtypes.

---

# 17. Principled cellular taxonomy

Primary:

* CHOIR.

Stability:

* repeated graph perturbation;
* repeated random seeds;
* feature perturbation;
* PC perturbation;
* neighborhood perturbation;
* cell subsampling;
* donor subsampling;
* scICE.

Reconciliation:

* scTriangulate.

Contested branches:

* sc-SHC;
* Cytocipher;
* recall;
* ClusterDE;
* SCCAF-style classification.

Validation dimensions:

```text
technical validity

structural support

stability

donor replication

molecular coherence

held-out generalization

geometry

external evidence
```

Fatal failures cannot be averaged away.

---

# 18. Rare-state analysis

Methods:

* CellSIUS;
* RareQ.

Activation:

* expected rare population;
* unexplained local topology;
* coherent rare marker program;
* suspected masking by broad clustering.

Required evidence:

* not ambient;
* not doublet-driven;
* donor replicated;
* molecularly coherent;
* generalizable when possible.

---

# 19. Discrete-versus-continuous decision

Possible interpretations:

```text
discrete identity
        → taxonomy

continuous program
        → cNMF / GeneNMF / LEMUR

local abundance shift
        → Milo

developmental transition
        → CytoTRACE2 / Palantir / scVelo / CellRank
```

A cell can possess:

```yaml
identity: basal_epithelial

programs:

  inflammatory: 0.81

  repair: 0.62

  interferon: 0.08
```

Continuous biology should not be forced into arbitrary clusters.

---

# 20. Metacells

Primary:

* SEACells.

Optional:

* MetaQ.

Uses:

* coexpression;
* cNMF;
* SCENIC;
* GRNs;
* topology;
* communication;
* visualization.

Metacells never become artificial biological replicates.

---

# 21. Differential abundance

## Compositional branch

Primary configurable options:

* scCODA;
* sccomp.

Sensitivity:

* propeller.

scCODA should be a first-class Bayesian compositional method rather than a minor optional check.

scCODA diagnostics:

* sample totals;
* zero prevalence;
* rare-population prevalence;
* reference-cell-type choice;
* reference sensitivity;
* posterior convergence;
* effective sample size;
* posterior inclusion;
* credible effects;
* posterior predictive checks.

Rigorous mode repeats plausible reference choices.

Classify:

```text
robust

reference_sensitive

rare_state_sensitive

unstable

unsupported
```

## Neighborhood branch

Primary:

* Milo.

This detects local abundance shifts that do not respect hard cluster boundaries.

---

# 22. Donor-aware differential state

Primary Python-native option:

* edgePython.

Reference validation:

* edgeR.

Complex repeated designs:

* dreamlet.

Additional:

* muscat;
* LEMUR;
* scDist.

edgePython now provides TMM normalization, dispersion estimation, GLM fitting, quasi-likelihood tests, likelihood-ratio tests, TREAT-style testing, gene-set testing, pseudobulk support, and a multi-subject single-cell mixed model.

Initial policy:

```text
edgePython primary

edgeR parity validation

dreamlet for complex repeated designs
```

Required outputs:

* volcano;
* effect-size table;
* donor-level plots;
* MA plot;
* dispersion diagnostics;
* model fit;
* confidence intervals;
* pathway results;
* TF activity;
* ligand–target interpretation;
* sensitivity comparison.

---

# 23. Phenotype-linked cell discovery

## Scissor

Activation requires:

```text
single-cell expression

bulk expression cohort

sample-level phenotype
```

Phenotypes:

* response;
* disease status;
* treatment benefit;
* toxicity;
* continuous outcomes;
* survival where supported.

Outputs:

```text
Scissor-positive cells

Scissor-negative cells

phenotype-neutral cells
```

Downstream:

```text
cell identities

cell states

programs

TF activity

GRNs

trajectory

fate

CCC roles
```

Diagnostics:

* cross-validation;
* held-out bulk samples;
* leakage detection;
* covariate review;
* donor composition;
* parameter sensitivity;
* cell-selection bootstrap;
* external validation.

---

# 24. Gene programs

Primary:

* cNMF.

Sensitivity:

* GeneNMF.

Program classes:

```text
identity

activation

stress

cell cycle

condition associated

trajectory associated

fate associated

multicellular

uncertain
```

Diagnostics:

* rank selection;
* reconstruction error;
* stability across runs;
* program redundancy;
* donor distribution;
* condition dependence;
* gene coherence.

---

# 25. Pathways

Methods:

* fgsea;
* Hallmark;
* Reactome;
* Gene Ontology;
* PROGENy;
* UCell;
* AUCell;
* decoupler.

Keep distinct:

```text
gene expression

gene-set enrichment

pathway activity

TF activity

regulon activity
```

---

# 26. Developmental potential

## CytoTRACE2

Outputs:

* developmental-potential score;
* potency class;
* donor distributions;
* condition shifts;
* potency-associated genes;
* plasticity-associated programs.

Use for:

* root evidence;
* progenitor identification;
* differentiation;
* dedifferentiation;
* plasticity.

Root selection combines:

```text
CytoTRACE2

known biology

markers

Palantir

scVelo when valid
```

---

# 27. Optional topology-aware decomposition

## scPrisma

Status:

```text
optional

explicitly activated

hypothesis-driven
```

Uses:

* reconstruct topology;
* enhance topology;
* remove topology.

Topologies:

* cyclic;
* linear;
* ordered;
* custom covariance templates.

Applications:

* cell cycle;
* circadian programs;
* developmental axes;
* topology-specific GRNs;
* topology-specific CCC;
* testing whether clusters persist after removing a known process.

Never overwrite the original matrix.

Store:

```text
scprisma_enhanced

scprisma_filtered

scprisma_ordering
```

Diagnostics:

* spectral agreement;
* convergence;
* ordering stability;
* donor consistency;
* known marker agreement;
* signal enhancement;
* off-target distortion;
* preservation of identity;
* preservation of condition biology.

---

# 28. Trajectory topology

Primary expression-only method:

* Palantir.

Supporting:

* PAGA;
* Slingshot.

Palantir outputs:

* pseudotime;
* branch probabilities;
* terminal probabilities;
* differentiation potential;
* entropy;
* gene trends.

PAGA:

* coarse topology.

Slingshot:

* trajectory sensitivity.

---

# 29. RNA velocity

Primary:

* scVelo dynamical mode.

Activation requires:

```text
spliced layer

unspliced layer
```

Eligibility does not imply validity.

Diagnostics:

* unspliced coverage;
* velocity-gene count;
* phase portraits;
* dynamical likelihood;
* latent-time coherence;
* direction consistency;
* gene-filter sensitivity;
* donor consistency;
* Palantir agreement;
* biological plausibility.

Statuses:

```text
validated_direction

supported_direction

weak_signal

conflicting_signal

kinetic_failure

not_available
```

---

# 30. Fate inference

Primary:

* CellRank 2.

With valid velocity:

```text
velocity kernel

+

connectivity kernel

+

Palantir information
```

Without velocity:

```text
Palantir

+

connectivity

+

CytoTRACE2 orientation
```

Outputs:

* initial macrostates;
* terminal macrostates;
* fate probabilities;
* absorption probabilities;
* lineage drivers;
* fate entropy;
* fate-associated programs.

---

# 31. Dynamic expression

Methods:

* tradeSeq;
* Palantir trends;
* CellRank lineage drivers;
* condiments.

Questions:

* which genes change along pseudotime;
* which differ by branch;
* which precede commitment;
* which are transient;
* which trajectories differ by condition.

Diagnostics:

* donor reproducibility;
* branch support;
* knot sensitivity;
* trajectory-method sensitivity;
* condition balance.

---

# 32. TF activity

Primary:

* CollecTRI through decoupler.

Estimators:

* ULM;
* MLM;
* consensus.

GPU acceleration when validated and beneficial.

Outputs:

* activity;
* direction;
* condition effect;
* state specificity.

---

# 33. Regulons

Primary:

* SCENIC.

Implementation:

* pySCENIC;
* SCENIC workflow;
* metacell-assisted inference.

Outputs:

* TF–target regulons;
* regulon activity;
* specificity;
* condition dependence;
* lineage dependence.

Interpretation:

| Pattern            | Meaning                  |
| ------------------ | ------------------------ |
| SCENIC + CollecTRI | convergent               |
| SCENIC only        | dataset-derived regulon  |
| CollecTRI only     | prior-supported activity |
| TF expression only | weak                     |
| discordant         | requires review          |

---

# 34. Context-specific GRNs

Primary:

* CellOracle.

Current scRNA-only mode:

```text
expression

+

promoter prior

+

cell-state context

→

directed context-specific GRN
```

Outputs:

* TF–target edges;
* edge coefficients;
* signs;
* regulator importance;
* state-specific networks;
* perturbation vectors.

RNA-only promoter-informed networks must be labeled as such.

---

# 35. Dynamic GRNs

Construct networks across:

* Palantir pseudotime;
* CellRank lineages;
* scVelo latent time;
* branches;
* conditions;
* treatment states.

Ordering priority:

```text
validated scVelo latent time

then

CellRank lineage ordering

then

Palantir pseudotime
```

Measure:

* edge gain;
* edge loss;
* weight change;
* sign change;
* hub change;
* module rewiring;
* feedback changes;
* feed-forward changes;
* entropy;
* controllability;
* spectral change;
* attractor stability.

Research sensitivity:

* SINGE;
* SCODE;
* Scribe-style velocity-informed approaches.

---

# 36. Network topology

Metrics:

* in-degree;
* out-degree;
* weighted degree;
* betweenness;
* PageRank;
* eigenvector centrality;
* hubness;
* communities;
* modularity;
* motifs;
* entropy;
* controllability;
* spectral properties.

---

# 37. Dynamic curvature

Experimental CellQuorum-native layer.

Candidate metrics:

* Ollivier–Ricci curvature;
* Forman–Ricci curvature;
* directed extensions;
* signed extensions.

Dynamic edge curvature:

[
\Delta\kappa_e
==============

\kappa_e(t_2)-\kappa_e(t_1)
]

Potential interpretation:

* bottlenecks;
* fragile transitions;
* stabilizing hubs;
* branch reorganization;
* network deformation;
* intervention targets.

Mandatory controls:

* edge bootstrap;
* donor bootstrap;
* window sensitivity;
* GRN-method sensitivity;
* degree-preserving nulls;
* density-matched nulls;
* weight permutation;
* sign-preserving nulls;
* sparsification sensitivity.

Curvature remains experimental until biologically benchmarked.

---

# 38. In silico perturbation

Primary:

* CellOracle.

Sensitivity:

* scTenifoldKnk;
* GenKI.

Workflow:

```text
validated GRN
       ↓
target regulator
       ↓
virtual knockout
       ↓
edge changes
       ↓
target genes
       ↓
program changes
       ↓
topology changes
       ↓
curvature changes
       ↓
state displacement
       ↓
trajectory displacement
       ↓
fate changes
```

Outputs are hypotheses, not experimental proof.

---

# 39. LIANA+ → Tensor-cell2cell

These should be one integrated multisample CCC pipeline.

The published workflow was designed so LIANA communication scores pass directly into Tensor-cell2cell for context-dependent decomposition.

Workflow:

```text
normalized expression
        ↓
per-sample LIANA+
        ↓
consensus LR magnitude scores
        ↓
LIANA-to-cell2cell conversion
        ↓
4D tensor
        ↓
Tensor-cell2cell
        ↓
contextual CCC programs
```

Tensor:

[
\text{context}
\times
\text{ligand–receptor}
\times
\text{sender}
\times
\text{receiver}
]

LIANA provides:

* method selection;
* resource selection;
* consensus LR inference;
* sample-level communication scores.

Tensor-cell2cell provides:

* context factors;
* LR factors;
* sender factors;
* receiver factors.

The coupled protocol preserves higher-order relationships across samples, interactions, senders, and receivers.

Diagnostics:

* tensor coverage;
* missingness;
* rank selection;
* reconstruction error;
* optimization loss;
* convergence;
* factor stability;
* initialization stability;
* sample-loading stability;
* donor influence;
* leave-one-sample-out stability.

---

# 40. MultiNicheNet

Question:

> Which sender ligands explain receiver transcriptional changes?

Required:

* sample replication;
* ligand expression;
* receptor expression;
* differential ligand expression;
* differential receptor expression;
* ligand activity;
* receiver targets;
* sender specificity;
* condition specificity.

Outputs:

* prioritized ligands;
* LR pairs;
* ligand activity;
* ligand–target links;
* sender–receiver networks.

---

# 41. DIALOGUE

Question:

> Which coordinated programs span multiple populations?

Output example:

```text
myeloid inflammation

↕

fibroblast remodeling

↕

endothelial activation

↕

T-cell dysfunction
```

No known LR pair is required.

---

# 42. FlowSig

Question:

> How are incoming signals, intracellular programs, and outgoing signals related?

```text
incoming signal
       ↓
intracellular program
       ↓
outgoing signal
```

Activation:

* disease/reference;
* treatment;
* perturbation;
* longitudinal;
* sufficiently replicated contexts.

Do not label as causal proof.

---

# 43. Mechanistic evidence graph

Example:

```text
Macrophage inflammatory state
             ↓
IFNG
             ↓
LIANA+ evidence
             ↓
IFNGR1 / IFNGR2
             ↓
MultiNicheNet ligand activity
             ↓
Fibroblast STAT1 activity
             ├── CollecTRI
             ├── SCENIC regulon
             ├── CellOracle GRN hub
             ├── pseudobulk DE
             └── cNMF inflammatory program
             ↓
CellRank inflammatory fate
             ↓
DIALOGUE multicellular program
             ↓
clinical phenotype

CellOracle knockout:
predicted reduction in inflammatory transition
```

Each edge stores:

```text
method

evidence type

direction

effect

uncertainty

p-value or posterior

donor replication

condition specificity

robustness

observed/inferred/predicted

provenance
```

Exports:

* Parquet;
* GraphML;
* Cytoscape;
* HTML;
* static publication network.

---

# Part V — Continuous diagnostics matrix

| Stage            | Mandatory diagnostics                                 |
| ---------------- | ----------------------------------------------------- |
| Input            | counts, sparsity, genes, cells, metadata completeness |
| Cell calling     | barcode rank, recovered cells, background profile     |
| Ambient          | contamination reduction, marker preservation          |
| QC               | sample distributions, thresholds, retention           |
| Doublets         | score agreement, lineage mixtures, donor distribution |
| Normalization    | numerical parity, rank preservation                   |
| Features         | stability, donor distribution, biology preservation   |
| Integration      | kBET, iLISI, cLISI, silhouettes, marker preservation  |
| scVI/scArches    | training/validation loss, ELBO, convergence           |
| Annotation       | markers, negative markers, scDiagnostics, entropy     |
| Clustering       | perturbation stability, donor replication             |
| Composition      | posterior convergence, reference sensitivity          |
| DE               | dispersion, fit, residuals, effect consistency        |
| Scissor          | cross-validation, leakage, selection stability        |
| Programs         | rank, reconstruction, stability                       |
| CytoTRACE2       | donor consistency, marker coherence                   |
| Palantir         | root sensitivity, branch stability                    |
| scVelo           | phase portraits, likelihood, latent-time coherence    |
| CellRank         | macrostate stability, fate robustness                 |
| SCENIC           | regulon stability, donor support                      |
| GRN              | edge bootstrap, donor bootstrap, sparsity sensitivity |
| Curvature        | graph nulls, density matching                         |
| Perturbation     | method sensitivity, GRN sensitivity                   |
| LIANA            | sample support, method/resource sensitivity           |
| Tensor-cell2cell | rank, loss, reconstruction, factor stability          |
| MultiNicheNet    | ligand-target support, sample replication             |
| DIALOGUE         | program stability and donor support                   |
| Evidence graph   | completeness, contradiction, evidence provenance      |

---

# Part VI — Compute architecture

## Public language

Python.

## Acceleration policy

> GPU acceleration whenever supported, validated, and computationally beneficial.

High-value GPU operations:

* QC on large matrices;
* PFlog1pPF;
* feature calculations;
* PCA;
* neighbors;
* Leiden;
* UMAP;
* integration;
* scVI;
* scANVI;
* scArches;
* scPoli;
* repeated robustness analyses;
* activity scoring;
* tensor decomposition;
* large GRNs;
* topology;
* bootstrap analysis.

Usually CPU:

* small pseudobulk models;
* edgePython;
* edgeR;
* dreamlet;
* modest mixed models;
* small Bayesian composition models.

GPU capability should never be forced when transfer overhead exceeds compute benefit.

---

## Backend registry

Each method declares:

```text
language

native or adapter

CPU support

GPU support

preferred device

memory estimator

determinism

required dependencies

fallback

parity status
```

---

## Rust

Do not rewrite the package in Rust.

Use Rust selectively for:

* sparse operations;
* graph traversal;
* graph comparison;
* motifs;
* spectral calculations;
* curvature;
* network alignment;
* evidence-graph traversal;
* high-performance serialization.

Expose Rust kernels through Python.

---

# Part VII — Native implementation policy

## Native immediately

CellQuorum should own:

* PFlog1pPF GPU implementation;
* data contracts;
* diagnostics;
* kBET/iLISI/cLISI interfaces;
* standardized silhouettes;
* donor representation;
* graph stability;
* taxonomy adjudication;
* evidence schemas;
* dynamic GRN comparison;
* topology;
* curvature;
* null models;
* mechanistic evidence graphs.

## Adapter first

Use existing implementations for:

* CHOIR;
* scICE;
* scDiagnostics;
* scCODA;
* Scissor;
* CytoTRACE2;
* Palantir;
* scVelo;
* CellRank;
* scPrisma;
* SCENIC;
* CellOracle;
* LIANA+;
* Tensor-cell2cell;
* MultiNicheNet;
* DIALOGUE;
* FlowSig.

## Selective future ports

Only recreate a method when:

* interoperability is a real bottleneck;
* performance is unacceptable;
* GPU changes feasibility;
* required diagnostics are inaccessible;
* the original implementation is unmaintained;
* CellQuorum requires fundamentally new behavior.

---

# Part VIII — Configuration model

Separate scientific profile from robustness.

Profiles:

```text
standard

publication

regulatory

trajectory

communication

perturbation

full
```

Robustness:

```text
fast

standard

rigorous

exhaustive
```

Example:

```yaml
run:

  profile: full

  robustness: rigorous
```

Method roles:

```text
primary

sensitivity

ensemble

benchmark

fallback

experimental

disabled
```

Activation:

```text
true

false

automatic

explicit
```

---

# Part IX — Standard result objects

```python
QCResult

NormalizationResult

FeatureSelectionResult

IntegrationResult

AnnotationResult

TaxonomyResult

CompositionResult

DifferentialStateResult

PhenotypeLinkResult

ProgramResult

PathwayResult

DevelopmentalPotentialResult

TopologySignalResult

TrajectoryResult

VelocityResult

FateResult

TFActivityResult

RegulonResult

GRNResult

NetworkTopologyResult

PerturbationResult

CommunicationResult

MulticellularProgramResult

EvidenceGraphResult
```

Every result contains:

```python
data

metrics

diagnostics

status

warnings

failures

artifacts

provenance

input_fingerprint

method

version

backend

device
```

---

# Part X — User-facing experience

## Notebook user

```python
import cellquorum as cq

adata = cq.read("dataset.h5ad")

qc = cq.pp.qc(
    adata,
    method="adaptive",
)

adata = cq.pp.normalize(
    adata,
    method="pflog1ppf",
)

integration = cq.tl.integrate(
    adata,
    method="scvi",
)

cq.diag.integration(
    integration,
)

taxonomy = cq.tl.taxonomy(
    adata,
    method="choir",
)

trajectory = cq.tl.trajectory(
    adata,
    method="palantir",
)

grn = cq.grn.infer(
    adata,
    method="celloracle",
)

ccc = cq.ccc.liana_tensor(
    adata,
)
```

## End-to-end user

```bash
cellquorum plan \
    --config project.yaml

cellquorum run \
    --config project.yaml
```

Both are first-class.

---

# Part XI — Outputs

Standard run:

```text
run/
├── config/
├── checkpoints/
├── figures/
├── logs/
├── models/
├── objects/
├── provenance/
├── reports/
├── results/
├── diagnostics/
├── evidence/
└── scratch/
```

Final products:

* validated `.h5ad`;
* checkpoint objects;
* tables;
* model objects;
* figures;
* HTML report;
* methods text;
* diagnostic dashboard;
* run manifest;
* environment lock;
* evidence graph;
* Cytoscape export;
* machine-readable provenance.

---

# Part XII — Implementation roadmap from the current repository

The current project already has the execution spine and seven implemented stages, with GPU routing for compatible operations.

## Phase A — stabilize current foundation

* reconcile README and actual execution behavior;
* finalize stage lifecycle records;
* finish artifact registration;
* stabilize config organization;
* preserve fail-loud contracts;
* strengthen CPU/GPU parity;
* create stable public result classes.

## Phase B — complete standard scRNA workflow

* feature selection;
* integration diagnostics;
* scArches;
* model loss curves;
* annotation diagnostics;
* scDiagnostics;
* reference mapping;
* improved reporting.

## Phase C — taxonomy

* lineage-specific analysis;
* CHOIR;
* graph perturbation;
* scICE;
* donor holdout;
* evidence matrix;
* rare states;
* discrete/continuous adjudication.

## Phase D — sample-aware statistics

* GPU pseudobulk aggregation;
* edgePython;
* edgeR parity;
* dreamlet;
* muscat;
* scCODA;
* sccomp;
* propeller;
* Milo;
* LEMUR;
* scDist.

## Phase E — programs and phenotype

* cNMF;
* GeneNMF;
* pathways;
* PROGENy;
* UCell;
* Scissor.

## Phase F — dynamics

* CytoTRACE2;
* scPrisma;
* Palantir;
* PAGA;
* Slingshot;
* scVelo;
* CellRank;
* tradeSeq;
* condiments.

## Phase G — regulatory systems

* CollecTRI;
* SCENIC;
* CellOracle;
* static GRNs;
* dynamic GRNs;
* donor bootstrap;
* network topology;
* curvature;
* null models;
* perturbation.

## Phase H — multicellular systems

* LIANA+;
* Tensor-cell2cell;
* MultiNicheNet;
* DIALOGUE;
* FlowSig.

## Phase I — evidence graph

* common evidence schema;
* biological entity graph;
* confidence;
* contradiction handling;
* provenance;
* GraphML;
* Cytoscape;
* interactive reports.

---

# Part XIII — What CellQuorum should not become

Not:

```text
a giant YAML file

a collection of wrappers

a forced GPU pipeline

a rewrite of every single-cell package

a Rust reimplementation of scverse

a UMAP generator

a majority-vote engine
```

Instead:

```text
a scientific library

a validation framework

a systems-biology inference framework

a regulatory topology platform

an evidence integration system

and an optional reproducible execution engine
```

---

# Final definition

> **CellQuorum is a Python-native, GPU-capable, continuously validated framework for single-cell RNA-seq systems biology. It integrates evidence-preserving preprocessing, principled cellular taxonomy, generalized reference mapping, donor-aware differential and compositional inference, phenotype-linked cell discovery, gene-program analysis, developmental potential, trajectory, RNA velocity, fate inference, transcription-factor activity, regulons, context-specific and dynamic gene-regulatory networks, network topology, in silico perturbation, cell–cell communication, multicellular program discovery, and mechanistic evidence integration through reusable scientific APIs and an optional config-driven execution engine.**

Its defining commitments are:

> Biological samples determine inferential validity.

> Cell states are not assumed to be discrete.

> Every major result carries diagnostics, robustness, and provenance.

> GPU acceleration is used whenever it materially improves computation.

> Mature methods are wrapped rather than rewritten without reason.

> New native methods are developed where CellQuorum adds scientific value.

> GRNs are foundational for topology, curvature, control, and perturbation.

> Communication, regulation, state, fate, and phenotype are connected through evidence rather than displayed as disconnected outputs.

> The workflow engine makes CellQuorum easy to run; the scientific library makes CellQuorum its own framework.
