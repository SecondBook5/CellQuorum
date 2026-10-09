# Paired MultiNicheNet

CellQuorum inherits `design.paired` and the resolved cohort donor key for MultiNicheNet. A paired run exports the donor column and passes it as a categorical covariate to multinichenetr. Its muscat pseudobulk DE fits `~0 + condition + donor`, with the configured case-minus-control contrast.

Eligibility requires exactly one eligible sample per donor and condition. Every selected cell type must have at least `cell_cell_communication.nichenet_min_cells` (default 10) in both samples. This conservative shared cohort prevents categorical donor levels from disappearing for individual cell types inside muscat. Donors failing that rule are excluded only from this method, not from the shared atlas. At least three complete pairs are required. Multiple samples per donor-condition need explicit aggregation first; they are not silently treated as independent replicates.

Set `nichenet_counts_layer: counts_raw` to export original integer counts rather than the AnnData X matrix. Unpaired callers retain the legacy X behavior when this option is unset. LIANA continues using its separately configured normalized layer.

The scratch `mnn_pairing_audit.csv` lists all samples and inclusion flags. `results/mnn_prioritization.csv.design.csv` records the full-rank donor-adjusted design checked before fitting; both are registered artifacts. Missing donor metadata, conflicting sample assignments, too few pairs, or a rank-deficient design fail instead of falling back to an unpaired model.

Donor-adjusted ligand/target DE does not turn inferred communication or a prior ligand-target link into experimentally demonstrated signaling. Group priorities remain network inference; sample-level evidence and receptor expression require inspection.

Regression coverage: method tests verify donor export, paired exclusions, counts-layer export, missing-donor rejection, and unchanged unpaired execution. Stage tests verify cohort donor inheritance and explicit overrides.
