CellRank runtime repairs

Hard pseudotime thresholding can strand rows in a sparse neighbor graph. CellQuorum retains the hard-threshold result when it succeeds. Only the specific non-row-stochastic ValueError triggers a fresh PseudotimeKernel with the supported soft threshold scheme. It uses existing neighbor edges and reports the change in method notes. Other failures remain failures. This is a methodological fallback, not a claim that the resulting trajectory is biologically correct.

CellRank 2.1 Lineage H5AD adapters call an AnnData write_basic keyword no longer accepted by the installed AnnData. The shared CellQuorum sanitizer converts only CellRank lineage matrices in obsm/varm to ordinary ndarrays while saving original column names and colors in uns/cellrank_lineage_array_metadata. Numerical values are unchanged. Consumers requiring a Lineage object may reconstruct it explicitly from this metadata.

Regression tests: tests/test_cellrank_runtime_safety.py, existing core/test_h5ad_io.py and trajectory kernel/save tests. Real-data verification: le_il33_raw_rebuild_20261001/runs/kc_trajectory_GRN_20261003/cellrank_repaired_v4. Previous run preserved.

Macrostate representative selection retries native n_cells values 30, 20, 10, and 5 only for the specific discretization confidence error. Other failures propagate. This retains native checks but changes representative selection and its associated confidence threshold; it does not prove biological state validity. The real-data run succeeded with 10 representatives.

Verified on 5,852 KCs: three fate columns, finite nonnegative probabilities, maximum row-sum error 9.55e-15. Fresh GPCCA pickle and fate-mapping H5AD reopen successfully. Inferred terminal labels include Basal_2, Granular/late-differentiating, and Transitional differentiating_2; these require biological review before interpreting terminal fates. Forty-five targeted regression tests passed.
