# Running sc-SHC through CellQuorum

Use the QC-eligible lineage subset. CHOIR can partition ambient-corrected expression,
but sc-SHC must receive original, finite, non-negative integer counts. Do not round
fractional corrected values to make this check pass. Do not use normalized or MAGIC
values for the count-model significance test.

```yaml
subclustering:
  counts_layer: counts_soupx
  formal_test:
    method: scshc
    counts_layer: counts_raw
    alpha: 0.05
    random_state: 1337
    timeout_seconds: 1800
```

If formal_test.counts_layer is unset, the wrapper prefers counts_raw when present,
otherwise validates subclustering.counts_layer. Cell barcodes must align to cluster
labels and batch values. The R wrapper drops all-zero genes and caps features/PCs
to the available dimensions. Missing batch labels are an error. Keep the actual
batch covariate; do not remove it just to obtain a successful result.

Package availability can produce an explicit skip. A started sc-SHC test that fails
now raises a backend error and saves scshc_stdout.log and scshc_stderr.log. Candidate
CHOIR labels are not validated if the test fails. Apply reconciled labels only after
successful aligned output; retain the original CHOIR labels as provenance.

The upstream batch-aware null model has a reported NaN/TridiagEigen failure:
https://github.com/igrabski/sc-SHC/issues/33. This report is not proof of the cause
in a particular dataset. Sparse within-batch expression and model variance estimates
need diagnosis if the error persists on valid integer counts. Do not silently clamp
variance, alter scSHC internals, pool batches, or loosen alpha to produce a preferred
cluster count. Record package versions, source hashes, count-layer provenance,
batch sizes, seed, candidate labels, and the exact error for each attempt.

Verification: input-safety, partition, and configuration tests passed, including
the existing real-R tests. The corrected 6,888-cell KC biological test is separate;
passing software checks does not certify its result.
