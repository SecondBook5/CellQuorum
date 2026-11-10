# SCC communication ranking across dependency versions

The public paired squamous-cell carcinoma reference contains 18 libraries. Both environments scored the same 54,450 library-specific sender, receiver, ligand and receptor combinations. The median within-library Spearman correlation for magnitude was 0.666. A median of 94 of each library’s top 100 interactions remained in the top 100 (range 87–96). Specificity ranks were more stable, with a median correlation of 0.998 and median top-100 overlap of 100 (range 89–100).

![Within-library magnitude rank and top-100 retention for both communication scores](scc-rank-comparison.png)

The left panel includes every matched interaction measurement. A smaller percentile means greater inferred magnitude. Tied scores share their mean percentile. The diagonal marks agreement. On the right, each connected pair represents one library. Sorting tied top-100 scores by sender, receiver, ligand and receptor gives a reproducible cutoff.

## Which dependency changed the rankings?

Each run used the same normalized SCC object, LIANA resource and CellQuorum recipe. Three comparisons independently varied the libraries within the diagnostic environment. The first and last controls preserved every magnitude and specificity score exactly.

| Change | Fixed libraries | Median magnitude ρ | Median shared magnitude top 100 |
|---|---|---:|---:|
| AnnData 0.12.10 → 0.13.4 | pandas 2.3.3, LIANA 1.10.0 | 1.000 | 100 |
| LIANA 1.10.0 → 2.1.1 | pandas 2.3.3, AnnData 0.13.4 | 0.666 | 94 |
| pandas 2.3.3 → 3.1.1 | AnnData 0.13.4, LIANA 2.1.1 | 1.000 | 100 |

The inferred CD8 T cell HMGB1 to keratinocyte SDC1 interaction entered the top 100 in seven libraries with LIANA 2.1.1, up from four with 1.10.0. Across the same 18 libraries, the three optimally matched Tensor-cell2cell sample factors had Pearson correlations of 0.821, 0.973 and 0.958. Matching maximized total sample-loading correlation across factor permutations.

## Figure and source data

- [All matched magnitude scores and percentiles (gzip CSV)](scc-magnitude-interactions.csv.gz)
- [Rank correlations, shared top 100 and maximum score changes for each library and control](scc-library-ranks.csv)
- [Paired tensor factor correlations](scc-tensor-alignment.csv)
- [Editable comparison figure](scc-rank-comparison.svg)
- [SCC reference data and recipe](../reference-studies.md)
