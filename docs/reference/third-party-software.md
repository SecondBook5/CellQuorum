# Third-party software

The software CellQuorum calls, what each piece is used for, its license, and how to cite it. Cite
the tools behind the stages you ran, together with [CellQuorum itself](#citing-cellquorum). Each
run records which stages and backends ran in `provenance/stage_plan.csv` and
`provenance/backend_status.json`.

Licenses were read from the installed software on 2026-10-02 (Python package metadata, or the
shipped license file where the metadata was missing or contradictory; R `DESCRIPTION` files).
Citations were compiled by hand, so check each project's own recommended citation before you
submit a paper.

## Core libraries (always installed)

| Software | Used for | License | Citation or project |
|---|---|---|---|
| [anndata](https://github.com/scverse/anndata) | The data object every stage shares | BSD-3-Clause | Virshup et al., *JOSS* 2024, [10.21105/joss.04371](https://doi.org/10.21105/joss.04371) |
| [Scanpy](https://github.com/scverse/scanpy) | Neighbours, UMAP, scoring, plotting helpers | BSD-3-Clause | Wolf, Angerer & Theis, *Genome Biol* 2018, [10.1186/s13059-017-1382-0](https://doi.org/10.1186/s13059-017-1382-0) |
| [leidenalg](https://github.com/vtraag/leidenalg) | Leiden clustering | GPL-3.0-or-later | Traag, Waltman & van Eck, *Sci Rep* 2019, [10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z) |
| [harmonypy](https://github.com/slowkow/harmonypy) | Harmony batch integration | GPL-3 (license file; the package metadata says MIT) | Korsunsky et al., *Nat Methods* 2019, [10.1038/s41592-019-0619-0](https://doi.org/10.1038/s41592-019-0619-0) |
| [kneed](https://github.com/arvkevi/kneed) | Choosing the number of principal components | BSD-3-Clause | Satopaa et al., *ICDCSW* 2011, [10.1109/ICDCSW.2011.20](https://doi.org/10.1109/ICDCSW.2011.20) |
| [NumPy](https://numpy.org) | Arrays | BSD-3-Clause | Harris et al., *Nature* 2020, [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2) |
| [SciPy](https://scipy.org) | Statistics and sparse matrices | BSD-3-Clause | Virtanen et al., *Nat Methods* 2020, [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2) |
| [pandas](https://pandas.pydata.org) | Tables | BSD-3-Clause | The pandas development team, [10.5281/zenodo.3509134](https://doi.org/10.5281/zenodo.3509134) |
| [scikit-misc](https://github.com/has2k1/scikit-misc) | LOESS for highly variable genes | BSD-3-Clause | Project page |
| [Matplotlib](https://matplotlib.org) | Every figure | PSF-based (BSD-compatible) | Hunter, *Comput Sci Eng* 2007, [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55) |
| [seaborn](https://seaborn.pydata.org) | Statistical plotting | BSD-3-Clause | Waskom, *JOSS* 2021, [10.21105/joss.03021](https://doi.org/10.21105/joss.03021) |
| [pydantic](https://github.com/pydantic/pydantic), [OmegaConf](https://github.com/omry/omegaconf), [Hydra](https://github.com/facebookresearch/hydra) (`hydra-core`), [PyYAML](https://pyyaml.org) | Configuration validation and loading | MIT, BSD, MIT, MIT | Project pages |
| [Typer](https://github.com/fastapi/typer), [Rich](https://github.com/Textualize/rich), [Jinja2](https://github.com/pallets/jinja), [filelock](https://github.com/tox-dev/filelock) | Command line, console output, reports, file locking | MIT, MIT, BSD-3-Clause, MIT | Project pages |

## Quality control, normalization and annotation

| Software | Used for | License | Citation or project |
|---|---|---|---|
| [SampleQC](https://github.com/wmacnair/SampleQC) (R) | Joint, multi-sample QC profiles | Not checked here | Project page, see the authors' recommended citation |
| [Scrublet](https://github.com/swolock/scrublet) | Doublet detection | MIT | Wolock, Lopez & Klein, *Cell Syst* 2019, [10.1016/j.cels.2018.11.005](https://doi.org/10.1016/j.cels.2018.11.005) |
| [scDblFinder](https://github.com/plger/scDblFinder) (R) | Doublet detection | GPL-3 + file LICENSE | Germain et al., *F1000Research* 2022, [10.12688/f1000research.73600.2](https://doi.org/10.12688/f1000research.73600.2) |
| [SoupX](https://github.com/constantAmateur/SoupX) (R) | Ambient RNA correction | GPL-2 | Young & Behjati, *GigaScience* 2020, [10.1093/gigascience/giaa151](https://doi.org/10.1093/gigascience/giaa151) |
| [CellTypist](https://github.com/Teichlab/celltypist) | Automated cell-type annotation | MIT | Domínguez Conde et al., *Science* 2022, [10.1126/science.abl5197](https://doi.org/10.1126/science.abl5197) |
| [scDiagnostics](https://github.com/ccb-hms/scDiagnostics) (R) | Annotation diagnostics | Artistic-2.0 | Project page |
| [scvi-tools](https://github.com/scverse/scvi-tools) | scVI and scANVI | BSD-3-Clause | Gayoso et al., *Nat Biotechnol* 2022, [10.1038/s41587-021-01206-w](https://doi.org/10.1038/s41587-021-01206-w); Lopez et al., *Nat Methods* 2018, [10.1038/s41592-018-0229-2](https://doi.org/10.1038/s41592-018-0229-2); Xu et al., *Mol Syst Biol* 2021, [10.15252/msb.20209620](https://doi.org/10.15252/msb.20209620) |
| scArches | Reference mapping (through scvi-tools) | BSD-3-Clause (as part of scvi-tools) | Lotfollahi et al., *Nat Biotechnol* 2022, [10.1038/s41587-021-01001-7](https://doi.org/10.1038/s41587-021-01001-7) |
| [scIB](https://github.com/theislab/scib) metrics | Integration benchmarking | Not checked here | Luecken et al., *Nat Methods* 2022, [10.1038/s41592-021-01336-8](https://doi.org/10.1038/s41592-021-01336-8) |
| [CHOIR](https://github.com/corceslab/CHOIR) (R) | Hierarchical subclustering | MIT + file LICENSE | Project page |
| [sc-SHC](https://github.com/igrabski/sc-SHC) (R package `scSHC`) | Significance test for clusters | Artistic-2.0 | Grabski, Street & Irizarry, *Nat Methods* 2023, [10.1038/s41592-023-01933-9](https://doi.org/10.1038/s41592-023-01933-9) |
| [clustree](https://github.com/lazappi/clustree) (R) | Clustering-resolution trees | GPL-3 | Zappia & Oshlack, *GigaScience* 2018, [10.1093/gigascience/giy083](https://doi.org/10.1093/gigascience/giy083) |
| [MetaCell-2](https://github.com/tanaylab/metacells) | Metacells | MIT | Ben-Kiki, Bercovich & Amit, *Genome Biol* 2022, [10.1186/s13059-022-02667-1](https://doi.org/10.1186/s13059-022-02667-1) |

### Methods implemented in CellQuorum or inside the libraries above

| Method | Used for | Citation |
|---|---|---|
| Cell-cycle gene lists | Cell-cycle scoring | Tirosh et al., *Science* 2016, [10.1126/science.aad0501](https://doi.org/10.1126/science.aad0501) |
| Seurat v3 variable genes | Highly variable gene selection | Stuart et al., *Cell* 2019, [10.1016/j.cell.2019.05.031](https://doi.org/10.1016/j.cell.2019.05.031) |
| Analytic Pearson residuals | Feature selection | Lause, Berens & Kobak, *Genome Biol* 2021, [10.1186/s13059-021-02451-7](https://doi.org/10.1186/s13059-021-02451-7) |
| Binomial deviance | Feature selection | Townes et al., *Genome Biol* 2019, [10.1186/s13059-019-1861-6](https://doi.org/10.1186/s13059-019-1861-6) |
| Consensus NMF | De-novo program discovery | Kotliar et al., *eLife* 2019, [10.7554/eLife.43803](https://doi.org/10.7554/eLife.43803) |

## Embeddings and trajectories

| Software | Used for | License | Citation or project |
|---|---|---|---|
| [UMAP](https://github.com/lmcinnes/umap) | UMAP embedding | BSD-3-Clause | McInnes, Healy & Melville, *JOSS* 2018, [10.21105/joss.00861](https://doi.org/10.21105/joss.00861) |
| PAGA (in Scanpy) | Cluster connectivity | BSD-3-Clause | Wolf et al., *Genome Biol* 2019, [10.1186/s13059-019-1663-x](https://doi.org/10.1186/s13059-019-1663-x) |
| [PHATE](https://github.com/KrishnaswamyLab/PHATE) | PHATE embedding | GPL-2 | Moon et al., *Nat Biotechnol* 2019, [10.1038/s41587-019-0336-3](https://doi.org/10.1038/s41587-019-0336-3) |
| [MAGIC](https://github.com/KrishnaswamyLab/MAGIC) | Imputation for overlays | GPL-2 | van Dijk et al., *Cell* 2018, [10.1016/j.cell.2018.05.061](https://doi.org/10.1016/j.cell.2018.05.061) |
| [scVelo](https://github.com/theislab/scvelo) | RNA velocity | BSD-3-Clause | Bergen et al., *Nat Biotechnol* 2020, [10.1038/s41587-020-0591-3](https://doi.org/10.1038/s41587-020-0591-3) |
| [velocyto](https://github.com/velocyto-team/velocyto.py) | Spliced and unspliced counts | BSD | La Manno et al., *Nature* 2018, [10.1038/s41586-018-0414-6](https://doi.org/10.1038/s41586-018-0414-6) |
| [CellRank](https://github.com/theislab/cellrank) | Fate probabilities | BSD-3-Clause | Lange et al., *Nat Methods* 2022, [10.1038/s41592-021-01346-6](https://doi.org/10.1038/s41592-021-01346-6); Weiler et al., *Nat Methods* 2024, [10.1038/s41592-024-02303-9](https://doi.org/10.1038/s41592-024-02303-9) |
| [Palantir](https://github.com/dpeerlab/Palantir) | Pseudotime | MIT | Setty et al., *Nat Biotechnol* 2019, [10.1038/s41587-019-0068-4](https://doi.org/10.1038/s41587-019-0068-4) |
| [CytoTRACE 2](https://github.com/digitalcytometry/cytotrace2) | Developmental potential | See [Licensing notes](#licensing-notes) | Project page |

## Comparing groups

| Software | Used for | License | Citation or project |
|---|---|---|---|
| [edgeR](https://bioconductor.org/packages/edgeR) (R) | Pseudobulk differential expression | GPL (>= 2) | Robinson, McCarthy & Smyth, *Bioinformatics* 2010, [10.1093/bioinformatics/btp616](https://doi.org/10.1093/bioinformatics/btp616) |
| [limma](https://bioconductor.org/packages/limma) (R) | Linear models | GPL (>= 2) | Ritchie et al., *Nucleic Acids Res* 2015, [10.1093/nar/gkv007](https://doi.org/10.1093/nar/gkv007) |
| [variancePartition](https://bioconductor.org/packages/variancePartition) (R) | `dream` mixed-model pseudobulk differential expression for repeated measures (`pseudobulk_dream`) | GPL (>= 2) | Hoffman & Roussos, *Bioinformatics* 2021, [10.1093/bioinformatics/btaa687](https://doi.org/10.1093/bioinformatics/btaa687) |
| [miloR](https://github.com/MarioniLab/miloR) (R) | Differential abundance on neighbourhoods | GPL-3 + file LICENSE | Dann et al., *Nat Biotechnol* 2022, [10.1038/s41587-021-01033-z](https://doi.org/10.1038/s41587-021-01033-z) |
| [scCODA](https://github.com/theislab/scCODA) | Compositional differential abundance | BSD | Büttner et al., *Nat Commun* 2021, [10.1038/s41467-021-27150-6](https://doi.org/10.1038/s41467-021-27150-6) |
| [speckle](https://github.com/phipsonlab/speckle) (R) | propeller, differences in proportions | GPL-3 | Phipson et al., *Bioinformatics* 2022, [10.1093/bioinformatics/btac582](https://doi.org/10.1093/bioinformatics/btac582) |
| [decoupler](https://github.com/scverse/decoupler) | Gene-set and regulon activity (ULM, AUCell, GSEA, ORA) | BSD-3-Clause | Badia-i-Mompel et al., *Bioinformatics Advances* 2022, [10.1093/bioadv/vbac016](https://doi.org/10.1093/bioadv/vbac016) |
| GSEA | Gene-set enrichment | n/a (method) | Subramanian et al., *PNAS* 2005, [10.1073/pnas.0506580102](https://doi.org/10.1073/pnas.0506580102) |
| GSVA | Gene-set variation analysis | n/a (method) | Hänzelmann, Castelo & Guinney, *BMC Bioinformatics* 2013, [10.1186/1471-2105-14-7](https://doi.org/10.1186/1471-2105-14-7) |
| AUCell | Per-cell gene-set scores | n/a (method) | Aibar et al., *Nat Methods* 2017, [10.1038/nmeth.4463](https://doi.org/10.1038/nmeth.4463) |
| [UCell](https://github.com/carmonalab/UCell) | Per-cell gene-set scores | Not checked here | Andreatta & Carmona, *Comput Struct Biotechnol J* 2021, [10.1016/j.csbj.2021.06.043](https://doi.org/10.1016/j.csbj.2021.06.043) |
| CollecTRI, DoRothEA, PROGENy (via OmniPath) | Regulon and pathway prior knowledge | See [Licensing notes](#licensing-notes) | Müller-Dott et al., *Nucleic Acids Res* 2023, [10.1093/nar/gkad841](https://doi.org/10.1093/nar/gkad841); Garcia-Alonso et al., *Genome Res* 2019, [10.1101/gr.240663.118](https://doi.org/10.1101/gr.240663.118); Schubert et al., *Nat Commun* 2018, [10.1038/s41467-017-02391-6](https://doi.org/10.1038/s41467-017-02391-6) |
| [DIALOGUE](https://github.com/livnatje/DIALOGUE) (R) | Multicellular programs | Custom, see [Licensing notes](#licensing-notes) | Jerby-Arnon & Regev, *Nat Biotechnol* 2022, [10.1038/s41587-022-01288-0](https://doi.org/10.1038/s41587-022-01288-0) |

## Gene regulation and communication

| Software | Used for | License | Citation or project |
|---|---|---|---|
| [hdWGCNA](https://github.com/smorabit/hdWGCNA) (R) | Co-expression modules | GPL-3 | Morabito et al., *Cell Rep Methods* 2023, [10.1016/j.crmeth.2023.100498](https://doi.org/10.1016/j.crmeth.2023.100498); Langfelder & Horvath, *BMC Bioinformatics* 2008, [10.1186/1471-2105-9-559](https://doi.org/10.1186/1471-2105-9-559) |
| [pySCENIC](https://github.com/aertslab/pySCENIC) | Regulon inference | GPL-3.0-or-later | Van de Sande et al., *Nat Protoc* 2020, [10.1038/s41596-020-0336-2](https://doi.org/10.1038/s41596-020-0336-2); Aibar et al., *Nat Methods* 2017, [10.1038/nmeth.4463](https://doi.org/10.1038/nmeth.4463) |
| [arboreto](https://github.com/aertslab/arboreto) (GRNBoost2) | Regulatory network inference | BSD-3-Clause | Moerman et al., *Bioinformatics* 2019, [10.1093/bioinformatics/bty916](https://doi.org/10.1093/bioinformatics/bty916) |
| [CellOracle](https://github.com/morris-lab/CellOracle) | In-silico transcription-factor knockouts | Apache-2.0 | Kamimoto et al., *Nature* 2023, [10.1038/s41586-022-05688-9](https://doi.org/10.1038/s41586-022-05688-9) |
| [LIANA](https://github.com/saezlab/liana-py) | Ligand-receptor inference | BSD-3-Clause | Dimitrov et al., *Nat Commun* 2022, [10.1038/s41467-022-30755-0](https://doi.org/10.1038/s41467-022-30755-0) |
| [cell2cell](https://github.com/earmingol/cell2cell) (Tensor-cell2cell) | Communication across samples | BSD-3-Clause | Armingol et al., *Nat Commun* 2022, [10.1038/s41467-022-31369-2](https://doi.org/10.1038/s41467-022-31369-2) |
| [NicheNet](https://github.com/saeyslab/nichenetr) (R) | Ligand activity | GPL-3 | Browaeys, Saelens & Saeys, *Nat Methods* 2020, [10.1038/s41592-019-0667-5](https://doi.org/10.1038/s41592-019-0667-5) |
| [MultiNicheNet](https://github.com/saeyslab/multinichenetr) (R) | Condition-aware communication | GPL-3 + file LICENSE | Browaeys et al., *bioRxiv* 2023, [10.1101/2023.06.13.544751](https://doi.org/10.1101/2023.06.13.544751) |
| [GraphRicciCurvature](https://github.com/saibalmars/GraphRicciCurvature) | Ollivier-Ricci curvature | Apache-2.0 | Ni et al., *Sci Rep* 2019, [10.1038/s41598-019-46380-9](https://doi.org/10.1038/s41598-019-46380-9) |
| [NetworkX](https://networkx.org) | Graph algorithms | BSD-3-Clause | Hagberg, Schult & Swart, *SciPy* 2008 |
| [pycirclize](https://github.com/moshi4/pyCirclize), [adjustText](https://github.com/Phlya/adjustText), [Plotly](https://plotly.com) | Circos plots, label placement, Sankey diagrams | MIT, MIT, MIT | Project pages |

## R infrastructure

| Software | Used for | License |
|---|---|---|
| [Seurat](https://satijalab.org/seurat) (Hao et al., *Nat Biotechnol* 2024, [10.1038/s41587-023-01767-y](https://doi.org/10.1038/s41587-023-01767-y)) | Object conversion and R-side analysis | MIT + file LICENSE |
| [SingleCellExperiment](https://bioconductor.org/packages/SingleCellExperiment), [scater](https://bioconductor.org/packages/scater), [scuttle](https://bioconductor.org/packages/scuttle) | Bioconductor single-cell containers and utilities | GPL-3 |
| [zellkonverter](https://bioconductor.org/packages/zellkonverter) | `.h5ad` to `SingleCellExperiment` | MIT + file LICENSE |
| [BiocParallel](https://bioconductor.org/packages/BiocParallel) | Serial back end for `variancePartition::dream` | GPL-2 \| GPL-3 |
| [ComplexHeatmap](https://github.com/jokergoo/ComplexHeatmap) (Gu et al., *Bioinformatics* 2016, [10.1093/bioinformatics/btw313](https://doi.org/10.1093/bioinformatics/btw313)), [circlize](https://github.com/jokergoo/circlize) (Gu et al., *Bioinformatics* 2014, [10.1093/bioinformatics/btu393](https://doi.org/10.1093/bioinformatics/btu393)) | R-side figures | MIT + file LICENSE |
| [Matrix](https://cran.r-project.org/package=Matrix), [data.tree](https://cran.r-project.org/package=data.tree) | Sparse matrices, trees | GPL (>= 2) |
| [dplyr](https://dplyr.tidyverse.org), [ggplot2](https://ggplot2.tidyverse.org), [jsonlite](https://github.com/jeroen/jsonlite) | Data handling and plotting | MIT + file LICENSE |
| [rpy2](https://github.com/rpy2/rpy2) | Optional Python to R bridge | GPL-2.0-or-later |

## Workflow

- [Snakemake](https://snakemake.github.io) runs multi-hypothesis workflows (Mölder et al.,
  *F1000Research* 2021, [10.12688/f1000research.29032.2](https://doi.org/10.12688/f1000research.29032.2)).

## Data resources

- The real-data benchmark uses the public GEO
  accession [GSE140819](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140819): Slyper et
  al., "A single-cell and single-nucleus RNA-Seq toolbox for fresh and frozen human tumors",
  *Nat Med* 2020, [10.1038/s41591-020-0844-1](https://doi.org/10.1038/s41591-020-0844-1).
- Reference-mapping examples use the Single Cell Portal atlas SCP2613, and the lab's own
  lymphedema and mast-cell studies are the validation projects for the engine.
- Synthetic data in the gallery, tutorial and `cellquorum demo` is generated by CellQuorum and
  means nothing biologically.

## Licensing notes

This section is a plain-language summary, not legal advice. If you plan to redistribute
CellQuorum together with other software, or to use it commercially, check the terms with your
institution.

**CellQuorum's own license.** CellQuorum's code, R scripts and documentation are released under
the [BSD 3-Clause License](https://github.com/SecondBook5/cellquorum/blob/main/LICENSE), copyright
2026 AJ Book. That licence covers CellQuorum only.

**Third-party software keeps its own license.** Installing it is a separate step, and each
license applies to its own code. In particular:

- **Copyleft (GPL) software is called by CellQuorum.** `leidenalg` (GPL-3.0-or-later) is a
  required dependency and is imported by the clustering stage. `harmonypy` ships a GPL-3 license
  file (its package metadata says MIT, so treat it as GPL-3). `rpy2`, PHATE and MAGIC are GPL-2.
  Most of the R packages above (edgeR, limma, variancePartition, SoupX, miloR, NicheNet, MultiNicheNet, scDblFinder,
  speckle, hdWGCNA and the Bioconductor containers) are GPL, as is pySCENIC. CellQuorum runs the
  R packages and pySCENIC as separate processes and imports the Python ones; it contains none of
  their code.
- **Containers redistribute everything they contain.** The images built from `docker/` install
  these packages. If you publish or share an image, you are redistributing them under their own
  licenses.
- **Read the terms before commercial use of:** `cytotrace2-py`, whose package declares a
  non-standard license and ships no license file, so check the project's terms; and DIALOGUE,
  which is released as "free to use, please cite our paper".
- **Prior-knowledge resources have their own terms.** The regulon and pathway resources
  (CollecTRI, DoRothEA, PROGENy) are fetched through OmniPath. CellQuorum asks for the
  academic-licensed subset (`license="academic"`), which is intended for non-commercial research.

## Citing CellQuorum

Please cite CellQuorum itself, and the methods above that produced your results.

```bibtex
@software{book_cellquorum_2026,
  author  = {Book, AJ},
  title   = {{CellQuorum}: a publication-grade, config-driven single-cell RNA-seq workflow engine},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/SecondBook5/cellquorum},
  license = {BSD-3-Clause}
}
```

GitHub also offers a "Cite this repository" button, which reads
[`CITATION.cff`](https://github.com/SecondBook5/cellquorum/blob/main/CITATION.cff).
