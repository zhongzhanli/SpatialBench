# SpatialBench

SpatialBench is a benchmarking framework for evaluating **spatial multi-omics integration methods**. It provides a unified workflow for running different integration methods, collecting their outputs, and comparing their performance across biological, integration, and spatial criteria.

## Features

* Benchmark **paired spatial multi-omics integration** methods.
* Benchmark **unpaired spatial multi-omics integration** methods.
* Support both Python- and R-based methods.
* Evaluate clustering accuracy, biological conservation, modality/batch integration, and spatial continuity.
* Provide example notebooks for multiple human and mouse spatial multi-omics datasets.

## Repository Structure

```text
SpatialBench/
├── benchmarker/
│   ├── bm.py              # Main benchmarking interface
│   ├── metrics.py         # Evaluation metrics
│   ├── plotting.py        # Visualization utilities
│   ├── call_methods.py    # Python/R method wrappers
│   ├── utils.py           # Utility functions
│   └── script/
│       └── multiomics/    # Spatial multi-omics integration methods
└── script/
    ├── multiOmicsPaired/  # Example notebooks for paired datasets
    └── multiOmicsUnpaired/# Example notebooks for unpaired datasets
```

## Supported Methods

SpatialBench includes wrappers for a range of spatial multi-omics integration methods.

### Paired integration

Examples include:

* Seurat WNN
* MOFA2
* MultiVI
* Multigrate
* scMDC
* scMM
* Matilda
* moETM
* TotalVI
* sciPENN
* SpatialGlue
* COSMOS
* MISO
* PRESENT
* spaMultiVAE
* SMOPCA
* CellCharter

### Unpaired integration

Examples include:

* GLUE
* Monae
* SIMBA
* SCALEX
* MaxFuse
* LIGER
* scConfluence
* Seurat CCA
* Seurat RPCA
* BindSC
* SWITCH

See `benchmarker/script/multiomics/` for the implementations included in the benchmark.

## Installation

Clone the repository:

```bash
git clone https://github.com/zhongzhanli/SpatialBench.git
cd SpatialBench
```

SpatialBench uses both Python and R methods. Core Python dependencies include packages such as `scanpy`, `anndata`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `scib`, and `scib-metrics`.

Additional dependencies should be installed according to the methods you want to benchmark. R-based methods require an R environment and the corresponding R packages.

## Basic Usage

The main interface is `Benchmarker`:

```python
from benchmarker import Benchmarker

bm = Benchmarker(R_conda_env="Rbase")

bm.run(
    RNA_file_path="rna.h5",
    ATAC_file_path="atac.h5",
    save_path="results/",
    n_cluster=7,
    methods=["MultiVI", "SpatialGlue"],
    batch_key="batch",
)
```

Depending on the dataset and method, SpatialBench can work with different combinations of spatial omics modalities, such as RNA, ATAC, and protein measurements.

Spatial coordinates should be provided with the input data when spatial evaluation is required.

More complete examples are available under:

```text
script/multiOmicsPaired/
script/multiOmicsUnpaired/
```

## Evaluation

SpatialBench supports multiple categories of evaluation metrics, including:

* **Clustering agreement:** ARI and NMI.
* **Biological conservation and integration quality:** scIB/scIB-metrics.
* **Spatial continuity:** CHAOS and PAS.
* **Aggregate scores:** combined metrics for overall method comparison.

## Citation

If you use SpatialBench in your work, please cite the corresponding SpatialBench study when citation information becomes available.
