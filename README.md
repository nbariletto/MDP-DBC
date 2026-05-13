# Scalable Posterior Uncertainty for Density-Based Clustering

Code accompanying the paper *"Scalable Posterior Uncertainty for Density-Based Clustering"*.

## Reproducing the experiments

The repository contains three experiment scripts:

- `circles_illustration.py` — noisy concentric circles illustration (Examples 1 and 2 in the paper).
- `digits_application.py` — MNIST handwritten digits application (Section 4.1).
- `scRNA_application.py` — bone marrow single-cell RNA sequencing application (Section 4.2).

`circles_illustration.py` and `digits_application.py` are fully self-contained and can be run directly.

`scRNA_application.py` requires a one-time data preparation step:

1. Download the raw data from [CZ CELLxGENE](https://datasets.cellxgene.cziscience.com/c7f0c3ea-2083-4d87-a8e0-7f69626aa40d.h5ad).
2. Run `scRNA_preprocessing.py` on the downloaded file.
3. Pass the preprocessed output to `scRNA_application.py`.

## Requirements

Required Python packages may need to be installed prior to running the scripts.

## Hardware

The compute times reported in the paper refer to runs on a single **NVIDIA RTX A4000 GPU**. Runtimes on different hardware will vary accordingly.
