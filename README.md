These scripts reproduce the numerical experiments presented in the paper "Scalable Posterior Uncertainty for Flexible Density-Based Clustering".

The ``digits_application``, ``mixture_illustration`` and ``circles_illustration`` scripts are fully self-contained.

Instead, to run the code in the ``scRNA_application`` script, one first needs to download the employed dataset at https://datasets.cellxgene.cziscience.com/c7f0c3ea-2083-4d87-a8e0-7f69626aa40d.h5ad, pass it through the preprocessing script ``scRNA_preprocessing``, and then use the output of the latter as an input to run ``scRNA_application``.

All required utilities may need installation prior to reproducing the experiments.
