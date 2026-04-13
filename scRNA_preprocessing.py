# Data can be downloaded from:
# https://datasets.cellxgene.cziscience.com/c7f0c3ea-2083-4d87-a8e0-7f69626aa40d.h5ad

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load dataset
adata = ad.read_h5ad("c7f0c3ea-2083-4d87-a8e0-7f69626aa40d.h5ad")

# Extract UMAP, first 10 PCA dimensions, and cell types
umap_coords = adata.obsm['X_umap']
pca_10 = adata.obsm['X_pca'][:, :10]
cell_types = adata.obs['cell_type']

# Combine into a single DataFrame
pca_columns = [f"PC_{i+1}" for i in range(10)]
df_combined = pd.DataFrame(pca_10, columns=pca_columns, index=adata.obs_names)

df_combined['UMAP_1'] = umap_coords[:, 0]
df_combined['UMAP_2'] = umap_coords[:, 1]
df_combined['cell_type'] = cell_types

# Save to CSV
output_filename = "tsbm.csv"
df_combined.to_csv(output_filename)

# Plot UMAP coordinates
plt.figure(figsize=(8, 8))
plt.scatter(
    umap_coords[:, 0], 
    umap_coords[:, 1], 
    s=1, 
    alpha=0.5, 
    c='tab:blue'
)
plt.title("Tabula Sapiens Bone Marrow - Precomputed UMAP")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.axis('equal')
plt.show()