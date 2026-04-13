import os
import sys

# ==========================================
# 0. REPRODUCIBILITY ENVIRONMENT SETUP
# ==========================================
SEED = 2026

if os.environ.get("PYTHONHASHSEED") != str(SEED):
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.execv(sys.executable, ['python'] + sys.argv)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import functools
from typing import Sequence
import random
import numpy as np
import pandas as pd
import argparse
import gc

# ---- JAX + flow stack ----
import jax
import jax.numpy as jnp
import haiku as hk
import distrax
import optax
from surjectors import TransformedDistribution, Chain, MaskedAutoregressive, Permutation
from surjectors.nn import MADE
from surjectors.util import unstack

# ---- clustering, plotting, utilities ----
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm
import gudhi.clustering.tomato as tomato

# ==========================================
# CONFIGURATION
# ==========================================
SAVE_DIR = "./tsbm_results"
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_PATH = "./tsbm.csv"
BATCH_SIZE = 500
LATENT_DIM = None 

jax.config.update("jax_enable_x64", False)

np.random.seed(SEED)
random.seed(SEED)

# ==========================================
# 1. DATA LOADING & NORMALIZATION
# ==========================================
def load_data():
    print(f"Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    feature_cols = [f"PC_{i+1}" for i in range(10)]
    umap_cols = ['UMAP_1', 'UMAP_2']
    
    X = df[feature_cols].values
    umap_coords = df[umap_cols].values
    cell_types = df['cell_type'].values 
    
    return X, umap_coords, cell_types

# ==========================================
# 2. FLOW MODEL (NF Density Estimator)
# ==========================================
def make_flow_model(n_dimensions: int, flow_depth: int = 16) -> TransformedDistribution:
    def bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return distrax.Inverse(
            distrax.ScalarAffine(shift=means, scale=jnp.exp(log_scales))
        )

    layers = []
    order = jnp.arange(n_dimensions)
    
    for i in range(flow_depth):
        conditioner = MADE(n_dimensions, [128, 128], 2)
        layer = MaskedAutoregressive(
            conditioner=conditioner,
            bijector_fn=bijector_fn,
        )
        layers.append(layer)
        order = order[::-1]
        layers.append(Permutation(order, 1))

    transform = Chain(layers)
    base_distribution = distrax.Independent(
        distrax.Normal(jnp.zeros(n_dimensions), jnp.ones(n_dimensions)),
        reinterpreted_batch_ndims=1,
    )
    return TransformedDistribution(base_distribution, transform)

@hk.without_apply_rng
@hk.transform
def logprob_fn(x: jnp.ndarray) -> jnp.ndarray:
    model = make_flow_model(LATENT_DIM)
    return model.log_prob(x)

@hk.transform
def sample_fn(num: int) -> jnp.ndarray:
    model = make_flow_model(LATENT_DIM)
    return model.sample(sample_shape=(num,))

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_tomato_clusters(X, log_probs, adj_list, min_persistence=0.1, plot_path=None, title_suffix="", ax=None):
    t = tomato.Tomato(graph_type='manual', density_type='manual')
    t.fit(X=adj_list, weights=np.exp(log_probs-np.max(log_probs)))  
    
    if ax is not None:
        plt.sca(ax)
        t.plot_diagram()
        ax.set_title(f"{title_suffix}")
    elif plot_path is not None:
        t.plot_diagram()
        plt.savefig(plot_path)
        plt.close()
        
    t.merge_threshold_ = 0.3
    return t.labels_, t.n_clusters_

def replicate_params(params, T: int):
    return jax.tree.map(lambda p: jnp.repeat(p[None, ...], T, axis=0), params)

def make_step_fn(logprob_apply, sample_apply, base_lr):
    def loss_for_single(params_single, x_single):
        lp = logprob_apply(params_single, x_single[None, :])
        return -jnp.squeeze(lp, axis=0)
    
    grad_single = jax.grad(loss_for_single)
    vmapped_sample = jax.vmap(lambda p, k: jnp.squeeze(sample_apply(p, k, 1), axis=0), in_axes=(0, 0))
    vmapped_grad = jax.vmap(grad_single, in_axes=(0, 0))

    @functools.partial(jax.jit, static_argnums=(3,))
    def step(params_batched, master_key, step_idx: int, T: int):
        keys = jax.random.split(master_key, T)
        xs = vmapped_sample(params_batched, keys)
        grads_batched = vmapped_grad(params_batched, xs)
        lr = base_lr / (27112 + step_idx)
        params_batched = jax.tree.map(lambda p, g: p - lr * jnp.clip(g, -1, 1), params_batched, grads_batched)
        return params_batched
    return step

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_cbi_only", action="store_true", help="Internal flag to run only CBI stage")
    args = parser.parse_args()

    if not args.run_cbi_only:
        print(f"Using JAX Device: {jax.devices()[0]}")

        # --- 1. Load & Normalize Data ---
        X_raw, umap_coords, cell_types = load_data()
        LATENT_DIM = X_raw.shape[1]
        print(f"Total samples: {len(X_raw)} | PCA Features (Latent Dim): {LATENT_DIM}")

        x_mean = np.mean(X_raw, axis=0)
        x_std = np.std(X_raw, axis=0) + 1e-6
        X_norm_np = (X_raw - x_mean) / x_std

        Z_norm = jnp.array(X_norm_np, dtype=jnp.float32)
        Z_np = np.array(X_norm_np)

        # --- 2. Train Flow ---
        print("\n=== STAGE: TRAINING MAF FLOW ===")
        rng, key = jax.random.split(jax.random.PRNGKey(SEED)) 
        params_flow = logprob_fn.init(key, jnp.zeros((1, LATENT_DIM)))
        
        epochs_flow = 100
        steps_per_epoch = int(np.ceil(len(Z_norm) / BATCH_SIZE))
        total_steps = epochs_flow * steps_per_epoch

        lr_schedule = optax.cosine_decay_schedule(
            init_value=1e-4,
            decay_steps=total_steps,
            alpha=0.1 
        )
        
        opt_flow = optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
        opt_state_flow = opt_flow.init(params_flow)

        @jax.jit
        def train_step_flow(params, opt_state, batch):
            def loss_fn(p, x):
                return -jnp.mean(logprob_fn.apply(p, x))
            loss, grads = jax.value_and_grad(loss_fn)(params, batch)
            updates, new_state = opt_flow.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_state, loss

        pbar = tqdm(range(epochs_flow), desc="Flow Training")
        np_rng = np.random.RandomState(SEED)
        
        for ep in pbar:
            perm = np_rng.permutation(len(Z_norm))
            avg_loss = 0.0
            steps = 0
            for i in range(0, len(Z_norm), BATCH_SIZE):
                batch = Z_norm[perm[i:i+BATCH_SIZE]]
                params_flow, opt_state_flow, loss = train_step_flow(params_flow, opt_state_flow, batch)
                avg_loss += float(loss)
                steps += 1
            pbar.set_postfix({'NLL': f"{avg_loss/steps:.4f}"})

        base_log_probs = np.array(logprob_fn.apply(params_flow, Z_norm))

        # --- 3. CLUSTERING & PLOT 1 ---
        print("\n=== STAGE: BASE CLUSTERING ===")
        nbrs = NearestNeighbors(n_neighbors=30, algorithm='brute', n_jobs=1).fit(Z_np)
        knn_graph = nbrs.kneighbors_graph(Z_np, mode='connectivity')
        rows, cols = knn_graph.nonzero()
        adj_list = [cols[knn_graph.indptr[i]:knn_graph.indptr[i+1]] for i in range(knn_graph.shape[0])]

        fig_pers, axes_pers = plt.subplots(2, 2, figsize=(5, 5))
        axes_flat = axes_pers.flatten()

        labels_pre, _ = get_tomato_clusters(
            Z_np, base_log_probs, adj_list, 
            ax=axes_flat[0], 
            title_suffix="Trained Density"
        )

        # Save the pretrained persistence diagram separately
        _, _ = get_tomato_clusters(
            Z_np, base_log_probs, adj_list, 
            plot_path=f"{SAVE_DIR}/pretrained_persistence.pdf"
        )

        # Plot 1: Side-by-side Cell Type UMAP + Pretrained Clustering UMAP
        cmap = plt.cm.tab20
        cmap20_colors = [cmap(i / 20) for i in range(20)]  

        # --- build 25-color palette (all alpha=1) ---
        _pal_colors = (
            list(plt.cm.tab20.colors) +          
            list(plt.cm.Set1.colors[:5])          
        )
        PALETTE25 = [(*c[:3], 1.0) for c in _pal_colors]

        # --- Group top 13 cell types + 'other' ---
        unique_ct_raw, ct_counts = np.unique(cell_types, return_counts=True)
        top_13_indices = np.argsort(ct_counts)[-13:][::-1]
        top_13_ct = unique_ct_raw[top_13_indices]

        cell_types_grouped = np.where(np.isin(cell_types, top_13_ct), cell_types, 'other')
        unique_ct = list(top_13_ct)
        if 'other' in cell_types_grouped and 'other' not in unique_ct:
            unique_ct.append('other')
        unique_ct = np.array(unique_ct)

        ct_to_idx = {ct: i for i, ct in enumerate(unique_ct)}
        ct_indices = np.array([ct_to_idx[ct] for ct in cell_types_grouped])

        unique_lbls = sorted([l for l in np.unique(labels_pre) if l != -1])
        n_ct_items = len(unique_ct)

        # Flatten aspect ratio and increase figsize for bigger fonts & legend space
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- left: cell-type colored ---
        colors_ct = np.array([PALETTE25[i % len(PALETTE25)] for i in ct_indices])
        axes[0].scatter(umap_coords[:, 0], umap_coords[:, 1],
                        c=colors_ct, s=5, edgecolors='none', alpha=0.7)
        axes[0].axis('off')
        axes[0].set_title("Cell Type", fontsize=18)
        handles_ct = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=PALETTE25[i % len(PALETTE25)][:3],
                       markersize=8, label=unique_ct[i])
            for i in range(n_ct_items)
        ]
        axes[0].legend(handles=handles_ct, fontsize=14,
                       loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       framealpha=0.7, ncol=2, markerscale=1.5)

        # --- right: trained clustering colored ---
        colors_pre = np.zeros((len(labels_pre), 4))
        for i, lbl in enumerate(labels_pre):
            if lbl == -1:
                colors_pre[i] = [0, 0, 0, 1.0]
            else:
                colors_pre[i] = cmap20_colors[lbl % 20]
        axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1],
                        c=colors_pre, s=5, edgecolors='none', alpha=0.7)
        axes[1].axis('off')
        axes[1].set_title("Trained Clustering", fontsize=18)
        handles_cl = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cmap20_colors[l % 20][:3],
                       markersize=8, label=f"cluster {l+1}")
            for l in unique_lbls
        ]
        if -1 in np.unique(labels_pre):
            handles_cl.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='black',
                                          markersize=8, label='Unassigned'))
        axes[1].legend(handles=handles_cl, fontsize=14,
                       loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       framealpha=0.7, ncol=2, markerscale=1.5)

        plt.savefig(f"{SAVE_DIR}/pretrained_clustering_umap.pdf", bbox_inches='tight')
        plt.close()

        # --- 4. RESAMPLING ---
        print("\n=== STAGE: RESAMPLING (JAX Parallel) ===")
        NUM_RESAMPLES = 500 
        NUM_UPDATES = 3000
        BASE_LR = 5e-3

        params_batched = replicate_params(params_flow, NUM_RESAMPLES)
        step_fn = make_step_fn(logprob_fn.apply, sample_fn.apply, BASE_LR)
        
        master_key = jax.random.PRNGKey(100)
        
        print(f"Running {NUM_RESAMPLES} chains for {NUM_UPDATES} steps...")
        for k in tqdm(range(NUM_UPDATES)):
            master_key, subkey = jax.random.split(master_key)
            params_batched = step_fn(params_batched, subkey, k, NUM_RESAMPLES)
        
        print("Evaluating resampled densities...")
        vmapped_logp = jax.vmap(lambda p, x: logprob_fn.apply(p, x), in_axes=(0, None))
        all_logps_list = []
        chunk_size = 50
        for i in tqdm(range(0, NUM_RESAMPLES, chunk_size)):
            p_chunk = jax.tree.map(lambda x: x[i:i+chunk_size], params_batched)
            lp = vmapped_logp(p_chunk, Z_norm)
            all_logps_list.append(np.array(lp))
        all_logps = np.vstack(all_logps_list) 

        del params_batched
        gc.collect()

        print("Clustering all resamples...")
        all_labels_list = []
        n_clusters_history = []
        
        for i in tqdm(range(NUM_RESAMPLES)):
            target_ax = axes_flat[i+1] if i < 3 else None
            
            lbs, n_cl = get_tomato_clusters(
                Z_np, all_logps[i], adj_list, 
                ax=target_ax,
                title_suffix=f"Resampled Density {i}"
            )
            
            if i == 4:
                plt.tight_layout()
                plt.savefig(f"{SAVE_DIR}/combined_persistence_diagrams.pdf")
                plt.close(fig_pers)
                
            all_labels_list.append(lbs)
            n_clusters_history.append(n_cl)
            
        del all_logps
        del all_logps_list
        gc.collect()

        all_labels_array = np.vstack(all_labels_list)

        # --- 5. ANALYSIS & PLOTTING ---
        print("\n=== STAGE: ANALYSIS ===")
        
        # Plot 3: Cluster Count Histogram
        counts = np.bincount(n_clusters_history, minlength=21)[1:21]  # counts for 1..20
        x = np.arange(1, 21)
        plt.figure(figsize=(6, 3))
        plt.plot(x, counts, marker='o', linewidth=1.5, markersize=4, color='steelblue')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Frequency")
        plt.xticks(x)
        plt.xlim(0.5, 20.5)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/cluster_count_histogram.pdf")
        plt.close()

        # --- Matrix Computation: Cell-type aggregated coclustering ---
        print("\nComputing cell-type aggregated coclustering matrix...")
        # Note: Using the original, ungrouped cell_types here as requested
        unique_types, type_indices = np.unique(cell_types, return_inverse=True)
        n_types = len(unique_types)
        type_counts = np.bincount(type_indices, minlength=n_types).astype(np.float64)

        agg_matrix = np.zeros((n_types, n_types), dtype=np.float64)
        for r in range(all_labels_array.shape[0]):
            labels_r = all_labels_array[r]
            for cl in np.unique(labels_r):
                if cl == -1:
                    continue
                mask = labels_r == cl
                types_in_cluster = type_indices[mask]
                ct = np.bincount(types_in_cluster, minlength=n_types).astype(np.float64)
                agg_matrix += np.outer(ct, ct)

        norm = np.outer(type_counts, type_counts)
        norm[norm == 0] = 1.0
        agg_matrix /= (norm * all_labels_array.shape[0])

        # Sort by hierarchical clustering
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        dist = 1.0 - agg_matrix
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0, None)
        linkage_matrix = linkage(squareform(dist), method='average')
        order = leaves_list(linkage_matrix)

        agg_sorted = agg_matrix[np.ix_(order, order)]
        types_sorted = unique_types[order]
        counts_sorted = type_counts[order].astype(int)

        # Y-axis labels: "cell name\n(count)"
        yticklabels = [f"{t} ({c})" for t, c in zip(types_sorted, counts_sorted)]

        # --- Matrix Computation: Neutrophil-restricted coclustering ---
        print("\nComputing neutrophil-restricted coclustering matrix...")
        neutrophil_idx = np.where(unique_types == 'neutrophil')[0][0]
        neutrophil_mask = type_indices == neutrophil_idx
        n_neutrophil = int(neutrophil_mask.sum())
        print(f"Neutrophil cells: {n_neutrophil}")

        neut_global_idx = np.where(neutrophil_mask)[0]  # global indices of neutrophil cells
        CHUNK = 500  # rows per chunk — tune down if memory is tight

        comat = np.zeros((n_neutrophil, n_neutrophil), dtype=np.float32)

        for r in range(all_labels_array.shape[0]):
            labels_r = all_labels_array[r]
            neut_labels = labels_r[neutrophil_mask]  # labels for neutrophil cells only
            for i_start in range(0, n_neutrophil, CHUNK):
                i_end = min(i_start + CHUNK, n_neutrophil)
                chunk_labels = neut_labels[i_start:i_end]  # (chunk,)
                # co-cluster with all neutrophils: broadcast comparison
                match = (chunk_labels[:, None] == neut_labels[None, :])  # (chunk, n_neutrophil)
                valid = (chunk_labels[:, None] != -1) & (neut_labels[None, :] != -1)
                comat[i_start:i_end] += (match & valid).astype(np.float32)

        comat /= all_labels_array.shape[0]

        # Sort neutrophil cells by hierarchical clustering of their coclustering
        dist_neut = 1.0 - comat
        np.fill_diagonal(dist_neut, 0.0)
        dist_neut = np.clip(dist_neut, 0, None)
        from scipy.spatial.distance import squareform as sq
        lm_neut = linkage(sq(dist_neut), method='average')
        order_neut = leaves_list(lm_neut)

        comat_sorted = comat[np.ix_(order_neut, order_neut)]

        # --- Combined Plot 4 & 5: Co-clustering heatmaps ---
        fig_heat, axes_heat = plt.subplots(1, 2, figsize=(16, 8)) # Increased size
        fig_heat.subplots_adjust(wspace=0.35) # Increased spacing between the two subplots

        # Cell-type Heatmap
        im1 = axes_heat[0].imshow(agg_sorted, vmin=0, vmax=1, cmap='cividis', aspect='auto')
        axes_heat[0].set_yticks(np.arange(n_types))
        axes_heat[0].set_yticklabels(yticklabels, fontsize=14)
        axes_heat[0].set_xticks([])
        axes_heat[0].set_title("Cell-Type Co-Clustering", fontsize=18)

        # Neutrophil Heatmap
        im2 = axes_heat[1].imshow(comat_sorted, vmin=0, vmax=1, cmap='cividis', aspect='auto')
        axes_heat[1].set_xticks([])
        axes_heat[1].set_yticks([])
        axes_heat[1].set_title("Neutrophil Co-Clustering", fontsize=18)

        # Shared colorbar (extends across y-axis bounds of both plots natively)
        cbar = fig_heat.colorbar(im2, ax=axes_heat.ravel().tolist())
        cbar.set_label('Co-clustering probability', size=16)
        cbar.ax.tick_params(labelsize=14)

        plt.savefig(f"{SAVE_DIR}/combined_coclustering_heatmaps.pdf", bbox_inches='tight')
        plt.close(fig_heat)

        print("\n=== DONE: All results saved to", SAVE_DIR, "===")
