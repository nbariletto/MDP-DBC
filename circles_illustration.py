import os
os.environ['PYTHONHASHSEED'] = '2026'

import time
import functools
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import distrax
import optax
import surjectors
from surjectors import TransformedDistribution, Chain, MaskedAutoregressive, Permutation
from surjectors.nn import MADE
from surjectors.util import unstack

from sklearn.datasets import make_circles
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from tqdm import tqdm


# ==============================================================================
# 1. FLOW MODEL DEFINITIONS
# ==============================================================================

def make_flow_model(event_shape: Sequence[int]) -> TransformedDistribution:
    """
    Constructs a Masked Autoregressive Flow (MAF) model using Surjectors and Distrax.
    
    Args:
        event_shape: The dimensionality of the input data.
        
    Returns:
        A distrax TransformedDistribution representing the normalizing flow.
    """
    n_dimensions = int(np.prod(event_shape))

    def bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return distrax.Inverse(
            distrax.ScalarAffine(means, jnp.exp(log_scales))
        )

    layers = []
    order = jnp.arange(n_dimensions)
    
    for i in range(12):
        layer = MaskedAutoregressive(
            conditioner=MADE(n_dimensions, [128, 128], 2),
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


EVENT_SHAPE = (2,)

@hk.without_apply_rng
@hk.transform
def logprob_fn(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluates the log-probability of the input data under the flow model."""
    model = make_flow_model(EVENT_SHAPE)
    return model.log_prob(x)

@hk.transform
def sample_fn(num: int) -> jnp.ndarray:
    """Generates samples from the trained flow model."""
    model = make_flow_model(EVENT_SHAPE)
    return model.sample(sample_shape=(num,))


# ==============================================================================
# 2. CLUSTERING ALGORITHM
# ==============================================================================

def cpu_cluster_optimized(X, sparse_graph, log_prob, threshold, min_points=100, cluster_all=True):
    """
    Performs fast, density-based clustering using precomputed spatial graphs and KD-trees.
    
    Args:
        X: Coordinates of the data points.
        sparse_graph: Precomputed sparse adjacency matrix of distances.
        log_prob: Estimated log-density for each point in X.
        threshold: Density threshold to distinguish core points from noise.
        min_points: Minimum number of points required to form a valid cluster.
        cluster_all: If True, forces all noise points to snap to the nearest valid cluster.
        
    Returns:
        A numpy array of integer cluster labels for each data point.
    """
    N = X.shape[0]
    mask_dense = log_prob > threshold
    n_dense = np.sum(mask_dense)
    
    labels = np.full(N, -1, dtype=int)
    if n_dense == 0:
        return labels
        
    dense_indices = np.where(mask_dense)[0]
    adj_dense = sparse_graph[dense_indices, :][:, dense_indices]
    
    _, labels_dense = connected_components(adj_dense, directed=False)
    labels[dense_indices] = labels_dense
    
    unique_labels, counts = np.unique(labels_dense, return_counts=True)
    large_labels = unique_labels[counts >= min_points]
    small_labels = unique_labels[(counts < min_points) & (counts > 0)]
    
    is_large = np.isin(labels, large_labels)
    is_small = np.isin(labels, small_labels)
    
    if np.any(is_small) and np.any(is_large):
        small_idx = np.where(is_small)[0]
        large_idx = np.where(is_large)[0]
        
        tree_large = cKDTree(X[large_idx])
        _, nearest_large_relative_idx = tree_large.query(X[small_idx], k=1)
        nearest_large_global_idx = large_idx[nearest_large_relative_idx]
        
        labels[small_idx] = labels[nearest_large_global_idx]
        
    if cluster_all:
        is_noise = labels == -1
        is_valid = labels != -1
        
        if np.any(is_noise) and np.any(is_valid):
            noise_idx = np.where(is_noise)[0]
            valid_idx = np.where(is_valid)[0]
            
            tree_valid = cKDTree(X[valid_idx])
            _, nearest_valid_relative_idx = tree_valid.query(X[noise_idx], k=1)
            nearest_valid_global_idx = valid_idx[nearest_valid_relative_idx]
            
            labels[noise_idx] = labels[nearest_valid_global_idx]
            
    unique_valid = np.unique(labels[labels != -1])
    label_map = {old: new for new, old in enumerate(unique_valid)}
    label_map[-1] = -1
    return np.vectorize(label_map.get)(labels)


# ==============================================================================
# 3. TRAINING & UPDATE UTILITIES
# ==============================================================================

def replicate_params(params, T: int):
    """Duplicates network parameters to create an ensemble/particle batch."""
    return jax.tree.map(lambda p: jnp.repeat(p[None, ...], T, axis=0), params)

def make_step_fn(logprob_apply, sample_apply, base_lr):
    """
    Constructs the JIT-compiled update step for sequential particle-based training.
    """
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
        
        def sanitize_and_clip(g):
            # Replace NaNs with 0, and Infs with a bounded number
            g_safe = jnp.nan_to_num(g, nan=0.0, posinf=1.0, neginf=-1.0)
            # Clip the gradient values to prevent massive leaps
            return jnp.clip(g_safe, -100.0, 100.0)
            
        grads_batched = jax.tree.map(sanitize_and_clip, grads_batched)
        
        lr = base_lr / (5000 + step_idx + 1.0)
        params_batched = jax.tree.map(lambda p, g: p - lr * g, params_batched, grads_batched)
        return params_batched, xs, grads_batched
    
    return step


# ==============================================================================
# 4. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    # --- Setup & Data Generation ---
    np.random.seed(2026)
    outdir = "jax_circles_maf_final"
    os.makedirs(outdir, exist_ok=True)

    cluster_all = True
    rng = jax.random.PRNGKey(0)
    rng, init_key = jax.random.split(rng)

    n_total = 5000
    factor = 0.25
    n_outer = int(n_total / (1 + factor))
    n_inner = n_total - n_outer

    X, _ = make_circles(n_samples=(n_outer, n_inner), noise=0.15, factor=factor, random_state=2026)
    X = X.astype(np.float32)
    mean, std = X.mean(0), X.std(0)
    Xn = (X - mean) / std
    X_jnp = jnp.array(Xn)
    X = X.astype(np.float32)
    mean, std = X.mean(0), X.std(0)
    Xn = (X - mean) / std
    X_jnp = jnp.array(Xn)

    # --- Plot Raw Data ---
    plt.figure(figsize=(5, 5))
    plt.scatter(Xn[:, 0], Xn[:, 1], s=5, color='gray', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "raw_data_plot.pdf"))
    plt.close()
    # ------------------------------------------

    # --- Model Initialization ---
    print("Initializing model...")
    params = logprob_fn.init(init_key, jnp.zeros((1, *EVENT_SHAPE)))

    learning_rate, epochs, batch_size = 1e-4, 10000, 5000
    num_batches = max(1, X_jnp.shape[0] // batch_size)
    
    # Fast convergence scheduler: Warmup + Cosine Decay
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=1e-3, 
        warmup_steps=int(epochs * num_batches * 0.05),
        decay_steps=epochs * num_batches,
        end_value=1e-6
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=1e-4)
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        loss = -jnp.mean(logprob_fn.apply(params, batch))
        grads = jax.grad(lambda p: -jnp.mean(logprob_fn.apply(p, batch)))(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    # --- Pre-training the Base Flow ---
    print("Training initial flow (MAF)...")
    pbar_ep = tqdm(range(epochs), desc="Training flow")
    
    rng, train_key = jax.random.split(rng)
    for ep in pbar_ep:
        train_key, epoch_key = jax.random.split(train_key)
        perm = jax.random.permutation(epoch_key, Xn.shape[0])
        perm = np.array(perm)
        for b in range(num_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            batch = jnp.array(Xn[idx])
            params, opt_state, loss = train_step(params, opt_state, batch)
        if (ep + 1) % 25 == 0:
            pbar_ep.set_postfix({"loss": f"{loss:.4f}"})

    initial_logp_np = np.array(jnp.squeeze(logprob_fn.apply(params, X_jnp)))

    # --- Sequential Particle Updates ---
    T, num_updates, base_lr = 1000, 3000, 0.02
    params_batched = replicate_params(params, T)
    step_fn = make_step_fn(logprob_fn.apply, sample_fn.apply, base_lr)

    rng, master_key = jax.random.split(rng)
    saved_logp_list, saved_densities = [], []

    grid_size = 100
    x_min, x_max = Xn[:,0].min() - 0.05, Xn[:,0].max() + 0.05
    y_min, y_max = Xn[:,1].min() - 0.05, Xn[:,1].max() + 0.05
    X_grid, Y_grid = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid_pts = jnp.array(np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1))

    vmapped_grid_logp = jax.vmap(lambda p: jnp.squeeze(logprob_fn.apply(p, grid_pts)), in_axes=(0,))

    print(f"Starting {num_updates} sequential updates...")
    for k in tqdm(range(num_updates), desc="Sequential updates"):
        master_key, subkey = jax.random.split(master_key)
        params_batched, xs_k, grads_k = step_fn(params_batched, subkey, k, T)

    # --- Particle Evaluation ---
    print("Updates complete. Running final evaluation...")
    vmapped_logp = jax.vmap(lambda p: jnp.squeeze(logprob_fn.apply(p, X_jnp)), in_axes=(0,))
    
    chunk_size = 50
    n_total_particles = T
    logps_chunks = []
    
    print(f"Evaluating {n_total_particles} particles in chunks of {chunk_size}...")
    for i in tqdm(range(0, n_total_particles, chunk_size), desc="Eval Chunks"):
        params_chunk = jax.tree.map(lambda x: x[i : i + chunk_size], params_batched)
        chunk_logp = vmapped_logp(params_chunk)
        logps_chunks.append(np.array(chunk_logp))
        
    all_logps = np.concatenate(logps_chunks, axis=0)
    saved_logp_list.append(all_logps)
    
    params_subset = jax.tree.map(lambda x: x[:6], params_batched)
    dens = vmapped_grid_logp(params_subset)
    saved_densities.append(np.array(dens).reshape((6, grid_size, grid_size)))
    all_logps_concat = np.vstack(saved_logp_list)
    
    # --- Graph Construction & Base Clustering ---
    print("Pre-computing Radius and Sparse Neighborhood Graph on CPU...")
    fixed_threshold = np.percentile(initial_logp_np, 10)
    
    mask_dense_init = initial_logp_np > fixed_threshold
    dense_indices_init = np.where(mask_dense_init)[0]
    X_dense_init = Xn[dense_indices_init]
    
    if len(X_dense_init) < 5:
        fixed_radius = 0.1 * 1.2
    else:
        tree_init = cKDTree(X_dense_init)
        dists, _ = tree_init.query(X_dense_init, k=10)
        fixed_radius = np.mean(dists[:, 9]) * 1.2
        
    print(f"Fixed adaptive radius calculated as: {fixed_radius:.4f}")
    
    sparse_graph = radius_neighbors_graph(Xn, radius=fixed_radius, mode='connectivity', include_self=True)

    print("Running initial clustering...")
    labels_initial = cpu_cluster_optimized(Xn, sparse_graph, initial_logp_np, fixed_threshold, 100, cluster_all)

    # --- CPU Clustering Loop ---
    print("Clustering resamples on CPU (sequential)...")
    all_labels_list = []
    for lp in tqdm(all_logps, desc="CPU Clustering"):
        lbls = cpu_cluster_optimized(Xn, sparse_graph, lp, fixed_threshold, 100, cluster_all)
        all_labels_list.append(lbls)
    all_labels_array = np.array(all_labels_list)

    # --- Visualization: Densities ---
    densities_to_plot = [np.array(jnp.squeeze(logprob_fn.apply(params, grid_pts))).reshape(grid_size, grid_size)]
    labels_to_plot = [labels_initial]

    grid_logp = np.array(jnp.squeeze(logprob_fn.apply(params, grid_pts))).reshape(grid_size, grid_size)
    densities_to_plot = [np.exp(grid_logp)]

    if len(saved_densities) > 0:
        for i in range(min(3, saved_densities[0].shape[0])):
            densities_to_plot.append(np.exp(saved_densities[0][i]))
            labels_to_plot.append(all_labels_list[i])

    fig_d, axes_d = plt.subplots(2, 2, figsize=(5, 5))
    for i, ax in enumerate(axes_d.flatten()):
        if i < len(densities_to_plot):
            im = ax.imshow(
                densities_to_plot[i],
                origin='lower',
                extent=[x_min, x_max, y_min, y_max],
                cmap='cividis',
                aspect='auto'
            )
            ax.set_aspect('equal')
            ax.set_xticks([-2,0,2])
            ax.set_yticks([-2,0,2])
        else:
            ax.axis('off')

        if i == 0:
            ax.set_title("Trained Density")
        else:
            ax.set_title(f"Resampled Density {i}")
    
    
    plt.tight_layout()
    fig_d.savefig(os.path.join(outdir, "density_plots.pdf"), dpi=150)
    plt.close(fig_d)

    # --- Visualization: Clustering Assignments ---
    fig_c, axes_c = plt.subplots(2, 2, figsize=(5, 5))
    for i, ax in enumerate(axes_c.flatten()):
        if i < len(labels_to_plot):
            lbls = labels_to_plot[i]
            unique_labels = set(lbls)
            colors = plt.cm.cividis(np.linspace(0, 1, max(1, len(unique_labels))))
            
            for k_lbl, col in zip(unique_labels, colors):
                mask = lbls == k_lbl
                if np.any(mask):
                    s, alpha, c = (5, 0.1, [0.8, 0.8, 0.8, 1]) if k_lbl == -1 else (10, 1.0, [col])
                    ax.scatter(Xn[mask, 0], Xn[mask, 1], c=c, s=s, alpha=alpha)
                    
            ax.set_aspect('equal')
            # ax.grid(True, alpha=0.3)
            ax.set_xticks([-2,0,2])
            ax.set_yticks([-2,0,2])
            
            if i == 0:
                ax.set_title("Trained Clustering")
            else:
                ax.set_title(f"Resampled Clustering {i}")
        else:
            ax.axis('off')
            
    plt.tight_layout()
    fig_c.savefig(os.path.join(outdir, "clustering_plots.pdf"), dpi=150)
    plt.close(fig_c)

    # --- Compute Co-occurrence & Certainty ---
    print("Computing co-occurrence matrix on GPU...")
    
    @jax.jit
    def batched_co_matrix(labels_chunk):
        is_same = labels_chunk[:, :, None] == labels_chunk[:, None, :]
        is_valid = labels_chunk[:, :, None] != -1
        valid_same = is_same & is_valid
        return jnp.sum(valid_same, axis=0, dtype=jnp.float32)

    gpu_chunk_size = 50 
    n_total_labels = all_labels_array.shape[0]
    co_matrix_gpu = jnp.zeros((Xn.shape[0], Xn.shape[0]), dtype=jnp.float32)

    for i in tqdm(range(0, n_total_labels, gpu_chunk_size), desc="GPU Co-occurrence"):
        chunk = jnp.array(all_labels_array[i : i + gpu_chunk_size])
        co_matrix_gpu += batched_co_matrix(chunk)
        
    co_matrix = np.array(co_matrix_gpu / n_total_labels)

    # --- Visualization: Certainty & Distribution ---
    print("Plotting certainty...")
    plt.figure(figsize=(8, 6))
    if cluster_all:
        plot_values = np.mean(np.abs(co_matrix - 0.5)**2, axis=1)
    else:
        plot_values = np.mean(all_labels_array == -1, axis=0)
    
    plt.scatter(Xn[:, 0], Xn[:, 1], c=plot_values, cmap='cividis', s=10, alpha=0.8)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "clustering_certainty.pdf"), dpi=100)
    plt.close()

    n_clusters_found = [len(set(lbls) - {-1}) for lbls in all_labels_array]
    plt.figure(figsize=(8, 6))
    plt.hist(n_clusters_found, bins=np.arange(min(n_clusters_found)-0.5, max(n_clusters_found)+1.5, 1), 
             rwidth=0.8, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(outdir, "cluster_count_histogram.pdf"), dpi=100)
    plt.close()

    # ==================================================================
    # NEW FIGURE 1: Raw data | Trained density | Resampled density 1 | Resampled density 2
    # ==================================================================
    _fs_title = 13
    _fs_tick = 10
    _xlim = (x_min, x_max)
    _ylim = (y_min, y_max)

    fig_new1, axes_new1 = plt.subplots(
        1, 4, figsize=(14, 3.5), constrained_layout=True,
    )

    # Panel 0: Raw data
    ax = axes_new1[0]
    ax.scatter(Xn[:, 0], Xn[:, 1], s=5, color=plt.cm.cividis(0.25), alpha=0.35)
    ax.set_title("Raw data", fontsize=_fs_title)

    # Compute shared color range across all density panels
    _dens_arrays = [densities_to_plot[i] for i in range(min(3, len(densities_to_plot)))]
    _vmin = min(d.min() for d in _dens_arrays)
    _vmax = max(d.max() for d in _dens_arrays)

    # Panel 1: Trained density (densities_to_plot[0])
    ax = axes_new1[1]
    im = ax.imshow(densities_to_plot[0], origin='lower',
                   extent=[x_min, x_max, y_min, y_max], cmap='cividis',
                   aspect='equal', vmin=_vmin, vmax=_vmax)
    ax.set_title("Trained density", fontsize=_fs_title)

    # Panel 2: Resampled density 1 (densities_to_plot[1])
    ax = axes_new1[2]
    if len(densities_to_plot) > 1:
        ax.imshow(densities_to_plot[1], origin='lower',
                  extent=[x_min, x_max, y_min, y_max], cmap='cividis',
                  aspect='equal', vmin=_vmin, vmax=_vmax)
    ax.set_title("Resampled density 1", fontsize=_fs_title)

    # Panel 3: Resampled density 2 (densities_to_plot[2])
    ax = axes_new1[3]
    if len(densities_to_plot) > 2:
        ax.imshow(densities_to_plot[2], origin='lower',
                  extent=[x_min, x_max, y_min, y_max], cmap='cividis',
                  aspect='equal', vmin=_vmin, vmax=_vmax)
    ax.set_title("Resampled density 2", fontsize=_fs_title)

    for j, ax in enumerate(axes_new1):
        ax.set_xlim(_xlim)
        ax.set_ylim(_ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(labelsize=_fs_tick)
        if j > 0:
            ax.set_yticklabels([])

    fig_new1.colorbar(im, ax=axes_new1[1:].tolist(), fraction=0.046, pad=0.02)
    fig_new1.savefig(os.path.join(outdir, "circles_raw_and_densities.pdf"), dpi=150)
    plt.close(fig_new1)

    # ==================================================================
    # NEW FIGURE 2: Trained clustering | Resampled clust. 1 | Resampled clust. 2 | Certainty
    # ==================================================================
    fig_new2, axes_new2 = plt.subplots(
        1, 4, figsize=(14, 3.5), constrained_layout=True,
    )

    def _scatter_clustering(ax, lbls):
        unique_lbls = sorted(set(lbls))
        colors = plt.cm.cividis(np.linspace(0, 1, max(1, len(unique_lbls))))
        for k_lbl, col in zip(unique_lbls, colors):
            mask = lbls == k_lbl
            if np.any(mask):
                if k_lbl == -1:
                    ax.scatter(Xn[mask, 0], Xn[mask, 1], c=[[0.8, 0.8, 0.8, 1]],
                               s=5, alpha=0.1)
                else:
                    ax.scatter(Xn[mask, 0], Xn[mask, 1], c=[col], s=10, alpha=1.0)

    # Panel 0: Trained clustering (labels_to_plot[0])
    ax = axes_new2[0]
    _scatter_clustering(ax, labels_to_plot[0])
    ax.set_title("Trained clustering", fontsize=_fs_title)

    # Panel 1: Resampled clustering 1 (labels_to_plot[1])
    ax = axes_new2[1]
    if len(labels_to_plot) > 1:
        _scatter_clustering(ax, labels_to_plot[1])
    ax.set_title("Resampled clustering 1", fontsize=_fs_title)

    # Panel 2: Resampled clustering 2 (labels_to_plot[2])
    ax = axes_new2[2]
    if len(labels_to_plot) > 2:
        _scatter_clustering(ax, labels_to_plot[2])
    ax.set_title("Resampled clustering 2", fontsize=_fs_title)

    # Panel 3: Coclustering certainty score
    ax = axes_new2[3]
    sc = ax.scatter(Xn[:, 0], Xn[:, 1], c=plot_values, cmap='cividis', s=10, alpha=0.8)
    ax.set_title("Coclust. certainty score", fontsize=_fs_title)
    fig_new2.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    for j, ax in enumerate(axes_new2):
        ax.set_xlim(_xlim)
        ax.set_ylim(_ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(labelsize=_fs_tick)
        if j > 0:
            ax.set_yticklabels([])

    fig_new2.savefig(os.path.join(outdir, "circles_clust_and_certainty.pdf"), dpi=150)
    plt.close(fig_new2)

    np.savez(os.path.join(outdir, "saved_results.npz"), X=Xn, all_logps=all_logps_concat, initial_logp=initial_logp_np)
    print("Done. Results saved to", outdir)
    
if __name__ == "__main__":
    main()