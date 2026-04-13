import os
import sys

# ==========================================
# 0. REPRODUCIBILITY ENVIRONMENT SETUP
# (Must happen before ANY library imports)
# ==========================================
SEED = 2026

if os.environ.get("PYTHONHASHSEED") != str(SEED):
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.execv(sys.executable, ['python'] + sys.argv)

# Limit threading to prevent non-deterministic floating-point reordering
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Required for PyTorch use_deterministic_algorithms on GPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

import time
import functools
from typing import Sequence
import random
import numpy as np
import argparse
import subprocess

# ---- PyTorch (for AE) ----
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ---- JAX + flow stack ----
import jax
import jax.numpy as jnp
import haiku as hk
import distrax
import optax
import surjectors
from surjectors import TransformedDistribution, Chain, MaskedAutoregressive, Permutation
from surjectors.nn import MADE
from surjectors.util import unstack

# ---- clustering, plotting, utilities ----
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm
import gudhi.clustering.tomato as tomato
import gc

# ---- CBI (Conformalized Bayesian Inference) ----
try:
    from cbi_partitions import PartitionKDE
except ImportError:
    print("Warning: cbi_partitions library not found. Please install it via `pip install https://github.com/nbariletto/cbi_partitions/archive/main.zip' to run the CBI section.")
    PartitionKDE = None

# ==========================================
# CONFIGURATION
# ==========================================
SAVE_DIR = "./mnist_results_oos"
os.makedirs(SAVE_DIR, exist_ok=True)

DIGIT_PAIR = [3, 8]
LATENT_DIM = 24
BATCH_SIZE = 128

# JAX Config
jax.config.update("jax_enable_x64", False)
rng = jax.random.PRNGKey(SEED)

# Strict seeding
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_data():
    """Load training data (first 5000) and OOS data (next 500+500 per digit)."""
    print(f"Loading MNIST (Subset: {DIGIT_PAIR})...")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    targets = dataset.targets.numpy()
    mask = np.isin(targets, DIGIT_PAIR)
    indices_subset = np.where(mask)[0]
    
    # --- Training set: first 5000 ---
    train_indices = indices_subset[:5000]
    
    X_list = []
    T_list = []
    for idx in train_indices:
        x, y = dataset[idx]
        X_list.append(x.numpy())
        T_list.append(y)
        
    X = np.array(X_list) 
    T = np.array(T_list)
    X_jax = X.transpose(0, 2, 3, 1)
    
    # --- OOS set: from remaining indices, pick 500 per digit ---
    remaining_indices = indices_subset[5000:]
    remaining_targets = targets[remaining_indices]
    
    d0, d1 = DIGIT_PAIR
    oos_idx_d0 = remaining_indices[remaining_targets == d0][:500]
    oos_idx_d1 = remaining_indices[remaining_targets == d1][:500]
    oos_indices = np.concatenate([oos_idx_d0, oos_idx_d1])
    
    X_oos_list = []
    T_oos_list = []
    for idx in oos_indices:
        x, y = dataset[idx]
        X_oos_list.append(x.numpy())
        T_oos_list.append(y)
    
    X_oos = np.array(X_oos_list)
    T_oos = np.array(T_oos_list)
    
    print(f"Training samples: {len(X)}, OOS samples: {len(X_oos)} "
          f"(digit {d0}: {len(oos_idx_d0)}, digit {d1}: {len(oos_idx_d1)})")
    
    return X_jax, T, X, X_oos, T_oos

# ==========================================
# 2. PYTORCH AUTOENCODER
# ==========================================
class AutoencoderPT(nn.Module):
    def __init__(self, latent_dim=10, input_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 7), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Unflatten(1, (128, 1, 1)),
            nn.ConvTranspose2d(128, 64, 7), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==========================================
# 3. FLOW MODEL
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
    model = make_flow_model(LATENT_DIM, flow_depth=16)
    return model.log_prob(x)

@hk.transform
def sample_fn(num: int) -> jnp.ndarray:
    model = make_flow_model(LATENT_DIM, flow_depth=16)
    return model.sample(sample_shape=(num,))

# ==========================================
# 4. HELPER FUNCTIONS
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
        
    t.merge_threshold_ = 0.5
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
        lr = base_lr / (5000 + step_idx)
        params_batched = jax.tree.map(lambda p, g: p - lr * g, params_batched, grads_batched)
        return params_batched
    return step

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_cbi_only", action="store_true", help="Internal flag to run only CBI stage")
    args = parser.parse_args()

    # =========================================================
    # PART 1: AE, FLOW, CLUSTERING (Run if --run_cbi_only is False)
    # =========================================================
    if not args.run_cbi_only:
        print(f"Using JAX Device: {jax.devices()[0]}")

        # --- 1. Load Data (train + OOS) ---
        X_jax, targets, X_plot, X_oos, T_oos = load_data()
        print(f"Total training samples: {len(X_plot)}")

        # --- 2. Train PyTorch Autoencoder (on training data only) ---
        print("\n=== STAGE: TRAINING PYTORCH AE (MSE) ===")
        device_pt = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ae_pt = AutoencoderPT(latent_dim=LATENT_DIM).to(device_pt)

        tensor_images = torch.from_numpy(X_plot).float()
        tensor_targets = torch.from_numpy(targets).long()
        pt_dataset = torch.utils.data.TensorDataset(tensor_images, tensor_targets)
        
        g = torch.Generator()
        g.manual_seed(SEED)
        pt_loader = torch.utils.data.DataLoader(
            pt_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, 
            worker_init_fn=seed_worker, generator=g
        )

        optimizer_pt = optim.Adam(ae_pt.parameters(), lr=1e-3)

        epochs_ae = 25
        ae_pt.train()
        for ep in range(epochs_ae):
            pbar = tqdm(pt_loader, desc=f"AE Reconstruction Epoch {ep+1}/{epochs_ae}")
            for x_batch, _ in pbar:
                x_batch = x_batch.to(device_pt)
                z = ae_pt.encode(x_batch)
                x_rec = ae_pt.decode(z)
                loss_recon = F.mse_loss(x_rec, x_batch, reduction='sum') / x_batch.size(0)
                optimizer_pt.zero_grad()
                loss_recon.backward()
                optimizer_pt.step()
                pbar.set_postfix({'Rec': f"{loss_recon.item():.6f}"})

        # --- Extract latent for TRAINING data (to compute z_mean, z_std and train flow) ---
        print("Extracting Latent Data from PyTorch AE (training set)...")
        ae_pt.eval()
        Z_list = []
        X_list = []
        T_list = []
        
        eval_loader = torch.utils.data.DataLoader(pt_dataset, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for x_batch, y_batch in tqdm(eval_loader, desc="Extracting latents"):
                x_batch = x_batch.to(device_pt)
                z = ae_pt.encode(x_batch)
                Z_list.append(z.cpu().numpy())
                X_list.append(x_batch.cpu().numpy())
                T_list.append(y_batch.numpy())
                
        Z_np_all = np.vstack(Z_list)
        X_plot = np.vstack(X_list)
        targets = np.concatenate(T_list)

        z_mean = np.mean(Z_np_all, axis=0)
        z_std = np.std(Z_np_all, axis=0) + 1e-6
        Z_norm_np = (Z_np_all - z_mean) / z_std

        Z_norm = jnp.array(Z_norm_np, dtype=jnp.float32)
        Z_np = np.array(Z_norm_np)

        # --- Extract latent for OOS data ---
        print("Extracting Latent Data from PyTorch AE (OOS set)...")
        tensor_oos = torch.from_numpy(X_oos).float()
        tensor_oos_targets = torch.from_numpy(T_oos).long()
        oos_dataset = torch.utils.data.TensorDataset(tensor_oos, tensor_oos_targets)
        oos_loader = torch.utils.data.DataLoader(oos_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        Z_oos_list = []
        X_oos_list = []
        T_oos_list = []
        with torch.no_grad():
            for x_batch, y_batch in tqdm(oos_loader, desc="Extracting OOS latents"):
                x_batch = x_batch.to(device_pt)
                z = ae_pt.encode(x_batch)
                Z_oos_list.append(z.cpu().numpy())
                X_oos_list.append(x_batch.cpu().numpy())
                T_oos_list.append(y_batch.numpy())
        
        Z_oos_np_all = np.vstack(Z_oos_list)
        X_oos_plot = np.vstack(X_oos_list)
        targets_oos = np.concatenate(T_oos_list)
        
        # Normalize OOS with its own statistics
        z_oos_mean = np.mean(Z_oos_np_all, axis=0)
        z_oos_std = np.std(Z_oos_np_all, axis=0) + 1e-6
        Z_oos_norm_np = (Z_oos_np_all - z_oos_mean) / z_oos_std
        
        print(f"OOS latent shape: {Z_oos_np_all.shape}")

        # --- Build COMBINED latent array [train ; OOS] for clustering ---
        N_train = Z_np.shape[0]
        N_oos = Z_oos_norm_np.shape[0]
        Z_combined_np = np.vstack([Z_np, Z_oos_norm_np])            # (6000, 24)
        Z_combined_jax = jnp.array(Z_combined_np, dtype=jnp.float32)
        
        # OOS indices within the combined array
        oos_slice = slice(N_train, N_train + N_oos)
        print(f"Combined latent shape: {Z_combined_np.shape}  "
              f"(train 0:{N_train}, OOS {N_train}:{N_train+N_oos})")

        # --- 3. Train Flow (on training data) ---
        print("\n=== STAGE: TRAINING MAF FLOW ===")
        # Re-initialize JAX random key safely
        rng, key = jax.random.split(jax.random.PRNGKey(SEED)) 
        params_flow = logprob_fn.init(key, jnp.zeros((1, LATENT_DIM)))
        opt_flow = optax.adam(1e-4)
        opt_state_flow = opt_flow.init(params_flow)

        @jax.jit
        def train_step_flow(params, opt_state, batch):
            def loss_fn(p, x):
                return -jnp.mean(logprob_fn.apply(p, x))
            loss, grads = jax.value_and_grad(loss_fn)(params, batch)
            updates, new_state = opt_flow.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_state, loss

        epochs_flow = 15
        pbar = tqdm(range(epochs_flow), desc="Flow Training")
        
        # Use an isolated generator for shuffling
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

        # --- Evaluate base density on COMBINED set ---
        base_log_probs_combined = np.array(logprob_fn.apply(params_flow, Z_combined_jax))

        # --- 4. CLUSTERING & PLOT 1 (on combined data, plots scoped to OOS) ---
        print("\n=== STAGE: BASE CLUSTERING (Combined) ===")
        # KNN graph on combined latents
        nbrs_comb = NearestNeighbors(n_neighbors=20, algorithm='brute', n_jobs=1).fit(Z_combined_np)
        knn_graph_comb = nbrs_comb.kneighbors_graph(Z_combined_np, mode='connectivity')
        rows_comb, cols_comb = knn_graph_comb.nonzero()
        adj_list_comb = [cols_comb[knn_graph_comb.indptr[i]:knn_graph_comb.indptr[i+1]] for i in range(knn_graph_comb.shape[0])]

        # --- Standalone persistence diagram for baseline trained density (early diagnostic) ---
        labels_pre_comb, _ = get_tomato_clusters(
            Z_combined_np, base_log_probs_combined, adj_list_comb, 
            plot_path=f"{SAVE_DIR}/baseline_persistence_oos.pdf",
            title_suffix="Trained Density (Combined)"
        )

        # --- SETUP COMBINED PLOT (2 Rows, 2 Columns) ---
        fig_pers, axes_pers = plt.subplots(2, 2, figsize=(5, 5))
        axes_flat = axes_pers.flatten()

        # Re-run on first axis of the combined plot
        labels_pre_comb, _ = get_tomato_clusters(
            Z_combined_np, base_log_probs_combined, adj_list_comb, 
            ax=axes_flat[0], 
            title_suffix="Trained Density"
        )

        # Co-clustering from base clustering — OOS samples only
        labels_pre_oos = labels_pre_comb[oos_slice]
        adjacency_pre_oos = (labels_pre_oos[:, None] == labels_pre_oos[None, :]) & (labels_pre_oos[:, None] != -1)

        plt.figure(figsize=(5, 4))
        plt.imshow(adjacency_pre_oos, cmap='cividis', interpolation='nearest', vmin=0, vmax=1)
        b = 500
        plt.axhline(b, color='white', linewidth=5, alpha=1)
        plt.axvline(b, color='white', linewidth=5, alpha=1)
        unique_digits = DIGIT_PAIR
        midpoints = [250, 750]
        plt.xticks(midpoints, unique_digits)
        plt.yticks(midpoints, unique_digits)
        plt.colorbar(label='Co-Clustering Probability')
        plt.xlabel('Digit Class')
        plt.ylabel('Digit Class')
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/pretrained_clustering_oos.pdf")
        plt.close()

        # --- 5. RESAMPLING ---
        print("\n=== STAGE: RESAMPLING (JAX Parallel) ===")
        NUM_RESAMPLES = 500 
        NUM_UPDATES = 3000
        BASE_LR = 0.002

        params_batched = replicate_params(params_flow, NUM_RESAMPLES)
        step_fn = make_step_fn(logprob_fn.apply, sample_fn.apply, BASE_LR)
        
        master_key = jax.random.PRNGKey(SEED)
        
        print(f"Running {NUM_RESAMPLES} chains for {NUM_UPDATES} steps...")
        for k in tqdm(range(NUM_UPDATES)):
            master_key, subkey = jax.random.split(master_key)
            params_batched = step_fn(params_batched, subkey, k, NUM_RESAMPLES)
        
        # --- Evaluate resampled densities on COMBINED data ---
        print("Evaluating resampled densities on combined data...")
        vmapped_logp = jax.vmap(lambda p, x: logprob_fn.apply(p, x), in_axes=(0, None))
        all_logps_list = []
        chunk_size = 50
        for i in tqdm(range(0, NUM_RESAMPLES, chunk_size)):
            p_chunk = jax.tree.map(lambda x: x[i:i+chunk_size], params_batched)
            lp = vmapped_logp(p_chunk, Z_combined_jax)
            all_logps_list.append(np.array(lp))
        all_logps = np.vstack(all_logps_list)   # (500, 6000)

        del params_batched
        gc.collect()

        # --- Cluster all resamples on COMBINED data ---
        print("Clustering all resamples (combined)...")
        all_labels_list = []
        n_clusters_history = []
        
        for i in tqdm(range(NUM_RESAMPLES)):
            if i < 3:
                target_ax = axes_flat[i+1]
            else:
                target_ax = None
                
            lbs, n_cl = get_tomato_clusters(
                Z_combined_np, all_logps[i], adj_list_comb, 
                ax=target_ax,
                title_suffix=f"Resampled Density {i}"
            )
            
            if i == 4:
                plt.tight_layout()
                plt.savefig(f"{SAVE_DIR}/combined_persistence_diagrams_oos.pdf")
                plt.close(fig_pers)
                
            all_labels_list.append(lbs)
            n_clusters_history.append(n_cl)
            
        del all_logps
        del all_logps_list
        gc.collect()

        all_labels_array = np.vstack(all_labels_list)  # (500, 6000)

        # --- 6. ANALYSIS — All samples ---
        print("\n=== STAGE: ANALYSIS (Training + OOS) ===")
        print("Computing certainty scores for all samples and co-clustering for OOS...")
        
        num_resamples = all_labels_array.shape[0]
        num_total = all_labels_array.shape[1]
        
        # Co-clustering matrix (OOS subset remains for consistency in heatmap)
        oos_labels = all_labels_array[:, oos_slice]
        num_oos = oos_labels.shape[1]
        co_matrix = np.zeros((num_oos, num_oos), dtype=np.float32)
        for r in range(num_resamples):
            l_v = oos_labels[r]
            valid_mask = (l_v[:, None] != -1) & (l_v[None, :] != -1)
            match = (l_v[:, None] == l_v[None, :]) & valid_mask
            co_matrix += match.astype(np.float32)
        co_matrix /= num_resamples

        # Certainty score for ALL samples (training + OOS)
        certainty_score = np.zeros(num_total, dtype=np.float32)
        chunk_size = 500
        for start in range(0, num_total, chunk_size):
            end = min(start + chunk_size, num_total)
            P_chunk = np.zeros((end - start, num_total), dtype=np.float32)
            chunk_labels = all_labels_array[:, start:end]
            for r in range(num_resamples):
                l_all = all_labels_array[r]
                l_chunk = chunk_labels[r]
                valid_mask = (l_chunk[:, None] != -1) & (l_all[None, :] != -1)
                match = (l_chunk[:, None] == l_all[None, :]) & valid_mask
                P_chunk += match.astype(np.float32)
            P_chunk /= num_resamples
            certainty_score[start:end] = np.mean(np.abs(P_chunk - 0.5)**2, axis=1)
            del P_chunk
            gc.collect()

        # --- PLOTTING ---
        plt.figure(figsize=(5, 4))
        plt.imshow(co_matrix, cmap='cividis', vmin=0, vmax=1)
        b = 500
        plt.axhline(b, color='white', linewidth=5, alpha=1)
        plt.axvline(b, color='white', linewidth=5, alpha=1)
        unique_digits = DIGIT_PAIR
        midpoints = [250, 750]
        plt.xticks(midpoints, unique_digits)
        plt.yticks(midpoints, unique_digits)
        plt.colorbar(label='Co-Clustering Probability')
        plt.xlabel('Digit Class')
        plt.ylabel('Digit Class')
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/co_clustering_matrix_grid_oos.pdf")
        plt.close()

        # Plot overall lower score digits from the combined pool
        X_combined_plot = np.vstack([X_plot, X_oos_plot])
        indices_low_certainty = np.argsort(certainty_score)[:25]

        fig, axes = plt.subplots(5, 5, figsize=(8, 8))
        axes = axes.flatten()
        for i, idx in enumerate(indices_low_certainty):
            ax = axes[i]
            img = X_combined_plot[idx].squeeze()
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Score: {certainty_score[idx]:.3f}", fontsize=15)
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/uncertain_digits_oos.pdf")
        plt.close()

        subset_ranks = [7, 10, 18, 21]
        fig_sub, axes_sub = plt.subplots(2, 2, figsize=(4, 4))
        axes_sub = axes_sub.flatten()
        for i, rank in enumerate(subset_ranks):
            ax = axes_sub[i]
            idx = indices_low_certainty[rank]
            img = X_combined_plot[idx].squeeze()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/uncertain_digits_subset_oos.pdf")
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.hist(n_clusters_history, bins=np.arange(min(n_clusters_history)-0.5, max(n_clusters_history)+1.5, 1), edgecolor='black', alpha=0.7)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Frequency")
        plt.xticks(np.arange(min(n_clusters_history), max(n_clusters_history)+1, 1))
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/cluster_count_histogram_oos.pdf")
        plt.close()
        
        # =========================================================
        # --- COMBINED PLOT (in the paper) ---
        # =========================================================
        print("Generating final combined composite plot for paper...")
        
        # Increase font size for paper readability globally temporarily
        old_params = plt.rcParams.copy()
        plt.rcParams.update({
            'font.size': 24, 
            'axes.titlesize': 26, 
            'axes.labelsize': 24,
            'xtick.labelsize': 22,
            'ytick.labelsize': 22,
            'legend.fontsize': 22
        })
        
        fig_combined, axs_all = plt.subplots(
            1, 5, 
            figsize=(32, 9), 
            gridspec_kw={'width_ratios': [1, 0.25, 1, 0.50, 1], 'wspace': 0}
        )
        axs_all[1].set_visible(False)
        axs_all[3].set_visible(False)
        axs_combined = [axs_all[0], axs_all[2], axs_all[4]]
        
        # Force perfectly square bounding boxes for all three subplots to ensure flawless alignment
        axs_combined[0].set_box_aspect(1)
        axs_combined[1].set_box_aspect(1)
        axs_combined[2].set_box_aspect(1)
        
        # 1. Histogram (converted to proportion)
        weights = np.ones_like(n_clusters_history) / len(n_clusters_history)
        bins = np.arange(min(n_clusters_history)-0.5, max(n_clusters_history)+1.5, 1)
        axs_combined[0].hist(n_clusters_history, bins=bins, weights=weights, edgecolor='black', alpha=0.7)
        axs_combined[0].set_xlabel("Number of Clusters")
        axs_combined[0].set_ylabel("Martingale posterior probability")
        axs_combined[0].set_xticks(np.arange(min(n_clusters_history), max(n_clusters_history)+1, 1))
        axs_combined[0].grid(axis='y', alpha=0.3)
        
        # 2. Co-clustering matrix
        im = axs_combined[1].imshow(co_matrix, cmap='cividis', vmin=0, vmax=1, aspect='auto')
        axs_combined[1].axhline(500, color='white', linewidth=4, alpha=1)
        axs_combined[1].axvline(500, color='white', linewidth=4, alpha=1)
        axs_combined[1].set_xticks([250, 750])
        axs_combined[1].set_xticklabels(DIGIT_PAIR)
        axs_combined[1].set_yticks([250, 750])
        axs_combined[1].set_yticklabels(DIGIT_PAIR)
        axs_combined[1].set_xlabel('Digit Class')
        axs_combined[1].set_ylabel('Digit Class')
        
        cax = axs_combined[1].inset_axes([1.05, 0.0, 0.05, 1.0])
        cb = fig_combined.colorbar(im, cax=cax)
        cb.set_label('Co-Clustering Probability', labelpad=15)
        
        # 3. 4 Uncertain Digits Subset 
        img_h, img_w = X_combined_plot[0].squeeze().shape
        composite_img = np.zeros((img_h * 2, img_w * 2))
        for i, rank in enumerate(subset_ranks):
            idx = indices_low_certainty[rank]
            row, col = divmod(i, 2)
            composite_img[row*img_h:(row+1)*img_h, col*img_w:(col+1)*img_w] = X_combined_plot[idx].squeeze()
            
        axs_combined[2].imshow(composite_img, cmap='gray', aspect='auto')
        axs_combined[2].axis('off')
        # Add internal white crossbars to visually separate the 4 sub-images
        axs_combined[2].axhline(img_h - 0.5, color='white', linewidth=4)
        axs_combined[2].axvline(img_w - 0.5, color='white', linewidth=4)
        
        plt.savefig(f"{SAVE_DIR}/final_combined_plot_oos.pdf", bbox_inches='tight')
        plt.close(fig_combined)
        
        plt.rcParams.update(old_params)

        # --- SAVE CHECKPOINTS FOR STAGE 2 ---
        print("\n=== INTERMISSION: SAVING DATA & RESTARTING FOR CBI ===")
        np.save("partitions_oos.npy", oos_labels)
        np.save("targets_oos.npy", targets_oos)
        
        cmd = [sys.executable, sys.argv[0], "--run_cbi_only"]
        subprocess.check_call(cmd)
        sys.exit(0)

    # =========================================================
    # PART 2: CBI CONFORMAL ANALYSIS (Only runs if --run_cbi_only is True)
    # =========================================================
    else:
        print("\n=== STAGE: CBI CONFORMAL ANALYSIS (Fresh Process, OOS) ===")
        if PartitionKDE is None:
            print("Skipping CBI analysis because 'cbi_partitions' is not installed.")
        else:
            partitions_all = np.load("partitions_oos.npy").astype(np.int64)
            targets = np.load("targets_oos.npy")
            NUM_RESAMPLES = partitions_all.shape[0]
            n_split = NUM_RESAMPLES // 2
            train_partitions = partitions_all[:n_split]
            calib_partitions = partitions_all[n_split:]
            kde = PartitionKDE(train_partitions=train_partitions, metric='vi', gamma=0.5, subsample_size=None)
            kde.calibrate(calib_partitions)
            p_val_true = kde.compute_p_value(targets.astype(np.int64))
            print(f"\n>>> Conformal p-value for True Labeling (Digits {DIGIT_PAIR}, OOS): {p_val_true:.4f} <<<")

        if os.path.exists("partitions_oos.npy"): os.remove("partitions_oos.npy")
        if os.path.exists("targets_oos.npy"): os.remove("targets_oos.npy")
