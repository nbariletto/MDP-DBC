import os
os.environ['PYTHONHASHSEED'] = '2026'

import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ==============================================================================
# 1. GMM DENSITY ESTIMATOR
# ==============================================================================

def fit_gmm(X, n_components=4):
    """
    Fits a Gaussian mixture model via EM and returns the parameters.

    Args:
        X: 1-d data array of shape (N,).
        n_components: Number of mixture components.

    Returns:
        A dict with keys 'weights', 'means', 'stds' (all 1-d arrays).
    """
    gm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=500,
        n_init=5,
        random_state=2026,
    )
    gm.fit(X[:, None])
    weights = gm.weights_
    means = gm.means_.ravel()
    stds = np.sqrt(gm.covariances_.ravel())
    return {'weights': weights, 'means': means, 'stds': stds}


def gmm_log_prob(x, params):
    """
    Evaluates log p(x) under a Gaussian mixture.

    Args:
        x: Array of query points.
        params: Dict with 'weights', 'means', 'stds'.

    Returns:
        Array of log-densities.
    """
    weights, means, stds = params['weights'], params['means'], params['stds']
    components = np.array([
        w * norm.pdf(x, loc=m, scale=s)
        for w, m, s in zip(weights, means, stds)
    ])
    return np.log(np.sum(components, axis=0) + 1e-30)


def gmm_prob(x, params):
    """Evaluates p(x) under a Gaussian mixture."""
    return np.exp(gmm_log_prob(x, params))


def gmm_sample(params, n, rng):
    """
    Draws n samples from a Gaussian mixture.

    Args:
        params: Dict with 'weights', 'means', 'stds'.
        n: Number of samples to draw.
        rng: numpy random Generator.

    Returns:
        Array of shape (n,).
    """
    weights, means, stds = params['weights'], params['means'], params['stds']
    comp = rng.choice(len(weights), size=n, p=weights)
    return rng.normal(loc=means[comp], scale=stds[comp])


def gmm_score(x, params):
    """
    Computes the score (gradient of log-density w.r.t. params) for a GMM.
    Returns gradients for weights (in unconstrained softmax space), means,
    and log-stds.

    Args:
        x: Scalar or array of shape (n,).
        params: Dict with 'weights', 'means', 'stds'.

    Returns:
        Dict with 'grad_logit_weights', 'grad_means', 'grad_log_stds'.
    """
    weights, means, stds = params['weights'], params['means'], params['stds']
    K = len(weights)
    x = np.atleast_1d(x)
    N = len(x)

    # Component densities: (K, N)
    comp_densities = np.array([
        w * norm.pdf(x, loc=m, scale=s)
        for w, m, s in zip(weights, means, stds)
    ])
    mix_density = np.sum(comp_densities, axis=0) + 1e-30  # (N,)
    responsibilities = comp_densities / mix_density  # (K, N)

    # d log p / d mu_k  =  sum_n r_k(x_n) * (x_n - mu_k) / sigma_k^2
    grad_means = np.array([
        np.mean(responsibilities[k] * (x - means[k]) / (stds[k] ** 2))
        for k in range(K)
    ])

    # d log p / d log(sigma_k)
    grad_log_stds = np.array([
        np.mean(responsibilities[k] * (((x - means[k]) ** 2) / (stds[k] ** 2) - 1.0))
        for k in range(K)
    ])

    # d log p / d logit_k  (softmax parameterisation for weights)
    # Using the identity: d log p / d logit_k = r_k(x) - w_k
    grad_logit_weights = np.array([
        np.mean(responsibilities[k] - weights[k])
        for k in range(K)
    ])

    return {
        'grad_logit_weights': grad_logit_weights,
        'grad_means': grad_means,
        'grad_log_stds': grad_log_stds,
    }


def params_to_unconstrained(params):
    """Maps constrained GMM params to unconstrained space."""
    log_weights = np.log(params['weights'] + 1e-30)
    logits = log_weights - np.mean(log_weights)
    return {
        'logits': logits,
        'means': params['means'].copy(),
        'log_stds': np.log(params['stds']),
    }


def unconstrained_to_params(unc):
    """Maps unconstrained params back to constrained GMM params."""
    logits = unc['logits']
    weights = np.exp(logits - np.max(logits))
    weights /= weights.sum()
    return {
        'weights': weights,
        'means': unc['means'].copy(),
        'stds': np.exp(unc['log_stds']),
    }


# ==============================================================================
# 2. LEVEL-SET CLUSTERING (1-D)
# ==============================================================================

def level_set_cluster_1d(grid, log_density, threshold):
    """
    Performs level-set clustering on a 1-d grid.

    Points above the threshold are grouped into connected intervals.
    Each connected interval becomes one cluster.

    Args:
        grid: 1-d sorted array of evaluation points.
        log_density: Log-density evaluated at each grid point.
        threshold: Log-density cutoff.

    Returns:
        Integer label array (same length as grid). -1 means noise.
    """
    above = log_density > threshold
    labels = np.full(len(grid), -1, dtype=int)
    cluster_id = 0
    in_cluster = False

    for i in range(len(grid)):
        if above[i]:
            if not in_cluster:
                cluster_id_current = cluster_id
                cluster_id += 1
                in_cluster = True
            labels[i] = cluster_id_current
        else:
            in_cluster = False

    return labels


def level_set_cluster_1d_all(grid, log_density, threshold):
    """
    Like level_set_cluster_1d but assigns every point to the nearest
    cluster (no noise label). Points below threshold snap to the
    closest cluster by Euclidean distance on the grid. Not shown in
    paper, left here for illustration.

    Args:
        grid: 1-d sorted array of evaluation points.
        log_density: Log-density evaluated at each grid point.
        threshold: Log-density cutoff.

    Returns:
        Integer label array (same length as grid). No -1 entries.
    """
    labels = level_set_cluster_1d(grid, log_density, threshold)

    noise_mask = labels == -1
    if not np.any(noise_mask) or not np.any(labels != -1):
        return labels

    valid_idx = np.where(labels != -1)[0]
    noise_idx = np.where(noise_mask)[0]

    # For each noise point, find nearest valid grid point
    nearest = np.searchsorted(valid_idx, noise_idx, side='left')
    nearest = np.clip(nearest, 0, len(valid_idx) - 1)

    # Check left neighbour too
    left = np.clip(nearest - 1, 0, len(valid_idx) - 1)
    dist_right = np.abs(grid[noise_idx] - grid[valid_idx[nearest]])
    dist_left = np.abs(grid[noise_idx] - grid[valid_idx[left]])
    use_left = dist_left < dist_right
    best = np.where(use_left, left, nearest)

    labels[noise_idx] = labels[valid_idx[best]]
    return labels


# ==============================================================================
# 3. SCORE-BASED PARTICLE UPDATES
# ==============================================================================

def run_particle_updates(base_params, T, num_updates, base_lr, N_data, seed=42):
    """
    Runs T independent particle chains starting from base_params.

    Each particle samples from its own current density and takes a
    stochastic gradient ascent step on the log-likelihood of that
    self-generated sample, with a decaying learning rate lr = base_lr / (N_data + k).

    Args:
        base_params: Fitted GMM parameters (dict).
        T: Number of particles.
        num_updates: Number of sequential update steps per particle.
        base_lr: Base learning rate.
        N_data: Number of observed data points (controls the decay schedule).
        seed: RNG seed.

    Returns:
        List of T final GMM parameter dicts.
    """
    rng = np.random.default_rng(seed)

    particles = []
    for t in range(T):
        particles.append(params_to_unconstrained(base_params))

    for k in tqdm(range(num_updates), desc="Particle updates"):
        lr = base_lr / (N_data + k + 1.0)
        for t in range(T):
            p = unconstrained_to_params(particles[t])
            x_sample = gmm_sample(p, n=1, rng=rng)
            grads = gmm_score(x_sample, p)

            # Gradient ascent on log-likelihood in unconstrained space
            particles[t]['logits'] += lr * np.clip(grads['grad_logit_weights'], -10, 10)
            particles[t]['means'] += lr * np.clip(grads['grad_means'], -10, 10)
            particles[t]['log_stds'] += lr * np.clip(grads['grad_log_stds'], -10, 10)

    return [unconstrained_to_params(p) for p in particles]


# ==============================================================================
# 4. MAIN EXECUTION PIPELINE
# ==============================================================================

def _contiguous_segments(mask, grid):
    """
    Given a boolean mask over a sorted grid, returns a list of (x_start, x_end)
    intervals where mask is True.
    """
    segments = []
    in_seg = False
    for i in range(len(mask)):
        if mask[i] and not in_seg:
            start = grid[i]
            in_seg = True
        elif not mask[i] and in_seg:
            segments.append((start, grid[i - 1]))
            in_seg = False
    if in_seg:
        segments.append((start, grid[-1]))
    return segments


def _align_labels(resampled_labels, grid):
    """
    Aligns cluster labels so that the leftmost cluster is always label 0.
    Operates in-place on the array rows.
    """
    for i in range(len(resampled_labels)):
        lbls = resampled_labels[i]
        unique = [u for u in np.unique(lbls) if u != -1]
        if len(unique) < 2:
            continue
        centroids = [np.mean(grid[lbls == u]) for u in unique]
        order = np.argsort(centroids)
        new_lbls = np.full_like(lbls, -1)
        for new_id, old_id in enumerate(order):
            new_lbls[lbls == unique[old_id]] = new_id
        resampled_labels[i] = new_lbls
    return resampled_labels


def _align_baseline(baseline_labels, grid):
    """Aligns baseline cluster labels so leftmost cluster is label 0."""
    unique_base = [u for u in np.unique(baseline_labels) if u != -1]
    if len(unique_base) >= 2:
        centroids_base = [np.mean(grid[baseline_labels == u]) for u in unique_base]
        order_base = np.argsort(centroids_base)
        new_base = np.full_like(baseline_labels, -1)
        for new_id, old_id in enumerate(order_base):
            new_base[baseline_labels == unique_base[old_id]] = new_id
        return new_base
    return baseline_labels


def main():
    # --- Setup & Data Generation ---
    np.random.seed(2026)
    outdir = "illustration_1d"
    os.makedirs(outdir, exist_ok=True)

    N = 100
    true_weights = np.array([0.5, 0.5])
    true_means = np.array([-1.0, 1])
    true_stds = np.array([0.4, 0.4])
    true_params = {'weights': true_weights, 'means': true_means, 'stds': true_stds}

    rng = np.random.default_rng(2026)
    X = gmm_sample(true_params, N, rng)

    # --- Fit Baseline GMM via EM (4 components, misspecified) ---
    print("Fitting baseline GMM via EM (K=4)...")
    baseline_params = fit_gmm(X, n_components=4)
    print(f"  weights: {baseline_params['weights']}")
    print(f"  means:   {baseline_params['means']}")
    print(f"  stds:    {baseline_params['stds']}")

    # --- Evaluation Grid ---
    grid_min, grid_max = -2.5, 2.5
    grid = np.linspace(grid_min, grid_max, 1000)

    true_density = gmm_prob(grid, true_params)
    baseline_density = gmm_prob(grid, baseline_params)
    baseline_log_density = gmm_log_prob(grid, baseline_params)

    # --- Threshold for Level-Set Clustering ---
    data_log_prob = gmm_log_prob(X, baseline_params)
    threshold = np.percentile(data_log_prob, 20)
    print(f"  log-density threshold (20th pctl): {threshold:.4f}")

    # --- Baseline Clustering (both variants) ---
    baseline_labels = level_set_cluster_1d(grid, baseline_log_density, threshold)
    baseline_labels_all = level_set_cluster_1d_all(grid, baseline_log_density, threshold)

    # --- Score-Based Particle Updates ---
    T = 50
    num_updates = 2000
    base_lr = 1

    print(f"Running {T} particle chains, {num_updates} updates each...")
    particle_params = run_particle_updates(
        baseline_params, T, num_updates, base_lr, N_data=N, seed=2026
    )

    # --- Evaluate Resampled Densities & Clusterings ---
    print("Evaluating resampled densities and clusterings...")
    resampled_densities = []
    resampled_labels = []
    resampled_labels_all = []

    for p in tqdm(particle_params, desc="Evaluating particles"):
        d = gmm_prob(grid, p)
        ld = gmm_log_prob(grid, p)
        lbls = level_set_cluster_1d(grid, ld, threshold)
        lbls_all = level_set_cluster_1d_all(grid, ld, threshold)
        resampled_densities.append(d)
        resampled_labels.append(lbls)
        resampled_labels_all.append(lbls_all)

    resampled_densities = np.array(resampled_densities)
    resampled_labels = np.array(resampled_labels)
    resampled_labels_all = np.array(resampled_labels_all)

    # --- Align Cluster Labels Across Particles ---
    resampled_labels = _align_labels(resampled_labels, grid)
    resampled_labels_all = _align_labels(resampled_labels_all, grid)
    baseline_labels = _align_baseline(baseline_labels, grid)
    baseline_labels_all = _align_baseline(baseline_labels_all, grid)

    # --- True clustering from true density ---
    true_log_density = gmm_log_prob(grid, true_params)
    true_labels = level_set_cluster_1d(grid, true_log_density, threshold)
    true_labels_all = level_set_cluster_1d_all(grid, true_log_density, threshold)
    true_labels = _align_baseline(true_labels, grid)
    true_labels_all = _align_baseline(true_labels_all, grid)

    # --- Plot style (match circles script: default sans-serif) ---
    print("Plotting...")
    plt.rcParams.update({
        'font.size': 11,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    tab10 = plt.cm.tab10
    max_clusters = 10  # tab10 has 10 colours
    density_ymin = -0.05  # lower bound for density y-axes

    def _get_cluster_colors(n):
        return [tab10(i % max_clusters) for i in range(n)]

    def _draw_clustering_band(ax, labels, grid, y_lo, y_hi):
        """Draw one clustering as colored horizontal bands between y_lo and y_hi."""
        unique = sorted(set(labels) - {-1})
        colors = _get_cluster_colors(len(unique))
        cmap_local = {uid: colors[i] for i, uid in enumerate(unique)}
        for uid, col in cmap_local.items():
            mask = labels == uid
            if not np.any(mask):
                continue
            segments = _contiguous_segments(mask, grid)
            for x0, x1 in segments:
                rect = Rectangle(
                    (x0, y_lo), x1 - x0, y_hi - y_lo,
                    facecolor=to_rgba(col, 0.5), edgecolor='none',
                )
                ax.add_patch(rect)

    def _draw_stacked_clusterings(ax, label_list, grid, row_labels=None):
        """
        Draw multiple clusterings stacked vertically in a single axis.
        Each clustering occupies one horizontal row.
        Row labels are placed on the right-hand y-axis.
        """
        n_rows = len(label_list)
        ax.set_ylim(0, n_rows)
        ax.set_xlim(grid[0], grid[-1])
        for row_idx, lbls in enumerate(label_list):
            # index 0 gets the highest y → appears at the top
            y_lo = n_rows - 1 - row_idx
            y_hi = y_lo + 1.0
            _draw_clustering_band(ax, lbls, grid, y_lo, y_hi)
            if row_idx > 0:
                ax.axhline(y_hi, color='white', lw=0.5)
        # Move labels to the right-hand side
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        if row_labels is not None:
            ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
            ax.set_yticklabels(row_labels, fontsize=10)
        else:
            ax.set_yticks([])

    def _set_density_ylim(ax):
        """Set density y-axis to start at density_ymin with first tick at 0."""
        ax.set_ylim(bottom=density_ymin)
        # Keep auto-generated ticks but ensure they start at 0
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=3))
        current_ticks = ax.get_yticks()
        ax.set_yticks([t for t in current_ticks if t >= 0])

    # --- Helper: cluster-whole-sample figure (cluster_all, 2 columns, no data on left) ---
    def _make_figure_wholecluster(true_lbls, baseline_lbls, resampled_lbls, outpath):
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 4.5),
            gridspec_kw={
                'height_ratios': [3, 1.5],
                'hspace': 0.18,
                'wspace': 0.4,
            },
        )

        # ==== Left column: true density + true clustering ====
        ax_dens_L = axes[0, 0]
        ax_clust_L = axes[1, 0]

        ax_dens_L.plot(grid, true_density, color='black', lw=1.4, label='True density')
        ax_dens_L.axhline(
            np.exp(threshold), color='0.45', lw=0.6, linestyle=':',
            label='Density level',
        )
        ax_dens_L.set_ylabel('Density')
        ax_dens_L.legend(frameon=False, loc='upper left')
        ax_dens_L.set_xlim(grid_min, grid_max)
        ax_dens_L.tick_params(labelbottom=False)

        _draw_stacked_clusterings(
            ax_clust_L, [true_lbls], grid,
            row_labels=['True clust.'],
        )
        ax_clust_L.set_xlabel('$x$')
        ax_clust_L.sharex(ax_dens_L)

        # ==== Right column: fitted + resampled densities ====
        ax_dens_R = axes[0, 1]
        ax_clust_R = axes[1, 1]

        for i in range(len(resampled_densities)):
            ax_dens_R.plot(grid, resampled_densities[i], color='0.75', alpha=0.2, lw=0.5)
        ax_dens_R.plot([], [], color='0.75', alpha=0.6, lw=1.0,
                       label='Resampled densities')
        ax_dens_R.plot(
            grid, baseline_density, color=tab10(0), lw=1.2,
            linestyle='--', label='Baseline fitted density',
        )
        ax_dens_R.axhline(
            np.exp(threshold), color='0.45', lw=0.6, linestyle=':',
            label='Density level',
        )
        ax_dens_R.plot(
            X, np.zeros_like(X), marker='+', ls='none',
            color='tab:red', ms=5, mew=0.8, label='Data',
        )
        ax_dens_R.legend(frameon=False, loc='upper left')
        ax_dens_R.set_xlim(grid_min, grid_max)
        ax_dens_R.tick_params(labelbottom=False)

        # Align y-axes (both start at density_ymin)
        y_top = max(ax_dens_L.get_ylim()[1], ax_dens_R.get_ylim()[1])
        ax_dens_L.set_ylim(density_ymin, y_top)
        ax_dens_R.set_ylim(density_ymin, y_top)
        # Ticks start at 0
        for ax in [ax_dens_L, ax_dens_R]:
            ticks = [t for t in ax.get_yticks() if t >= 0]
            ax.set_yticks(ticks)

        n_resample_show = min(4, len(resampled_lbls))
        stacked = [baseline_lbls] + [resampled_lbls[i] for i in range(n_resample_show)]
        row_labels = ['Baseline fitted clust.'] + [f'Resampled clust. {i+1}' for i in range(n_resample_show)]
        _draw_stacked_clusterings(ax_clust_R, stacked, grid, row_labels=row_labels)
        ax_clust_R.set_xlabel('$x$')
        ax_clust_R.sharex(ax_dens_R)

        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- Main figure (noise-unmarked): 3 columns with histogram ---
    def _make_figure_main(true_lbls, baseline_lbls, resampled_lbls, all_resampled_lbls, outpath):
        # Compute cluster counts across all resamples
        n_clusters_per_particle = [
            len(set(lbls) - {-1}) for lbls in all_resampled_lbls
        ]

        fig = plt.figure(figsize=(15, 4.5))
        # Outer grid: two regions — left pair of columns, right histogram
        gs_outer = fig.add_gridspec(
            1, 2,
            width_ratios=[2.2, 0.6],
            wspace=0.55,
        )
        # Left region: 2×2 grid for density + clustering panels
        gs_left = gs_outer[0].subgridspec(
            2, 2,
            height_ratios=[3, 1.5],
            hspace=0.18,
            wspace=0.4,
        )
        # Right region: single histogram spanning full height
        ax_hist = fig.add_subplot(gs_outer[1])

        # ==== Left column: true density + true clustering ====
        ax_dens_L = fig.add_subplot(gs_left[0, 0])
        ax_clust_L = fig.add_subplot(gs_left[1, 0], sharex=ax_dens_L)

        ax_dens_L.plot(grid, true_density, color='black', lw=1.4, label='True density')
        ax_dens_L.axhline(
            np.exp(threshold), color='0.45', lw=0.6, linestyle=':',
            label='Density level',
        )
        ax_dens_L.set_ylabel('Density')
        ax_dens_L.legend(frameon=False, loc='upper left')
        ax_dens_L.set_xlim(grid_min, grid_max)
        ax_dens_L.tick_params(labelbottom=False)

        _draw_stacked_clusterings(
            ax_clust_L, [true_lbls], grid,
            row_labels=['True clust.'],
        )
        ax_clust_L.set_xlabel('$x$')

        # ==== Middle column: fitted + resampled densities ====
        ax_dens_M = fig.add_subplot(gs_left[0, 1])
        ax_clust_M = fig.add_subplot(gs_left[1, 1], sharex=ax_dens_M)

        for i in range(len(resampled_densities)):
            ax_dens_M.plot(grid, resampled_densities[i], color='0.75', alpha=0.2, lw=0.5)
        ax_dens_M.plot([], [], color='0.75', alpha=0.6, lw=1.0,
                       label='Resampled densities')
        ax_dens_M.plot(
            grid, baseline_density, color=tab10(0), lw=1.2,
            linestyle='--', label='Baseline fitted density',
        )
        ax_dens_M.axhline(
            np.exp(threshold), color='0.45', lw=0.6, linestyle=':',
            label='Density level',
        )
        ax_dens_M.plot(
            X, np.zeros_like(X), marker='+', ls='none',
            color='tab:red', ms=5, mew=0.8, label='Data',
        )
        ax_dens_M.legend(frameon=False, loc='upper left')
        ax_dens_M.set_xlim(grid_min, grid_max)
        ax_dens_M.tick_params(labelbottom=False)

        # Align y-axes across density panels (both start at density_ymin)
        y_top = max(ax_dens_L.get_ylim()[1], ax_dens_M.get_ylim()[1])
        ax_dens_L.set_ylim(density_ymin, y_top)
        ax_dens_M.set_ylim(density_ymin, y_top)
        # Ticks start at 0
        for ax in [ax_dens_L, ax_dens_M]:
            ticks = [t for t in ax.get_yticks() if t >= 0]
            ax.set_yticks(ticks)

        n_resample_show = min(4, len(resampled_lbls))
        stacked = [baseline_lbls] + [resampled_lbls[i] for i in range(n_resample_show)]
        row_labels = ['Baseline fitted clust.'] + [f'Resampled clust. {i+1}' for i in range(n_resample_show)]
        _draw_stacked_clusterings(ax_clust_M, stacked, grid, row_labels=row_labels)
        ax_clust_M.set_xlabel('$x$')

        # ==== Right column: histogram of number of clusters ====

        counts = np.array(n_clusters_per_particle)
        unique_nc = sorted(set(counts))
        bins = np.arange(min(unique_nc) - 0.5, max(unique_nc) + 1.5, 1)
        ax_hist.hist(
            counts, bins=bins, weights=np.ones_like(counts) / len(counts),
            rwidth=0.8, color='0.55', edgecolor='white', lw=0.5,
        )
        ax_hist.set_xlabel('Number of clusters')
        ax_hist.set_ylabel('Mart. posterior frequency')
        ax_hist.set_xticks(unique_nc)

        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ==========================================================================
    # Plot 1: noise points unmarked (main figure, 3 columns)
    # ==========================================================================
    _make_figure_main(
        true_labels, baseline_labels, resampled_labels, resampled_labels,
        os.path.join(outdir, "martingale_clustering_1d.pdf"),
    )

    # ==========================================================================
    # Plot 2: every point assigned to closest cluster (not shown in paper)
    # ==========================================================================
    _make_figure_wholecluster(
        true_labels_all, baseline_labels_all, resampled_labels_all,
        os.path.join(outdir, "martingale_clustering_1d_all.pdf"),
    )

    # --- Summary Statistics ---
    n_clusters_per_particle = [
        len(set(lbls) - {-1}) for lbls in resampled_labels
    ]
    print(f"Cluster count distribution across {T} particles:")
    for nc in sorted(set(n_clusters_per_particle)):
        freq = n_clusters_per_particle.count(nc)
        print(f"  {nc} clusters: {freq}/{T} ({100*freq/T:.1f}%)")

    np.savez(
        os.path.join(outdir, "saved_results.npz"),
        grid=grid,
        true_density=true_density,
        baseline_density=baseline_density,
        resampled_densities=resampled_densities,
        resampled_labels=resampled_labels,
        resampled_labels_all=resampled_labels_all,
        baseline_labels=baseline_labels,
        baseline_labels_all=baseline_labels_all,
        X=X,
    )
    print("Done. Results saved to", outdir)


if __name__ == "__main__":
    main()
