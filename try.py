import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs # Can be used for simpler examples if needed
from scipy.spatial import Voronoi, voronoi_plot_2d

# Set random seed for reproducibility
np.random.seed(42)

# --- Helper Function for Plotting ---
def plot_kmeans_result(X, kmeans, centroids_init, title, subtitle_local, subtitle_global=None):
    """Plots the K-means clustering result and initial/final centroids."""
    
    # Get final results
    centroids_final = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Determine colors for clusters
    colors = plt.cm.viridis(np.linspace(0, 1, kmeans.n_clusters))
    
    fig, axes = plt.subplots(1, 2 if subtitle_global else 1, figsize=(12 if subtitle_global else 6, 5.5))
    fig.suptitle(title, fontsize=16)

    # --- Plot 1: Local Optimum Result ---
    ax1 = axes[0] if subtitle_global else axes # Get the correct axis

    # Plot data points colored by cluster label
    for k, col in enumerate(colors):
        cluster_data = X[labels == k]
        ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, c=[col], label=f'Cluster {k+1}')

    # Plot initial centroids
    ax1.scatter(centroids_init[:, 0], centroids_init[:, 1], s=250, marker='P', c='orange', 
                edgecolor='black', label='Initial Centroids')
    # Plot final centroids
    ax1.scatter(centroids_final[:, 0], centroids_final[:, 1], s=250, marker='X', c='red', 
                edgecolor='black', label='Final (Local Opt.) Centroids')

    ax1.set_title(subtitle_local)
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_aspect('equal', adjustable='box')
    
    # --- Plot 2: Global Optimum Hint (Optional) ---
    if subtitle_global:
        ax2 = axes[1]
         # Just plot data points - global optimum clustering is usually visually obvious
        ax2.scatter(X[:, 0], X[:, 1], s=50, c='grey', alpha=0.6) 
       
        # Optionally, you could run KMeans with multiple initializations to *likely* find the global optimum
        kmeans_global = KMeans(n_clusters=kmeans.n_clusters, n_init=10, random_state=42)
        kmeans_global.fit(X)
        centroids_global_opt = kmeans_global.cluster_centers_
        labels_global_opt = kmeans_global.labels_
        
        # Color points by likely global optimum
        for k, col in enumerate(colors):
            cluster_data = X[labels_global_opt == k]
            ax2.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, c=[col], label=f'Cluster {k+1}')
        
        ax2.scatter(centroids_global_opt[:, 0], centroids_global_opt[:, 1], s=250, marker='*', c='green', 
                    edgecolor='black', label='Likely Global Opt. Centroids')

        ax2.set_title(subtitle_global)
        ax2.set_xlabel("Feature 1")
        ax2.set_ylabel("Feature 2")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

# --- Method 1: Elongated Clusters (K=2) ---

print("Generating Visualization for Method 1: Elongated Clusters (K=2)")

# 1. Data Generation
n_samples = 100
# Cluster A: Elongated vertically, centered near (0, 2.5)
cov_a = [[0.1, 0], [0, 4]] 
X_a = np.random.multivariate_normal([0, 2.5], cov_a, n_samples // 2)
# Cluster B: Elongated vertically, centered near (0, -2.5)
cov_b = [[0.1, 0], [0, 4]] 
X_b = np.random.multivariate_normal([0, -2.5], cov_b, n_samples // 2)

X_method1 = np.vstack((X_a, X_b))

# 2. Bad Initialization (encourages horizontal split)
# Place centroids left and right of center, near y=0
centroids_init_method1 = np.array([
    [-1.5, 0], 
    [1.5, 0]   
])

# 3. Run K-means with bad initialization (n_init=1 forces use of provided init)
kmeans_method1 = KMeans(n_clusters=2, init=centroids_init_method1, n_init=1, max_iter=300, random_state=42)
kmeans_method1.fit(X_method1)

# 4. Visualize
plot_kmeans_result(X_method1, 
                   kmeans_method1, 
                   centroids_init_method1, 
                   title="K-Means Local Optimum: Elongated Clusters (K=2)",
                   subtitle_local="Result Converged to Local Optimum",
                   subtitle_global="Hint: Globally Optimal Clustering")


# --- Method 2: Uneven Density / Bridge (K=3) ---

print("\nGenerating Visualization for Method 2: Uneven Density/Bridge (K=3)")

# 1. Data Generation
n_samples_dense = 60
n_samples_sparse = 20
n_samples_bridge = 10

# Cluster A: Dense, ~circular, top-left
X_a = make_blobs(n_samples=n_samples_dense, centers=[(-3, 3)], cluster_std=0.6, random_state=10)[0]
# Cluster B: Dense, ~circular, bottom-left
X_b = make_blobs(n_samples=n_samples_dense, centers=[(-3, -3)], cluster_std=0.6, random_state=20)[0]
# Cluster C: Sparse, ~circular, right
X_c = make_blobs(n_samples=n_samples_sparse, centers=[(3, 0)], cluster_std=0.5, random_state=30)[0]
# Bridge points
X_bridge = np.random.rand(n_samples_bridge, 2)
X_bridge[:, 0] = X_bridge[:, 0] * 1.0 - 0.5 # x between -0.5 and 0.5
X_bridge[:, 1] = X_bridge[:, 1] * 4.0 - 2.0 # y between -2 and 2

X_method2 = np.vstack((X_a, X_b, X_c, X_bridge))

# 2. Bad Initialization (Place centroids to likely split dense clusters badly)
# One centroid near C, two centroids placed vertically in the middle, 
# likely splitting A/B horizontally and grabbing bridge points incorrectly.
centroids_init_method2 = np.array([
    [-1, 1.5],  # Near top part of A/B/Bridge
    [-1, -1.5], # Near bottom part of A/B/Bridge
    [3, 0]    # Near C
])

# 3. Run K-means
kmeans_method2 = KMeans(n_clusters=3, init=centroids_init_method2, n_init=1, max_iter=300, random_state=42)
kmeans_method2.fit(X_method2)

# 4. Visualize
plot_kmeans_result(X_method2, 
                   kmeans_method2, 
                   centroids_init_method2, 
                   title="K-Means Local Optimum: Uneven Density/Bridge (K=3)",
                   subtitle_local="Result Converged to Local Optimum",
                   subtitle_global="Hint: Globally Optimal Clustering")

print("\nExplanation:")
print("The plots illustrate how K-Means can converge to a stable but suboptimal clustering.")
print("In Method 1, the bad initial centroids (orange P) cause K-Means to split the two naturally vertical clusters horizontally (red X), resulting in higher within-cluster distances than the optimal vertical split.")
print("In Method 2, the initial centroids cause the two dense clusters on the left to be split poorly, grouping parts of both with the bridge points, while the globally optimal solution would likely keep the three distinct groups (two dense, one sparse) separate.")
print("The 'Final (Local Opt.) Centroids' (red X) represent a state where K-Means makes no further progress, even though a better clustering exists (hinted at by the 'Likely Global Opt. Centroids' (green *) in the right plots).")