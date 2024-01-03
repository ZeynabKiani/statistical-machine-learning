import matplotlib.pyplot as plt

from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.datasets import make_blobs

def generate_sample_data(n_samples, random_state):
    """Generate synthetic data with two centers."""
    return make_blobs(n_samples=n_samples, centers=2, random_state=random_state)

def plot_clusters(ax, X, algo, centers, n_clusters, algorithm_name):
    """Plot clusters and centers on a given subplot."""
    
    ax.scatter(X[:, 0], X[:, 1], s=10, c=algo.labels_)
    ax.scatter(centers[:, 0], centers[:, 1], c="r", s=20)
    ax.set_title(f"{algorithm_name} : {n_clusters} clusters")

def compare_clustering_algorithms(X, random_state, n_clusters_list, clustering_algorithms):
    """Compare different clustering algorithms and plot the results."""
    fig, axs = plt.subplots(len(clustering_algorithms), len(n_clusters_list), figsize=(12, 5))
    axs = axs.T

    for i, (algorithm_name, Algorithm) in enumerate(clustering_algorithms.items()):
        for j, n_clusters in enumerate(n_clusters_list):
            algo = Algorithm(n_clusters=n_clusters, random_state=random_state, n_init=3)
            algo.fit(X)
            centers = algo.cluster_centers_
            plot_clusters(axs[j, i], X, algo, centers, n_clusters, algorithm_name)

    for ax in axs.flat:
        ax.label_outer()
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def main():
    n_samples = 10000
    random_state = 0
    n_clusters_list = [4, 8, 16]

    X, _ = generate_sample_data(n_samples, random_state)

    clustering_algorithms = {
        "Bisecting K-Means": BisectingKMeans,
        "K-Means": KMeans,
    }

    compare_clustering_algorithms(X, random_state, n_clusters_list, clustering_algorithms)

if __name__ == "__main__":
    main()
