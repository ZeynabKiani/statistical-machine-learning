import matplotlib.pyplot as plt
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.datasets import make_blobs

# Print documentation
print(__doc__)

def generate_sample_data(n_samples, random_state):
    """Generate synthetic data with two centers."""
    return make_blobs(n_samples=n_samples, centers=2, random_state=random_state)

def plot_clusters(ax, X, algo, centers, n_clusters, algorithm_name):
    """Plot clusters and centers on a given subplot."""
    ax.scatter(X[:, 0], X[:, 1], s=10, c=algo.labels_)
    ax.scatter(centers[:, 0], centers[:, 1], c="r", s=20)
    ax.set_title(f"{algorithm_name} : {n_clusters} clusters")

def main():
    # Parameters
    n_samples = 10000
    random_state = 0
    n_clusters_list = [4, 8, 16]

    # Generate sample data
    X, _ = generate_sample_data(n_samples, random_state)

    # Algorithms to compare
    clustering_algorithms = {
        "Bisecting K-Means": BisectingKMeans,
        "K-Means": KMeans,
    }

    # Make subplots for each variant
    fig, axs = plt.subplots(
        len(clustering_algorithms), len(n_clusters_list), figsize=(12, 5)
    )

    axs = axs.T

    # Compare clustering algorithms
    for i, (algorithm_name, Algorithm) in enumerate(clustering_algorithms.items()):
        for j, n_clusters in enumerate(n_clusters_list):
            # Initialize and fit the algorithm
            algo = Algorithm(n_clusters=n_clusters, random_state=random_state, n_init=3)
            algo.fit(X)
            centers = algo.cluster_centers_

            # Plot clusters on the current subplot
            plot_clusters(axs[j, i], X, algo, centers, n_clusters, algorithm_name)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
