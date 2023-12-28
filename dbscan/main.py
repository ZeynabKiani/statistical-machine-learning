import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import DBSCAN

class DBSCANDemo:
    def __init__(self):
        self.X, self.labels_true = self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        # Generate synthetic data
        centers = [[1, 1], [-1, -1], [1, -1]]
        
        X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
        X = StandardScaler().fit_transform(X)
        return X, labels_true

    def visualize_data(self):
        # Visualize the data
        plt.scatter(self.X[:, 0], self.X[:, 1])
        plt.show()

    def apply_dbscan(self, eps=0.3, min_samples=10):
        # Apply DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.X)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print(f"Estimated number of clusters: {n_clusters_}")
        print(f"Estimated number of noise points: {n_noise_}")

        return labels, n_clusters_, db

    def evaluate_clusters(self, labels):
        # Evaluate clustering metrics
        print(f"Homogeneity: {metrics.homogeneity_score(self.labels_true, labels):.3f}")
        print(f"Completeness: {metrics.completeness_score(self.labels_true, labels):.3f}")
        print(f"V-measure: {metrics.v_measure_score(self.labels_true, labels):.3f}")
        print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(self.labels_true, labels):.3f}")
        print(f"Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(self.labels_true, labels):.3f}")
        print(f"Silhouette Coefficient: {metrics.silhouette_score(self.X, labels):.3f}")

    def plot_results(self, labels, db):
        # Plot clustering results
        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = self.X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=14)

            xy = self.X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6)

        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.show()

def run_dbscan_demo():
    dbscan_demo = DBSCANDemo()
    dbscan_demo.visualize_data()

    labels, n_clusters_, db = dbscan_demo.apply_dbscan(eps=0.3, min_samples=10)

    dbscan_demo.evaluate_clusters(labels)
    dbscan_demo.plot_results(labels, db)

if __name__ == "__main__":
    run_dbscan_demo()
