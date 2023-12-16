import logging
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Sampler

from geogenie.utils.scorers import haversine
from geogenie.utils.utils import geo_coords_is_valid


def SD_KMSMOTE(X_minority, k=5, K_clusters=5, total_synthetic_samples=1000):
    """
    Implement SD-KMSMOTE algorithm for oversampling minority class samples.

    Args:
        X_minority (np.array): Array containing minority class samples.
        k (int): Number of nearest neighbors for filtering and synthesis.
        K_clusters (int): Number of clusters for K-means clustering.
        total_synthetic_samples (int): Total number of synthetic samples to generate.

    Returns:
        np.array: Array containing the original and synthesized minority class samples.
    """
    # Filter isolated points using k-nearest neighbors
    nn = NearestNeighbors(n_neighbors=k).fit(X_minority)
    distances, indices = nn.kneighbors(X_minority)
    filtered_indices = [
        i
        for i, d in enumerate(distances)
        if not all(np.isin(d, X_minority[indices[i]]))
    ]
    X_filtered = X_minority[filtered_indices]

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=K_clusters, init="k-means++").fit(X_filtered)
    clusters = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Calculate cluster weights
    majority_center = np.mean(
        X_minority, axis=0
    )  # Assuming majority class center is the mean
    distances_to_majority = np.linalg.norm(cluster_centers - majority_center, axis=1)
    weights = 1 - distances_to_majority / np.sum(distances_to_majority)

    # Allocate synthetic samples to clusters based on weights
    synthetic_samples_per_cluster = np.round(weights * total_synthetic_samples).astype(
        int
    )

    # Synthesize new samples
    new_samples = []
    for i in range(K_clusters):
        cluster_samples = X_filtered[clusters == i]
        nn = NearestNeighbors(n_neighbors=k).fit(cluster_samples)
        num_samples_to_generate = synthetic_samples_per_cluster[i]
        for _ in range(num_samples_to_generate):
            sample_idx = np.random.choice(len(cluster_samples))
            sample = cluster_samples[sample_idx]
            _, neighbors_idx = nn.kneighbors(
                [sample], n_neighbors=min(k, len(cluster_samples))
            )
            for idx in neighbors_idx[0]:
                diff = cluster_samples[idx] - sample
                new_sample = sample + np.random.rand() * diff
                new_samples.append(new_sample)

    return np.vstack([X_minority, np.vstack(new_samples)])


def cluster_minority_samples(
    minority_samples: np.ndarray, n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster minority class samples using k-means and calculate the Euclidean distances
    from each cluster center to the majority class center.

    Args:
    minority_samples (np.ndarray): Array of minority class samples.
    n_clusters (int): Number of clusters to form.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the cluster centers and the Euclidean distances of each cluster center to the majority class center.
    """
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(minority_samples)
    cluster_centers = kmeans.cluster_centers_

    # Calculate the majority class center (mean of minority samples)
    majority_class_center = np.mean(minority_samples, axis=0)

    # Calculate Euclidean distances from each cluster center to the majority class center
    distances = np.linalg.norm(cluster_centers - majority_class_center, axis=1)

    return cluster_centers, distances


def assign_cluster_weights(distances: np.ndarray) -> np.ndarray:
    """
    Assign weight values to each cluster based on the Euclidean distance to the majority class center.

    Args:
    distances (np.ndarray): Euclidean distances of each cluster center to the majority class center.

    Returns:
    np.ndarray: Weight values assigned to each cluster.
    """
    # Inverse distances to assign higher weights to clusters closer to the majority class center
    inverse_distances = 1 / distances
    # Normalize the weights to sum up to 1
    weights = inverse_distances / np.sum(inverse_distances)

    return weights


def filter_isolated_samples(
    minority_samples: np.ndarray, majority_samples: np.ndarray, n_neighbors: int
) -> np.ndarray:
    """
    Filters out isolated minority samples that are surrounded by majority class samples.

    Args:
    minority_samples (np.ndarray): Array of minority class samples.
    majority_samples (np.ndarray): Array of majority class samples.
    n_neighbors (int): Number of neighbors to consider for determining if a sample is isolated.

    Returns:
    np.ndarray: Filtered array of minority samples without isolated points.
    """
    # Combine minority and majority samples
    all_samples = np.vstack((minority_samples, majority_samples))
    # Create labels for minority (1) and majority (0) samples
    labels = np.hstack(
        (np.ones(minority_samples.shape[0]), np.zeros(majority_samples.shape[0]))
    )

    # Use NearestNeighbors to find the nearest neighbors of each sample
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(all_samples)
    distances, indices = nn.kneighbors(minority_samples)

    # Check if the neighbors are majority samples
    is_isolated = lambda idx: all(labels[indices[idx][1:]] == 0)
    isolated_indices = [
        idx for idx in range(minority_samples.shape[0]) if is_isolated(idx)
    ]

    # Filter out isolated minority samples
    return np.delete(minority_samples, isolated_indices, axis=0)

    # Example usage:
    # minority_samples = np.array([[1, 2], [2, 3], [3, 4]])  # Replace with actual minority class data
    # majority_samples = np.array([[5, 6], [6, 7], [7, 8]])  # Replace with actual majority class data
    # filtered_minority_samples = filter_isolated_samples(minority_samples, majority_samples, n_neighbors=5)

    # The function will filter out minority samples that are surrounded by majority samples.


class GeographicDensitySampler(Sampler):
    def __init__(
        self,
        data,
        focus_regions=None,
        use_kmeans=True,
        use_kde=True,
        w_power=1,
        max_clusters=10,
        max_neighbors=10,
        indices=None,
        objective_mode=False,
        normalize=False,
    ):
        """
        Args:
            data (pandas DataFrame): DataFrame containing 'longitude' and 'latitude'.
            focus_regions (list of tuples): Regions of interest (longitude and latitude ranges).
            use_kmeans (bool): Whether to use KMeans clustering.
            use_kde (bool): Whether to use KernelDensity Estimation with adaptive bandwidth.
            w_power (float): Power to which the inverse density is raised.
            max_clusters (int): Maximum number of clusters to try for KMeans.
            max_neighbors (int): Maximum number of neighbors to try for adaptive bandwidth.
            indices (numpy.ndarray): Indices to use. If None, the uses all indices.
            objective_mode (bool): Whether to use objective mode with Optuna search. If True, use_kde and use_kmeans can both be False, but sample_weights will be all 1.0.
            normalize (bool): Whether to normalize the sample weights. Defaults to False.
        """
        self.logger = logging.getLogger(__name__)

        if not any([objective_mode, use_kmeans, use_kde, focus_regions]):
            msg = "Either KMeans, KernelDensity, or Focus Regions must be used with 'GeographicDensitySampler' if objective_mode is False."
            self.logger.error(msg)
            raise AssertionError(msg)

        geo_coords_is_valid(data.to_numpy())

        self.data = data
        self.focus_regions = focus_regions
        self.use_kmeans = use_kmeans
        self.use_kde = use_kde
        self.w_power = w_power
        self.max_clusters = max_clusters
        self.max_neighbors = max_neighbors
        self.objective_mode = objective_mode
        self.normalize = normalize

        if indices is None:
            self.indices = np.arange(data.shape[0])
        else:
            self.indices = indices

        self.optimal_k_neighbors = self.find_optimal_k_neighbors()
        self.weights = self.calculate_weights()

    def perform_gwr(self, predictions, targets, sample_weights=None):
        """
        Perform Geographically Weighted Regression for post-training assessment.

        Args:
            predictions (np.ndarray): Model predictions.
            targets (np.ndarray): Actual target values.
            sample_weights (np.ndarray): Sample weights. Defaults to None.

        Returns:
            float: Aggregated GWR-based performance metric.
        """
        # Assume predictions and targets are 2D arrays with shape (n_samples, 2),
        # where each row contains [longitude, latitude].

        # Placeholder for local metrics
        local_metrics = []

        for i, point in enumerate(targets):
            # Define the local region around the current point
            # This could be done using a distance threshold or nearest neighbors
            local_region = self.define_local_region(targets, point)

            # Extract the subset of predictions and targets that fall within the local region
            local_predictions = predictions[local_region]
            local_targets = targets[local_region]

            # Calculate Haversine error for each pair of points
            local_error = np.array(
                [
                    haversine(act[0], act[1], pred[0], pred[1])
                    for act, pred in zip(local_targets, local_predictions)
                ]
            )

            # Optionally, adjust the local error using sample weights
            local_error_weighted = local_error * sample_weights[local_region]

            # Calculate a local metric, e.g., mean local error
            local_metric = np.mean(local_error_weighted)
            local_metrics.append(local_metric)

        # Aggregate the local metrics into a global performance measure
        global_performance = np.mean(local_metrics)
        return global_performance

    def define_local_region(self, y_true, point, threshold=5.0):
        """
        Define a local region around a given point for GWR.

        Args:
            y_true (np.ndarray): Truth values.
            point (np.ndarray): The central point of the local region.
            threshold (float): The distance threshold to include points in the local region.

        Returns:
            np.ndarray: A boolean array indicating which points are in the local region.
        """
        # Calculate the distance from the current point to all other points
        distances = np.sqrt(
            (y_true[:, 0] - point[0]) ** 2 + (y_true[:, 1] - point[1]) ** 2
        )

        # Define the local region as points within the distance threshold
        local_region = distances < threshold
        return local_region

    def calculate_weights(self):
        if not self.objective_mode:
            self.logger.info("Estimating sample weights...")

        weights = np.ones(len(self.data))

        if self.use_kmeans:
            if not self.objective_mode:
                self.logger.info("Estimating sample weights with KMeans...")
            n_clusters = self.find_optimal_clusters()
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
            labels = kmeans.fit_predict(self.data)
            cluster_counts = np.bincount(labels, minlength=n_clusters)
            cluster_weights = 1 / (cluster_counts[labels] + 1e-5)
            weights *= cluster_weights

        if self.use_kde:
            if not self.objective_mode:
                self.logger.info("Estimating sample weights with Kernel Density...")
            adaptive_bandwidth = self.calculate_adaptive_bandwidth(
                self.optimal_k_neighbors
            )
            kde = KernelDensity(bandwidth=adaptive_bandwidth, kernel="gaussian")
            kde.fit(self.data)
            log_density = kde.score_samples(self.data)
            density = np.exp(log_density)

            # control aggressiveness of inverse weighting.
            weights *= 1 / np.power(density + 1e-5, self.w_power)

        if self.focus_regions:
            for region in self.focus_regions:
                lon_min, lon_max, lat_min, lat_max = region
                in_region = (
                    (self.data["longitude"] >= lon_min)
                    & (self.data["longitude"] <= lon_max)
                    & (self.data["latitude"] >= lat_min)
                    & (self.data["latitude"] <= lat_max)
                )
                weights[in_region] *= 2

        if not self.objective_mode:
            self.logger.info("Done estimating sample weights.")

        if self.normalize:
            mms = MinMaxScaler()
            weights = np.squeeze(mms.fit_transform(weights.reshape(-1, 1)))
        return weights

    def calculate_adaptive_bandwidth(self, k_neighbors):
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, n_jobs=-1).fit(self.data)
        distances, _ = nbrs.kneighbors(self.data)

        # Exclude the first distance (self-distance)
        average_distance = np.mean(distances[:, 1:], axis=1)
        adaptive_bandwidth = np.mean(average_distance)  # Mean over all points
        return adaptive_bandwidth

    def find_optimal_k_neighbors(self):
        """Uses use the stability of the average distance as the criterion for determining optimal nearest neighbors."""
        optimal_k = 2
        min_variation = float("inf")
        for k in range(2, self.max_neighbors + 1):
            nbrs = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(self.data)
            distances, _ = nbrs.kneighbors(self.data)
            average_distance = np.mean(distances[:, 1:], axis=1)
            variation = np.var(average_distance)
            if variation < min_variation:
                min_variation = variation
                optimal_k = k
        return optimal_k

    def find_optimal_clusters(self):
        best_score = -1
        best_n_clusters = 2
        for n_clusters in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
            labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
        return best_n_clusters

    def __iter__(self):
        return (
            self.indices[i]
            for i in np.random.choice(
                self.indices,
                size=len(self.indices),
                p=self.weights / np.sum(self.weights),
            )
        )

    def __len__(self):
        return len(self.data)
