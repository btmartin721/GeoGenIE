import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity, NearestNeighbors
from torch.utils.data import Sampler

from geogenie.utils.utils import geo_coords_is_valid


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
        """
        self.logger = logging.getLogger(__name__)

        if not use_kmeans and not use_kde:
            msg = "Either KMeans or KernelDensity must be used with 'GeographicDensitySampler'."
            self.logger.error(msg)
            raise AssertionError(msg)

        geo_coords_is_valid(data)

        self.data = data
        self.focus_regions = focus_regions
        self.use_kmeans = use_kmeans
        self.use_kde = use_kde
        self.w_power = w_power
        self.max_clusters = max_clusters
        self.max_neighbors = max_neighbors
        self.indices = np.arange(len(data))

        self.optimal_k_neighbors = self.find_optimal_k_neighbors()
        self.weights = self.calculate_weights()

    def calculate_weights(self):
        self.logger.info("Estimating sample weights...")

        weights = np.ones(len(self.data))

        if self.use_kmeans:
            self.logger.info("Estimating sample weights with KMeans...")
            n_clusters = self.find_optimal_clusters()
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
            labels = kmeans.fit_predict(self.data)
            cluster_counts = np.bincount(labels, minlength=n_clusters)
            cluster_weights = 1 / (cluster_counts[labels] + 1e-5)
            weights *= cluster_weights

        if self.use_kde:
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
        self.logger.info("Done estimating sample weights.")
        return weights

    def calculate_adaptive_bandwidth(self, k_neighbors):
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(self.data)
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
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(self.data)
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
