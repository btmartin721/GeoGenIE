import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gamma
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from geogenie.plotting.plotting import PlotGenIE


class GeoGeneticOutlierDetector:
    """
    A class to detect outliers based on genomic SNPs and geographic coordinates.

    Attributes:
        genetic_data (np.array): 2D array of shape (n_samples, n_loci) for SNP data.
        geographic_data (np.array): 2D array of shape (n_samples, 2) for geographic coordinates.
    """

    def __init__(
        self,
        genetic_data,
        geographic_data,
        output_dir,
        prefix,
        url="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip",
        buffer=0.1,
        show_plots=False,
    ):
        """
        Initializes the GeoGeneticOutlierDetector with genetic and geographic data.

        Args:
            genetic_data (np.array): The SNP data as a 2D numpy array.
            geographic_data (np.array): The geographic coordinates as a 2D numpy array.
        """
        self.genetic_data = genetic_data
        self.geographic_data = geographic_data
        self.geographic_data_original = geographic_data.copy()
        self.genetic_data_original = genetic_data.copy()
        self.original_indices = np.arange(len(self.geographic_data))
        self.plotting = PlotGenIE("cpu", output_dir, prefix)
        self.url = url
        self.buffer = buffer
        self.output_dir = output_dir
        self.prefix = prefix
        self.show_plots = show_plots

        self.logger = logging.getLogger(__name__)

    def haversine_distance(self, coords1, coords2):
        """
        Calculate the Haversine distance between two points on the earth.

        Args:
            coords1 (tuple): Longitude and latitude for the first point.
            coords2 (tuple): Longitude and latitude for the second point.

        Returns:
            float: Haversine distance in kilometers.
        """
        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert latitude and longitude from degrees to radians
        lon1, lat1 = np.radians(coords1)
        lon2, lat2 = np.radians(coords2)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def get_dist_matrices(self, independent_data, independent_type):
        if independent_type == "geographic":
            distances = squareform(pdist(self.geographic_data, metric="euclidean"))
        elif independent_type == "genetic":
            distances = squareform(pdist(self.genetic_data, metric="euclidean"))
        else:
            raise ValueError(
                f"'independent_type' must be either 'genetic' or 'geographic', but got {independent_type}"
            )

        return distances

    def knn_regression(self, target_coordinates, distance_matrix, k=5, min_nn_dist=100):
        """
        Performs KNN regression to predict target coordinates based on a distance matrix.
        """
        neighbors = NearestNeighbors(n_neighbors=k, metric="precomputed")
        neighbors.fit(distance_matrix)
        distances, indices = neighbors.kneighbors(distance_matrix)

        # Adjust weights to ignore neighbors within the min_nn_dist
        adjusted_distances = np.where(distances < min_nn_dist, np.inf, distances)
        weights = np.where(adjusted_distances == np.inf, 0, 1 / adjusted_distances)

        # Normalize weights, handle cases where sum of weights is zero
        weight_sums = weights.sum(axis=1)
        valid_weight_sums = weight_sums != 0
        weights[valid_weight_sums] /= weight_sums[valid_weight_sums, np.newaxis]

        # For rows where weight sum is zero, assign equal weights
        weights[~valid_weight_sums] = 1 / k

        predictions = np.array(
            [
                np.average(target_coordinates[indices[i]], axis=0, weights=weights[i])
                for i in range(len(target_coordinates))
            ]
        )
        return predictions

    def optimize_k(self, target_coordinates, distance_matrix, k_range):
        best_k = k_range.start
        min_error = float("inf")

        for k in k_range:
            if k > len(target_coordinates):
                break
            predictions = self.knn_regression(target_coordinates, distance_matrix, k)
            error = np.mean((target_coordinates - predictions) ** 2)
            if error < min_error:
                min_error = error
                best_k = k

        return best_k

    def detect_outliers(
        self,
        target_coordinates,
        distance_matrix,
        max_k=20,
        method="gamma",
        significance_level=0.95,
        eps=1e-8,
        min_nn_dist=0.1,
    ):
        """
        Detects outliers using prediction errors and statistical testing.
        """
        opt_k = self.optimize_k(target_coordinates, distance_matrix, range(2, max_k))
        predictions = self.knn_regression(
            target_coordinates, distance_matrix, opt_k, min_nn_dist=min_nn_dist
        )
        errors = np.mean((target_coordinates - predictions) ** 2, axis=1)

        # Ensure that errors do not contain non-finite values
        errors = np.where(np.isfinite(errors), errors, 0) + eps

        if method == "gamma":
            # Fit a Gamma distribution to the errors, ensuring all values are finite
            finite_errors = errors[np.isfinite(errors)]
            if len(finite_errors) > 0:
                shape, loc, scale = gamma.fit(finite_errors)
                p_values = gamma.cdf(finite_errors, shape, loc=loc, scale=scale)
                outliers = np.where(p_values > significance_level)
            else:
                outliers = ([],)  # No outliers if all errors are non-finite
        else:
            raise NotImplementedError(
                f"'method' argument supports only 'gamma', but got: {method}"
            )

        return outliers[0]

    def compute_geographic_prediction_errors(
        self, target_coordinates, predicted_coordinates
    ):
        """
        Computes the geographic prediction errors.
        """
        errors = np.array(
            [
                np.linalg.norm(target_coordinates[i] - predicted_coordinates[i])
                for i in range(len(target_coordinates))
            ]
        )
        return errors

    def multi_stage_outlier_knn(
        self,
        dependent_data,
        independent_data,
        independent_type,
        significance_level=0.95,
        max_k=50,
        min_nn_dist=0.1,
    ):
        """
        Iterative Outlier Detection using KNN.

        Start a loop that will continue until no new outliers are detected.
        Calculate the distance matrix using get_dist_matrices, which computes pairwise distances between samples based on the specified independent data type ('genetic' or 'geographic').
        Apply a mask to ignore previously identified outliers. This is done by selecting rows and columns from the distance matrix where outlier_flags is False, effectively filtering the data.
        Detect current outliers using the detect_outliers method. This method uses the filtered distance matrix and performs KNN regression on the dependent data (masked to exclude previously detected outliers). It then uses a statistical test (Gamma distribution) to determine if any of the samples are outliers based on the significance level provided.
        """
        outlier_flags = np.zeros(len(dependent_data), dtype=bool)
        original_indices = np.arange(len(dependent_data))

        if independent_type not in ["genetic", "geographic"]:
            raise ValueError(
                f"'independent_type' must be either 'genetic' or 'geographic', "
                f"but got: {independent_type}"
            )

        while True:
            distances = self.get_dist_matrices(
                independent_data,
                independent_type,
            )
            mask = ~outlier_flags
            filtered_distances = distances[mask][:, mask]
            filtered_indices = original_indices[mask]

            current_outliers = self.detect_outliers(
                dependent_data[mask],
                filtered_distances,  # Use the filtered distance matrix
                method="gamma",
                significance_level=significance_level,
                max_k=max_k,
                min_nn_dist=min_nn_dist,
            )

            if len(current_outliers) == 0:
                break

            # Map current_outliers back to original indices
            original_outlier_indices = filtered_indices[current_outliers]
            outlier_flags[original_outlier_indices] = True

        return np.where(outlier_flags)[0]

    def find_optimal_clusters(self, max_clusters=10):
        """
        Determines the optimal number of clusters using the Elbow Method.

        Args:
            max_clusters (int): The maximum number of clusters to test.

        Returns:
            int: Optimal number of clusters.
        """
        inertias = []
        range_clusters = range(1, max_clusters + 1)

        for k in range_clusters:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(
                self.geographic_data
            )
            inertias.append(kmeans.inertia_)

        kneedle = KneeLocator(
            range_clusters, inertias, curve="convex", direction="decreasing"
        )

        kneedle.plot_knee(
            figsize=(10, 10),
            title="Optimal K for Cluster Centers",
            xlabel="Number of Clusters",
            ylabel="Inertia",
        )

        outfile = os.path.join(
            self.output_dir,
            "plots",
            f"{self.prefix}_optimal_kmeans_clusters.png",
        )

        if self.show_plots:
            plt.show()
        plt.savefig(outfile, facecolor="white")
        plt.close()

        return kneedle.knee

    def calculate_cluster_centroids(self, data, max_clusters):
        """
        Calculates the centroids of geographic clusters.

        Args:
            max_clusters (int): The maximum number of clusters to form. Will be optimized with KMeans clustering.

        Returns:
            dict: A dictionary with cluster ids as keys and centroid coordinates as values.
        """
        optk = self.find_optimal_clusters(max_clusters=max_clusters)

        kmeans = KMeans(n_clusters=optk, random_state=0, n_init="auto").fit(data)
        centroids = kmeans.cluster_centers_
        cluster_assignments = kmeans.labels_  # Store cluster assignments
        return {
            f"Cluster {i}": tuple(centroid) for i, centroid in enumerate(centroids)
        }, cluster_assignments

    def find_correct_cluster_for_sample(
        self, sample_idx, data_type, genetic_centroids, geographic_centroids
    ):
        """
        Find the correct cluster for a sample based on the specified data type (genetic or geographic).

        Args:
            sample_idx (int): Index of the sample.
            data_type (str): Type of data to use for determining the correct cluster ('genetic' or 'geographic').

        Returns:
            int: Index of the correct cluster for the sample.
        """
        if data_type == "genetic":
            # Use genetic data to find the correct cluster
            sample_data = self.genetic_data[sample_idx]
            centroids = genetic_centroids
        elif data_type == "geographic":
            # Use geographic data to find the correct cluster
            sample_data = self.geographic_data[sample_idx]
            centroids = geographic_centroids
        else:
            raise ValueError("data_type must be 'genetic' or 'geographic'")

        min_distance = float("inf")
        correct_cluster_id = -1

        for centroid_id, centroid in enumerate(centroids):
            distance = np.linalg.norm(sample_data - centroid)
            if distance < min_distance:
                min_distance = distance
                correct_cluster_id = centroid_id

        return correct_cluster_id

    def find_misclustered_samples(
        self,
        geographic_data,
        genetic_data,
        geographic_centroids,
        genetic_centroids,
        current_cluster_assignments,
    ):
        """
        Identify samples that are incorrectly clustered, considering both geographic and genetic distances.

        Args:
            geographic_data (numpy.ndarray): Array of geographic coordinates.
            genetic_data (numpy.ndarray): Array of genetic data.
            geographic_centroids (numpy.ndarray): Array of geographic centroids.
            genetic_centroids (numpy.ndarray): Array of genetic centroids.
            current_cluster_assignments (numpy.ndarray): Current cluster assignments for each sample.

        Returns:
            dict: Dictionary with keys 'genetic' and 'geographic', each containing an array of indices representing samples that are incorrectly clustered.
        """
        misclustered_samples = {"genetic": [], "geographic": []}

        # Check for genetic misclustering
        for idx, gen_sample in enumerate(genetic_data):
            closest_genetic_centroid_id = self.find_closest_centroid_id(
                gen_sample, genetic_centroids
            )
            if closest_genetic_centroid_id != current_cluster_assignments[idx]:
                misclustered_samples["genetic"].append(idx)

        # Check for geographic misclustering
        for idx, geo_sample in enumerate(geographic_data):
            closest_geographic_centroid_id = self.find_closest_centroid_id(
                geo_sample, geographic_centroids.values()
            )
            if closest_geographic_centroid_id != current_cluster_assignments[idx]:
                misclustered_samples["geographic"].append(idx)

        return misclustered_samples

    def find_closest_centroid_id(self, sample, centroids):
        """
        Find the closest centroid to a given sample.

        Args:
            sample (numpy.ndarray): The sample data (genetic or geographic).
            centroids (list): List of centroids (genetic or geographic).

        Returns:
            int: The ID of the closest centroid.
        """
        min_distance = float("inf")
        closest_centroid_id = -1
        if isinstance(centroids, dict):
            centroids = centroids.values()
        for centroid_id, centroid in enumerate(centroids):
            distance = np.linalg.norm(sample - centroid)
            if distance < min_distance:
                min_distance = distance
                closest_centroid_id = centroid_id
        return closest_centroid_id

    def calculate_genetic_centroids(self, genetic_data, cluster_assignments):
        """
        Calculates the centroids of genetic clusters.

        Args:
            genetic_data (numpy.ndarray): Array of genetic data where rows are samples and columns are features.
            cluster_assignments (numpy.ndarray): Array of cluster assignments for each sample.

        Returns:
            numpy.ndarray: An array where each row is the centroid of a cluster.
        """
        unique_clusters = np.unique(cluster_assignments)
        centroids = np.zeros((len(unique_clusters), genetic_data.shape[1]))

        for i, cluster in enumerate(unique_clusters):
            members = genetic_data[cluster_assignments == cluster]
            centroids[i] = np.mean(members, axis=0)

        return centroids

    def composite_outlier_detection(
        self,
        significance_level=0.95,
        max_k=50,
        min_nn_dist=100,
        max_clusters=10,
    ):
        # Geographic centroids and assignments.
        geographic_centroids, geographic_assignments = self.calculate_cluster_centroids(
            self.geographic_data, max_clusters=max_clusters
        )

        # genetic_centroids, genetic_assignments = self.calculate_cluster_centroids(
        #     self.genetic_data, max_clusters=max_clusters
        # )

        genetic_centroids = self.calculate_genetic_centroids(
            self.genetic_data, geographic_assignments
        )

        outliers_geo = self.multi_stage_outlier_knn(
            self.geographic_data,
            self.genetic_data,
            "geographic",
            significance_level,
            max_k=max_k,
            min_nn_dist=min_nn_dist,
        )
        outliers_genetic = self.multi_stage_outlier_knn(
            self.genetic_data,
            self.geographic_data,
            "genetic",
            significance_level,
            max_k=max_k,
            min_nn_dist=min_nn_dist,
        )

        # Determine the misclustered samples
        misclustered_indices = self.find_misclustered_samples(
            self.geographic_data,
            self.genetic_data,
            geographic_centroids,
            genetic_centroids,
            geographic_assignments,
        )

        # Determine the correct centroids for the misclustered samples
        correct_centroids_geo = {
            idx: genetic_centroids[
                self.find_closest_centroid_id(
                    self.genetic_data[idx], list(genetic_centroids.values())
                )
            ]
            for idx in misclustered_indices["geographic"]
        }
        correct_centroids_gen = {
            idx: geographic_centroids[
                "Cluster "
                + str(
                    self.find_closest_centroid_id(
                        self.geographic_data[idx], list(geographic_centroids.values())
                    )
                )
            ]
            for idx in misclustered_indices["genetic"]
        }

        # Call the plotting function
        self.plotting.plot_outliers_with_traces(
            self.genetic_data,
            self.geographic_data_original,
            outliers_genetic,
            outliers_geo,
            correct_centroids_gen,
            correct_centroids_geo,
            self.url,
            self.buffer,
        )

        # Return composite outliers
        composite_outliers = set(outliers_geo).union(set(outliers_genetic))
        return np.array(list(composite_outliers))
