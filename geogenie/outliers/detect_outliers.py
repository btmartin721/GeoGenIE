import logging
import sys
from os import path

import numpy as np
from haversine import haversine
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import gamma
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import NearestNeighbors

from geogenie.plotting.plotting import PlotGenIE
from geogenie.utils.scorers import calculate_r2_knn, haversine_distance


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
        seed=None,
        url="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip",
        buffer=0.1,
        show_plots=False,
        debug=False,
    ):
        """
        Initializes GeoGeneticOutlierDetector with genetic and geographic data.

        Args:
            genetic_data (pd.DataFrame): The SNP data as a DataFrame.
            geographic_data (np.array): geographic coordinates as 2D array.
        """
        if debug:
            genetic_data.to_csv("data/test_eigen.csv", header=False, index=False)
            tmp = geographic_data.reset_index()
            tmp.columns = ["sampleID", "x", "y"]
            tmp.to_csv("data/test_coords_train.txt", header=True, index=False)
        self.genetic_data = genetic_data.to_numpy()
        self.geographic_data = geographic_data.to_numpy()
        self.output_dir = output_dir
        self.prefix = prefix
        self.seed = seed
        self.url = url
        self.buffer = buffer
        self.show_plots = show_plots
        self.logger = logging.getLogger(__name__)
        self.plotting = PlotGenIE("cpu", output_dir, prefix)

    def get_dist_matrices(self, data, dtype, eps=1e-6):
        """Get distance matrix for independent data."""
        if dtype not in ["genetic", "geographic", "composite"]:
            msg = f"'data_type' must be 'genetic' or 'geographic': {dtype}."
            self.logger.error(msg)
            raise ValueError(msg)
        metric = (
            "euclidean" if dtype in ["genetic", "composite"] else haversine_distance
        )
        distances = squareform(pdist(data, metric=metric))
        distances[distances == 0] += eps  # Add small value to avoid dist of 0.
        return distances

    def calculate_dgeo(self, pred_geo_coords, actual_geo_coords, scale_factor):
        """Calculate Dgeo statistic, which is the scaled geographical distance between predicted and actual geographic coordinates.

        Args:
            pred_geo_coords (np.array): Predicted geographic coordinates.
            actual_geo_coords (np.array): Actual geographic coordinates.
            scalar (float): Scaling factor for the distances.

        Returns:
            np.array: Scaled distances representing the Dgeo statistic.
        """

        actual_geo_coords = actual_geo_coords.copy()
        pred_geo_coords = pred_geo_coords.copy()
        actual_geo_coords = actual_geo_coords[:, [1, 0]]
        pred_geo_coords = pred_geo_coords[:, [1, 0]]

        # Compute pairwise distances using Haversine formula
        distances = cdist(actual_geo_coords, pred_geo_coords, metric=haversine)

        # Diagonal of the distance matrix gives the distances between
        # corresponding points
        diag_distances = np.diagonal(distances)

        # Scale the distances
        Dgeo = diag_distances / scale_factor
        return Dgeo

    def calculate_geographic_distances(
        self, geo_coords, scale_factor=100, min_nn_dist=None
    ):
        # Check for empty or improperly formatted geo_coords
        if geo_coords is None or len(geo_coords) == 0:
            raise ValueError(
                "Geographic coordinates are empty or not properly formatted."
            )

        geo_coords_tmp = geo_coords[:, [1, 0]]

        # Convert latitude and longitude from degrees to radians
        geo_coords_rad = np.radians(geo_coords_tmp)

        # Calculate the Haversine distances, which returns results in radians
        dist_matrix = haversine_distances(geo_coords_rad)

        # Convert distance from radians to kilometers (Earth radius of 6371 km)
        dist_matrix *= 6371.0

        # Scale the distance matrix
        dist_matrix /= scale_factor

        # Add a small amount to zero distances if min_nn_dist is not None
        if min_nn_dist is not None:
            min_nn_dist_scaled = min_nn_dist / scale_factor
            np.fill_diagonal(
                dist_matrix, 0
            )  # Set diagonal to zero to avoid self-distances
            # Add one unit of distance to pairs with identical geographic
            # coordinates
            dist_matrix[dist_matrix == 0] += min_nn_dist_scaled

        return dist_matrix

    def calculate_statistic(self, predicted_data, actual_data):
        """
        Calculate the Dg or Dgeo statistic based on the difference between predicted and actual data.

        Args:
            predicted_data (np.array): Predicted data from KNN.
            actual_data (np.array): Actual data.

        Returns:
            np.array: Dg or Dgeo statistic for each sample.
        """
        D_statistic = np.sqrt(np.sum((predicted_data - actual_data) ** 2, axis=1))
        D_statistic = self.rescale_statistic(D_statistic)
        D_statistic[D_statistic == 0] = 1e-8  # To avoid zeros in D_statistic
        return D_statistic

    def rescale_statistic(self, D_statistic):
        """
        Rescale the D statistic if its maximum value exceeds a threshold.

        Args:
            D_statistic (np.array): Dg or Dgeo statistic.

        Returns:
            np.array: Rescaled Dg or Dgeo statistic.
        """
        max_threshold = 20
        while np.max(D_statistic) > max_threshold:
            D_statistic /= 10
        return D_statistic

    def detect_genetic_outliers(
        self,
        geo_coords,
        gen_coords,
        maxk,
        min_nn_dist=100,
        w_power=2,
        sig_level=0.95,
        scale_factor=100,
    ):
        """
        Detect outliers based on geographic data using the KNN approach.

        Args:
            geo_coords (np.array): Array of geographic coordinates.
            gen_coords (np.array): Array of genetic data coordinates.
            k (int): Number of neighbors.
            min_nn_dist (float): Minimum neighbor distance for geographic KNN.
            w_power (float): Power of distance weight in KNN prediction.
            sig_level (float): Significance level for detecting outliers.

        Returns:
            np.array: Indices of detected outliers.
        """
        # Step 1: Calculate geographic distances and scale
        geo_dist_matrix = self.calculate_geographic_distances(
            geo_coords, scale_factor=scale_factor, min_nn_dist=min_nn_dist
        )
        gen_dist_matrix = self.get_dist_matrices(gen_coords, dtype="genetic")

        optk = self.find_optimal_k(
            gen_coords,
            geo_dist_matrix,
            (2, maxk),
            w_power,
            min_nn_dist,
            is_genetic=False,
        )

        self.logger.info(f"Optimal K for KNN Genetic Regression: {optk}")

        # Step 2: Find KNN based on geographic distances
        knn_indices = self.find_geo_knn(geo_dist_matrix, optk, min_nn_dist)

        # Step 3: Predict genetic data using weighted KNN
        predicted_gen_data = self.predict_coords_knn(
            gen_coords, geo_dist_matrix, knn_indices, w_power
        )

        predicted_geo_data = self.predict_coords_knn(
            geo_coords, gen_dist_matrix, knn_indices, w_power
        )

        # Step 4: Calculate Dg statistic and detect outliers
        dg = self.calculate_statistic(predicted_gen_data, gen_coords)
        dgeo = self.calculate_dgeo(
            predicted_geo_data, geo_coords, scale_factor=scale_factor
        )
        r2 = calculate_r2_knn(predicted_gen_data, gen_coords)
        self.logger.info(f"r-squared for genetic outlier detection: {r2}")
        outliers, p_values, gamma_params = self.fit_gamma_mle(dg, sig_level)
        fn = path.join(self.output_dir, "plots", f"{self.prefix}_gamma_genetic.png")

        self.plotting.plot_gamma_distribution(
            gamma_params[0],
            gamma_params[1],
            dg,
            sig_level,
            fn,
            "Genetic Outlier Gamma Distribution",
        )
        return outliers, p_values, gamma_params

    def detect_geographic_outliers(
        self,
        geo_coords,
        gen_coords,
        maxk,
        min_nn_dist=100,
        w_power=2,
        sig_level=0.95,
        scale_factor=100,
    ):
        # Calculate genetic distances
        gen_dist_matrix = self.get_dist_matrices(gen_coords, dtype="genetic")

        # Find the optimal K using genetic distances to predict geographic
        # coordinates
        optk = self.find_optimal_k(
            geo_coords, gen_dist_matrix, (2, maxk), w_power, min_nn_dist=min_nn_dist
        )

        knn_indices = self.find_gen_knn(gen_dist_matrix, optk)

        # Predict geographic coordinates using weighted KNN based on genetic
        # distances
        predicted_geo_coords = self.predict_coords_knn(
            geo_coords, gen_dist_matrix, knn_indices, w_power
        )

        # Calculate the Dgeo statistic
        dgeo = self.calculate_dgeo(predicted_geo_coords, geo_coords, scale_factor)

        r2 = calculate_r2_knn(predicted_geo_coords, geo_coords)
        self.logger.info(f"r-squared for geographic outlier detection: {r2}")
        outliers, p_values, gamma_params = self.fit_gamma_mle(dgeo, sig_level)
        fn = path.join(self.output_dir, "plots", f"{self.prefix}_gamma_geographic.png")

        self.plotting.plot_gamma_distribution(
            gamma_params[0],
            gamma_params[1],
            dgeo,
            sig_level,
            fn,
            "Geographic Outlier Gamma Distribution",
        )
        return outliers, p_values, gamma_params

    def find_gen_knn(self, dist_matrix, k):
        """
        Find K-nearest neighbors for geographic data considering minimum neighbor distance.

        Args:
            dist_matrix (np.array): Distance matrix.
            k (int): Number of neighbors.
            min_nn_dist (float): Minimum distance to consider for neighbors.

        Returns:
            np.array: Indices of K-nearest neighbors.
        """
        neighbors = NearestNeighbors(n_neighbors=k, metric="precomputed")
        neighbors.fit(dist_matrix)
        _, indices = neighbors.kneighbors(dist_matrix)
        return indices

    def find_geo_knn(self, dist_matrix, k, min_nn_dist):
        """
        Find K-nearest neighbors for geographic data considering minimum neighbor distance.

        Args:
            dist_matrix (np.array): Distance matrix.
            k (int): Number of neighbors.
            min_nn_dist (float): Minimum distance to consider for neighbors.

        Returns:
            np.array: Indices of K-nearest neighbors.
        """
        neighbors = NearestNeighbors(n_neighbors=k, metric="precomputed")
        neighbors.fit(dist_matrix)
        distances, indices = neighbors.kneighbors(dist_matrix)

        # Set indices to -1 if below min_nn_dist
        indices[distances < min_nn_dist] = -1
        return indices

    def find_optimal_k(
        self, coords, dist_matrix, klim, w_power, min_nn_dist, is_genetic=True
    ):
        """
        Find the optimal number of nearest neighbors (K) for either genetic or geographic KNN.

        Args:
            coords (np.array): Genetic or geographic coordinates.
            dist_matrix (np.array): Distance matrix.
            klim (tuple): Range of K values to search (min_k, max_k).
            w_power (float): Power for distance weighting.
            is_genetic (bool): Flag to determine if the calculation is for genetic data as distance matrix.

        Returns:
            int: Optimal K value.
        """
        min_k, max_k = klim
        all_D = []

        for k in range(min_k, max_k + 1):
            if is_genetic:
                knn_indices = self.find_gen_knn(dist_matrix, k)
                predictions = self.predict_coords_knn(
                    coords, dist_matrix, knn_indices, w_power
                )
                D_statistic = self.calculate_statistic(predictions, coords)
            else:
                knn_indices = self.find_geo_knn(dist_matrix, k, min_nn_dist)
                predictions = self.predict_coords_knn(
                    coords, dist_matrix, knn_indices, w_power
                )
                D_statistic = self.calculate_statistic(predictions, coords)

            all_D.append(np.sum(D_statistic))

        optimal_k = min_k + np.argmin(all_D)  # Adjust index to actual K value
        return optimal_k

    def predict_coords_knn(self, coords, dist_matrix, knn_indices, w_power):
        """Predict coordinates data using weighted KNN based on distance matrix.

        Args:
            coords (np.array): Array of genetic or geographic coordinates.
            dist_matrix (np.array): Precomputed distance matrix.
            knn_indices (np.array): Indices of K-nearest neighbors.
            w_power (float): Power of distance weight in prediction.

        Returns:
            np.array: Predicted coordinates using weighted KNN.
        """
        predictions = np.zeros_like(coords)

        for i in range(len(coords)):
            # Extract the distances and indices of the K-nearest neighbors
            neighbor_distances = dist_matrix[i, knn_indices[i]]
            neighbor_indices = knn_indices[i]

            # Compute weights inversely proportional to the distance
            # Avoid division by zero by adding a small constant
            weights = 1 / (neighbor_distances + 1e-8) ** w_power
            normalized_weights = weights / np.sum(weights)

            # Calculate the weighted average of the neighbors
            predictions[i] = np.average(
                coords[neighbor_indices], axis=0, weights=normalized_weights
            )

        return predictions

    def fit_gamma_mle(self, D_statistic, Dgeo, sig_level, initial_params=None):
        """
        Detect outliers using a Gamma distribution fitted to the Dg or Dgeo statistic.

        Args:
            D_statistic (np.array): Dg or Dgeo statistic for each sample.
            sig_level (float): Significance level for detecting outliers.
            initial_params (tuple): Initial shape and scale parameters for gamma distribution.

        Returns:
            tuple: Indices of outliers, p-values, and fitted Gamma parameters.
        """
        if initial_params:
            initial_shape, initial_scale = initial_params
        else:
            initial_shape = np.mean(Dgeo) ** 2 / np.var(Dgeo)
            initial_scale = np.mean(Dgeo) / np.var(Dgeo)

        result = minimize(
            self.gamma_neg_log_likelihood,
            x0=(initial_shape, initial_scale),
            args=(D_statistic,),
            bounds=[(1e-8, 1e8), (1e-8, 1e8)],
            method="L-BFGS-B",
        )
        shape, scale = result.x
        p_values = 1 - gamma.cdf(D_statistic, a=shape, scale=scale)
        outliers = np.where(p_values < sig_level)[0]
        gamma_params = (shape, scale)
        return outliers, p_values, gamma_params

    def gamma_neg_log_likelihood(self, params, data):
        """Negative log likelihood for gamma distribution.

        Args:
            params (tuple): Contains the shape and scale parameters for the gamma distribution.
            data (np.array): Data to fit the gamma distribution to.

        Returns:
            float: Negative log likelihood value.
        """
        shape, scale = params
        return -np.sum(gamma.logpdf(data, a=shape, scale=scale))

    def multi_stage_outlier_knn(
        self,
        geo_coords,
        gen_coords,
        idtype,
        sig_level=0.95,
        maxk=50,
        min_nn_dist=100,
        scale_factor=100,
    ):
        """Iterative Outlier Detection via KNN for genetic and geographic data."""
        outlier_flags = np.zeros(len(geo_coords), dtype=bool)
        original_indices = np.arange(len(geo_coords))
        outlier_flags_geo = np.zeros(len(geo_coords), dtype=bool)
        outlier_flags_gen = np.zeros(len(gen_coords), dtype=bool)
        original_indices = np.arange(len(geo_coords))

        if idtype not in ["genetic", "geographic", "composite"]:
            raise ValueError(f"'idtype' must be 'genetic' or 'geographic': {idtype}")

        iteration = 0
        while True:
            iteration += 1
            if idtype == "genetic":
                distances = self.get_dist_matrices(gen_coords, idtype)
                mask = ~outlier_flags
                filtered_distances = distances[mask][:, mask]
                filtered_indices = original_indices[mask]
                (
                    current_outliers,
                    p_values,
                    gamma_params,
                ) = self.detect_geographic_outliers(
                    geo_coords[mask],
                    filtered_distances,
                    maxk=maxk,
                    sig_level=sig_level,
                    min_nn_dist=min_nn_dist,
                )
            elif idtype == "geographic":
                distances = self.calculate_geographic_distances(
                    geo_coords, scale_factor=scale_factor, min_nn_dist=min_nn_dist
                )
                mask = ~outlier_flags
                filtered_distances = distances[mask][:, mask]
                filtered_indices = original_indices[mask]
                (
                    current_outliers,
                    p_values,
                    gamma_params,
                ) = self.detect_genetic_outliers(
                    gen_coords[mask],
                    filtered_distances,
                    maxk=maxk,
                    min_nn_dist=min_nn_dist,
                    sig_level=sig_level,
                )
            elif idtype == "composite":
                geo_dist = self.calculate_geographic_distances(
                    geo_coords, scale_factor=scale_factor, min_nn_dist=min_nn_dist
                )
                gen_dist = self.get_dist_matrices(gen_coords, dtype="genetic")
                mask = np.logical_or(~outlier_flags_geo, ~outlier_flags_gen)
                filtered_gen_distances = gen_dist[mask][:, mask]
                filtered_geo_distances = geo_dist[mask][:, mask]
                filtered_indices_geo = original_indices[mask]
                filtered_indices_gen = original_indices[mask]

                (
                    current_outliers_geo,
                    current_outliers_gen,
                    p_values,
                    gamma_params,
                ) = self.detect_composite_outliers(
                    geo_coords[mask],
                    gen_coords[mask],
                    filtered_geo_distances,
                    filtered_gen_distances,
                    maxk=maxk,
                    sig_level=sig_level,
                    min_nn_dist=min_nn_dist,
                )

            if idtype == "composite":
                # Check if no new outliers are detected
                if len(current_outliers_gen) == 0 or len(current_outliers_geo) == 0:
                    break
            else:
                if len(current_outliers) == 0:
                    break

            if idtype != "composite":
                # Update the outlier flags
                original_outlier_indices = filtered_indices[current_outliers]
                outlier_flags[original_outlier_indices] = True
            else:
                original_outlier_indices_geo = filtered_indices_geo[
                    current_outliers_geo
                ]
                original_outlier_indices_gen = filtered_indices_gen[
                    current_outliers_gen
                ]
                outlier_flags_geo[original_outlier_indices_geo] = True
                outlier_flags_gen[original_outlier_indices_gen] = True

        if idtype == "composite":
            return np.where(outlier_flags_geo)[0], np.where(outlier_flags_gen)[0]

        else:
            return np.where(outlier_flags)[0]

    def detect_composite_outliers(
        self,
        geo_coords,
        gen_coords,
        geo_dist,
        gen_dist,
        maxk=50,
        w_power=2,
        sig_level=0.05,
        min_nn_dist=100,
        scale_factor=100,
    ):
        """
        Detect outliers based on composite data using the KNN approach.

        Args:
            geo_coords (np.array): Array of geographic coordinates.
            gen_coords (np.array): Array of genetic data coordinates.
            k (int): Number of neighbors.
            min_nn_dist (float): Minimum neighbor distance for geographic KNN.
            w_power (float): Power of distance weight in KNN prediction.
            sig_level (float): Significance level for detecting outliers.

        Returns:
            np.array: Indices of detected outliers.
        """
        optk_gen = self.find_optimal_k(
            gen_coords,
            geo_dist,
            (2, maxk),
            w_power,
            min_nn_dist,
            is_genetic=False,
        )

        optk_geo = self.find_optimal_k(
            geo_coords,
            gen_dist,
            (2, maxk),
            w_power,
            min_nn_dist,
            is_genetic=True,
        )

        self.logger.info(f"Optimal K for KNN Genetic Regression: {optk_gen}")
        self.logger.info(f"Optimal K for KNN Geographic Regression: {optk_geo}")

        # Step 2: Find KNN based on geographic distances
        knn_indices_geo = self.find_geo_knn(geo_dist, optk_geo, min_nn_dist)
        knn_indices_gen = self.find_gen_knn(gen_dist, optk_gen)

        # Step 3: Predict genetic data using weighted KNN
        predicted_gen_data = self.predict_coords_knn(
            gen_coords, geo_dist, knn_indices_geo, w_power
        )

        predicted_geo_data = self.predict_coords_knn(
            geo_coords, gen_dist, knn_indices_gen, w_power
        )

        # Step 4: Calculate Dg statistic and detect outliers
        dgen = self.calculate_statistic(predicted_gen_data, gen_coords)
        dg = self.calculate_statistic(predicted_geo_data, geo_coords)
        dgeo = self.calculate_dgeo(predicted_geo_data, geo_coords, scale_factor)
        r2 = calculate_r2_knn(predicted_gen_data, gen_coords)
        self.logger.info(f"r-squared for genetic outlier detection: {r2}")

        r2 = calculate_r2_knn(predicted_geo_data, geo_coords)
        self.logger.info(f"r-squared for geographic outlier detection: {r2}")

        outliers_gen, p_value_gen, gamma_params_gen = self.fit_gamma_mle(
            dgen, dgeo, sig_level
        )

        outliers_geo, p_value_geo, gamma_params_geo = self.fit_gamma_mle(
            dgeo, dgeo, sig_level
        )

        fn = path.join(self.output_dir, "plots", f"{self.prefix}_gamma_geographic.png")

        self.plotting.plot_gamma_distribution(
            gamma_params_geo[0],
            gamma_params_geo[1],
            dg,
            sig_level,
            fn,
            "Geographic Outlier Gamma Distribution",
        )

        fn = path.join(self.output_dir, "plots", f"{self.prefix}_gamma_genetic.png")

        self.plotting.plot_gamma_distribution(
            gamma_params_gen[0],
            gamma_params_gen[1],
            dgen,
            sig_level,
            fn,
            "Genetic Outlier Gamma Distribution",
        )

        return (
            outliers_geo,
            outliers_gen,
            (p_value_geo, p_value_gen),
            (gamma_params_geo, gamma_params_gen),
        )

    def composite_outlier_detection(
        self, sig_level=0.05, maxk=50, min_nn_dist=100, max_clusters=10
    ):
        self.logger.info("Starting composite outlier detection...")

        dgen = self.genetic_data
        dgeo = self.geographic_data
        # geocent, geoassn = self.cluster_centroids(dgeo, max_clusters)
        # gencent = self.genetic_centroids(dgen, geoassn)
        outliers = {}
        for idtype in ["composite"]:
            outliers[idtype] = self.multi_stage_outlier_knn(
                dgeo,
                dgen,
                idtype,
                sig_level,
                maxk=maxk,
                min_nn_dist=min_nn_dist,
                scale_factor=100,
            )

        self.logger.info("Outlier detection completed.")
        return outliers

        # misclust_idx = self.misclust_samp(dgeo, dgen, geocent, gencent, geoassn)

        # # Get correct centroids for misclustered samps
        # corr_cent_geo = {
        #     idx: gencent[self.closest_cent(dgen[idx], list(gencent.values()))]
        #     for idx in misclust_idx["geographic"]
        # }
        # corr_cent_gen = {
        #     idx: geocent[
        #         f"Cluster {str(self.closest_cent(dgeo[idx], list(geocent.values())))}"
        #     ]
        #     for idx in misclust_idx["genetic"]
        # }
        # args = [dgen, dgeo, outliers["genetic"], outliers["geographic"]]
        # args += [corr_cent_gen, corr_cent_geo, self.url, self.buffer]
        # self.plotting.plot_outliers_with_traces(*args)
        # return outliers

    def compute_pairwise_p(self, distmat, gamma_params, k):
        """Compute p-values for each pair of sample and its K nearest neighbors."""
        shape, scale, _ = gamma_params
        n_samples = distmat.shape[0]
        pairwise_p_val = np.zeros((n_samples, k))
        for i in range(n_samples):
            sorted_indices = np.argsort(distmat[i])
            for j in range(k):
                neighbor_index = sorted_indices[j]
                dist = distmat[i, neighbor_index]
                pairwise_p_val[i, j] = 1 - gamma.cdf(dist, a=shape, scale=scale)
        return pairwise_p_val

    def construct_adjacency_matrix(self, pairwise_pval, distmat, k):
        """Construct an adjacency matrix based on pairwise p-values."""
        n_samples = distmat.shape[0]
        adjacency_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            knn_indices = np.argsort(distmat[i])[:k]
            for j in knn_indices:
                # Direct assignment of p-value as connection strength
                adjacency_matrix[i, j] = pairwise_pval[i, j]
        return self.adjust_mutual_neighborhoods(adjacency_matrix)

    def adjust_mutual_neighborhoods(self, adjacency_mat):
        """Adjust the adjacency matrix for mutual neighborhoods."""
        n_samples = adjacency_mat.shape[0]
        for i in range(n_samples):
            for j in range(n_samples):
                if adjacency_mat[i, j] != adjacency_mat[j, i]:
                    adjacency_mat[i, j] = adjacency_mat[j, i] = max(
                        adjacency_mat[i, j], adjacency_mat[j, i]
                    )
        return adjacency_mat

    def cluster_centroids(self, data, max_clusters):
        """Gets geographic cluster centroids.

        Args:
            max_clusters (int): Maximum clusters to form. Gets optimized.

        Returns:
            dict: Dictionary with cluster ids as keys and centroid coords as values.
        """
        optk = self.find_optk(max_clusters=max_clusters)
        kmeans = KMeans(optk, random_state=self.seed, n_init="auto").fit(data)
        cent, cluster_assn = kmeans.cluster_centers_, kmeans.labels_
        d = {f"Cluster {i}": tuple(centroid) for i, centroid in enumerate(cent)}
        return d, cluster_assn

    def misclust_samp(
        self,
        dgeo,
        dgen,
        geographic_centroids,
        gen_cent,
        curr_clust_assn,
    ):
        """
        Identify samples that are incorrectly clustered, considering both geographic and genetic distances.

        Args:
            dgeo (numpy.ndarray): Array of geographic coordinates.
            dgen (numpy.ndarray): Array of genetic data.
            geo_cent(numpy.ndarray): Array of geographic centroids.
            gen_cent (numpy.ndarray): Array of genetic centroids.
            curr_clust_assn (numpy.ndarray): Current cluster assignments for each sample.

        Returns:
            dict: Dictionary with keys 'genetic' and 'geographic', each containing an array of indices representing samples that are incorrectly clustered.
        """
        misclust_samp = {"genetic": [], "geographic": []}
        for idx, gen_sample in enumerate(dgen):
            closest_gen_cent = self.closest_cent(gen_sample, gen_cent)
            if closest_gen_cent != curr_clust_assn[idx]:
                misclust_samp["genetic"].append(idx)
        for idx, geo_sample in enumerate(dgeo):
            closest_geo_cent = self.closest_cent(
                geo_sample, geographic_centroids.values()
            )
            if closest_geo_cent != curr_clust_assn[idx]:
                misclust_samp["geographic"].append(idx)
        return misclust_samp

    def closest_cent(self, sample, centroids):
        """Find the closest centroid to a given sample.

        Args:
            sample (numpy.ndarray): The sample data (genetic or geographic).
            centroids (list): List of centroids (genetic or geographic).

        Returns:
            int: The ID of the closest centroid.
        """
        min_distance = float("inf")
        closest_cent_id = -1
        if isinstance(centroids, dict):
            centroids = centroids.values()
        for centroid_id, centroid in enumerate(centroids):
            distance = np.linalg.norm(sample - centroid)
            if distance < min_distance:
                min_distance = distance
                closest_cent_id = centroid_id
        return closest_cent_id

    def genetic_centroids(self, genetic_data, cluster_assignments):
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
