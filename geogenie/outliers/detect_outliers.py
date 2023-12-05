import logging
import os
import time
from multiprocessing import Pool
from os import path

import numpy as np
from pynndescent import NNDescent
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import gamma
from sklearn.metrics.pairwise import haversine_distances as sklearn_haversine
from sklearn.neighbors import NearestNeighbors

from geogenie.plotting.plotting import PlotGenIE
from geogenie.utils.scorers import calculate_r2_knn, haversine_distance


class GeoGeneticOutlierDetector:
    """A class to detect outliers based on genomic SNPs and geographic coordinates."""

    def __init__(
        self,
        genetic_data,
        geographic_data,
        output_dir,
        prefix,
        n_jobs=-1,
        seed=None,
        url="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip",
        buffer=0.1,
        show_plots=False,
        debug=False,
        verbose=0,
    ):
        """Initialize GeoGeneticOutlierDetector.

        Args:
            genetic_data (pd.DataFrame): The SNP data as a DataFrame.
            geographic_data (np.array): geographic coordinates as 2D array.
            output_dir (str): Output directory.
            prefix (str): Prefix for output files.
            seed (int): Random seed to use. If None, then the seed is random.
            url (str): url to download base map for plots.
            buffer (float): Buffer to put around sampling area on base map when plotting.
            show_plots (bool): Whether to show plots in-line.
            debug (bool): If True, writes genetic_data and geographic_data to file.
            verbose (int): Verbosity setting (0-2), least to most verbose.
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
        self.n_jobs = n_jobs
        self.seed = seed
        self.url = url
        self.buffer = buffer
        self.show_plots = show_plots
        self.logger = logging.getLogger(__name__)
        self.plotting = PlotGenIE("cpu", output_dir, prefix)
        self.verbose = verbose

    def calculate_dg(pred_gen, gen_coords):
        """Calculate the Dg statistic for genetic coordinates.

        Args:
            pred_gen (np.array): Predicted genetic coordinates.
            gen_coords (np.array): Actual genetic coordinates.

        Returns:
            np.array: Calculated Dg values.
        """
        # Calculating mean squared error for each row
        return np.mean((pred_gen - gen_coords) ** 2, axis=1)

    def calculate_dgeo(self, pred_geo_coords, geo_coords, scalar):
        """Calculate the Dgeo statistic for geographic coordinates.

        Args:
            pred_geo_coords (np.array): Predicted geographic coordinates.
            geo_coords (np.array): Actual geographic coordinates.
            scalar (float): Scalar value to divide the distance.

        Returns:
            np.array: Calculated Dgeo values.
        """
        if self.verbose >= 2:
            self.logger.info("Estimating Dgeo statistic...")

        if pred_geo_coords.shape[1] > 2:
            msg = f"Invalid shape for pred_geo_coords: {pred_geo_coords.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        if geo_coords.shape[1] > 2:
            msg = f"Invalid shape for geo_coords: {geo_coords.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        distances = (
            np.diagonal(cdist(geo_coords, pred_geo_coords, metric=haversine_distance))
            / scalar
        )

        if self.verbose >= 2:
            self.logger.info("Finished estimating Dgeo statistic.")
        return distances

    def calculate_genetic_distances(self, gen_coords):
        """Calculate Euclidean distances between genetic coordinates.

        Args:
            gen_coords (np.array): Genetic coordinates.

        Returns:
            np.array: Distance matrix for genetic data.
        """
        if gen_coords is None or len(gen_coords) == 0:
            raise ValueError("Genetic coordinates are empty or not properly formatted.")

        if self.verbose >= 2:
            self.logger.info("Estimating genetic distance matrix...")

        # Calculate the Euclidean distance matrix
        dist_matrix = cdist(gen_coords, gen_coords, metric="euclidean")

        # Add a small value to diagonal to avoid zeros (if required)
        np.fill_diagonal(dist_matrix, 1e-8)

        if self.verbose >= 2:
            self.logger.info("Finished estimating genetic distances.")
        return dist_matrix

    def calculate_geographic_distances(
        self,
        geo_coords,
        scale_factor=100,
        min_nn_dist=None,
        eps=1e-8,
    ):
        """Calculate scaled Haversine distances between geographic coordinates.

        Args:
            geo_coords (np.array): Geographic coordinates (latitude, longitude).
            scale_factor (float): Factor to scale the distance.
            min_nn_dist (float): Minimum neighbor distance to consider.

        Returns:
            np.array: Scaled distance matrix.
        """
        if self.verbose >= 2:
            self.logger.info("Estimating geographic distance matrix...")

        geo_coords = np.flip(geo_coords, axis=1)
        if geo_coords is None or geo_coords.shape[1] != 2:
            msg = f"Geographic coordinates must be a 2D array with two columns (longitude, latitude): {geo_coords.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Convert latitude and longitude from degrees to radians
        geo_coords_rad = np.radians(geo_coords)

        # Calculate the Haversine distances, which returns results
        # in radians
        # Earth radius in km = 6371.0
        dist_matrix = sklearn_haversine(geo_coords_rad) * 6371.0

        # Scale the distance matrix
        dist_matrix /= scale_factor

        # Handle minimum nearest neighbor distance
        if min_nn_dist is not None:
            min_nn_dist_scaled = min_nn_dist / scale_factor
            np.fill_diagonal(dist_matrix, 0)
            dist_matrix[dist_matrix < min_nn_dist_scaled] = eps

        if self.verbose >= 2:
            self.logger.info("Finished calculating geographic distance matrix.")
        return dist_matrix

    def calculate_statistic(
        self, predicted_data, actual_data, is_genetic, scale_factor
    ):
        """Calculate the Dg or Dgeo statistic based on the difference between predicted and actual data.

        Args:
            predicted_data (np.array): Predicted data from KNN.
            actual_data (np.array): Actual data.
            is_genetic (bool): Flag to determine if the calculation is for genetic data.
            scale_factor (float): Scaling factor for geo_coords.

        Returns:
            np.array: Dg or Dgeo statistic for each sample.
        """
        if is_genetic:
            # Dg calculation (mean squared error)
            D_statistic = np.mean((predicted_data - actual_data) ** 2, axis=1)
        else:
            # Dgeo calculation (geographical distance)
            D_statistic = self.calculate_dgeo(predicted_data, actual_data, scale_factor)

        D_statistic = self.rescale_statistic(D_statistic)
        D_statistic[D_statistic == 0] = 1e-8  # To avoid zeros in D_statistic
        return D_statistic

    def rescale_statistic(self, D_statistic):
        """Rescale the D statistic if its maximum value exceeds a threshold.

        Args:
            D_statistic (np.array): Dg or Dgen statistic.

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
        """Detect outliers based on geographic data using the KNN approach.

        Args:
            geo_coords (np.array): Array of geographic coordinates.
            gen_coords (np.array): Array of genetic data coordinates.
            maxk (int): Maximum number of neighbors to search for optimal K.
            min_nn_dist (float): Minimum neighbor distance for geographic KNN.
            w_power (float): Power of distance weight in KNN prediction.
            sig_level (float): Significance level for detecting outliers.
            scale_factor (int): Scaling factor for geogrpahic coordinates.

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
            (1, maxk),  # does k + 1
            w_power,
            min_nn_dist,
            is_genetic=False,
        )

        if self.verbose >= 1:
            self.logger.info(f"Optimal K for KNN Genetic Regression: {optk}")

        # Step 2: Find KNN based on geographic distances
        knn_indices = self.find_geo_knn(optk, min_nn_dist)

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

        # Find optimal K using genetic distances to predict geo coords
        optk = self.find_optimal_k(
            geo_coords, gen_dist_matrix, (1, maxk), w_power, min_nn_dist=min_nn_dist
        )

        knn_indices = self.find_gen_knn(optk)

        # Predict geographic coordinates using weighted genetic KNN
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

    def find_gen_knn(self, coords, k):
        """Find K-nearest neighbors for genetic data using PyNNDescent.

        Args:
            dist_matrix (np.array): Distance matrix.
            k (int): Number of neighbors.

        Returns:
            np.array: Indices of K-nearest neighbors.
        """
        # As before, set the diagonal to infinity to ignore self-matching
        # np.fill_diagonal(dist_matrix, np.finfo(np.float32).max)

        # Initialize NNDescent with the precomputed distance matrix
        nnd = NNDescent(
            coords,
            metric="euclidean",
            n_neighbors=k + 1,
            n_jobs=self.n_jobs,
        )
        indices, distances = nnd.neighbor_graph

        # Exclude the first column which is the point itself
        return indices[:, 1 : k + 1], distances[:, 1 : k + 1]

    def find_geo_knn(self, coords, k, min_nn_dist):
        """Find K-nearest neighbors for geographic data considering minimum neighbor distance using PyNNDescent.

        Args:
            dist_matrix (np.array): Distance matrix.
            k (int): Number of neighbors.
            min_nn_dist (float): Minimum distance to consider for neighbors.

        Returns:
            np.array: Indices of K-nearest neighbors.
        """
        coords = np.flip(coords)

        # Initialize NNDescent with the precomputed distance matrix
        nnd = NNDescent(
            coords,
            metric="euclidean",
            n_neighbors=k + 1,
            n_jobs=self.n_jobs,
        )
        indices, distances = nnd.neighbor_graph

        # Exclude neighbors that are too close (less than min_nn_dist)
        valid_indices = distances >= min_nn_dist
        result_indices = np.where(valid_indices, indices, -1)

        # Truncate or pad the result_indices to ensure exactly k neighbors
        # (excluding the point itself which is at index 0)
        k_indices = result_indices[:, 1 : k + 1]

        return k_indices, distances[:, 1 : k + 1]

    def compute_d_statistic_for_k_range(self, args):
        (
            range_k,
            coords,
            dist_matrix,
            w_power,
            min_nn_dist,
            is_genetic,
            scale_factor,
            find_knn_function,
            predict_function,
            calculate_statistic_function,
        ) = args
        all_D = []
        for k in range_k:
            # Assuming find_knn_function, predict_function, and calculate_statistic_function
            # are provided as arguments and replicate the logic of the corresponding instance methods.
            knn_indices = find_knn_function(dist_matrix, k, min_nn_dist)
            predictions = predict_function(coords, dist_matrix, knn_indices, w_power)
            D_statistic = calculate_statistic_function(
                predictions, coords, is_genetic, scale_factor
            )
            all_D.append(np.sum(D_statistic))
        return all_D

    def find_optimal_k(
        self,
        coords,
        klim,
        w_power,
        min_nn_dist,
        is_genetic,
        scale_factor,
    ):
        """Find optimal number of nearest neighbors for KNN.

        Args:
            coords (np.array): Genetic or geographic coordinates.
            dist_matrix (np.array): Distance matrix.
            klim (tuple): Range of K values to search (min_k, max_k).
            w_power (float): Power for distance weighting.
            min_nn_dist (int): Minimum nearest neighbor distance to consider points.
            is_genetic (bool): Flag to determine if the calculation is for genetic data as distance matrix.
            scale_factor (float): Factor to scale geo_coords by.

        Returns:
            int: Optimal K value.
        """
        self.logger.info("Finding optimal K for Nearest Neighbors...")

        min_k, max_k = klim
        all_D = []
        for k in range(min_k, max_k + 1):
            if is_genetic:
                # Genetic coords, geo dist matrix.
                knn_indices, distances = self.find_geo_knn(coords, k, min_nn_dist)

                # Genetic predictions.
                predictions = self.predict_coords_knn(
                    coords, distances, knn_indices, w_power
                )

                # Geographic error.
                D_statistic = self.calculate_statistic(
                    predictions, coords, is_genetic, scale_factor=scale_factor
                )
            else:
                # Geo coords, genetic dist matrix.
                knn_indices, distances = self.find_gen_knn(coords, k)

                # Geographic predictions.
                predictions = self.predict_coords_knn(
                    coords, distances, knn_indices, w_power
                )

                # Geographic error.
                D_statistic = self.calculate_statistic(
                    predictions, coords, is_genetic, scale_factor=scale_factor
                )

            all_D.append(np.sum(D_statistic))
        optimal_k = min_k + np.argmin(all_D)  # Adjust index to actual K value
        self.logger.info("Completed optimal K search.")
        return optimal_k

    def find_optimal_k_parallel(
        self,
        coords,
        dist_matrix,
        klim,
        w_power,
        min_nn_dist,
        is_genetic,
        scale_factor,
        n_processes=4,
    ):
        if self.verbose >= 2:
            self.logger.info("Finding optimal K for Nearest Neighbors in parallel...")

        n_processes = self.n_jobs

        if n_processes == -1:
            n_processes = os.cpu_count()

        if n_processes < -1 or n_processes == 0:
            raise ValueError(f"'n_jobs' must equal -1 or be > 0: {self.n_jobs}")

        find_knn_func = self.find_geo_knn if is_genetic else self.find_gen_knn

        min_k, max_k = klim
        k_range = range(min_k, max_k + 1)
        k_chunks = np.array_split(k_range, n_processes)

        # Prepare arguments for each chunk
        args = [
            (
                k_chunk,
                coords,
                dist_matrix,
                w_power,
                min_nn_dist,
                is_genetic,
                scale_factor,
                find_knn_func,
                self.predict_coords_knn,
                self.calculate_statistic,
            )
            for k_chunk in k_chunks
        ]

        # Use multiprocessing Pool
        with Pool(n_processes) as pool:
            results = pool.map(self.compute_d_statistic_for_k_range, args)

        # Flatten and combine results from all chunks
        all_D = [d for sublist in results for d in sublist]
        optimal_k = min_k + np.argmin(all_D)  # Adjust index to actual K value

        if self.verbose >= 2:
            self.logger.info("Completed optimal K search in parallel.")
        return optimal_k

    def predict_coords_knn(self, coords, knn_distances, knn_indices, w_power):
        """Predict coordinates data using weighted KNN.

        Args:
            coords (np.array): Array of genetic or geographic coordinates.
            knn_distances (np.array): Distances to K-nearest neighbors.
            knn_indices (np.array): Indices of K-nearest neighbors.
            w_power (float): Power of distance weight in prediction.

        Returns:
            np.array: Predicted coordinates using weighted KNN.
        """
        predictions = np.zeros_like(coords)

        for i in range(len(coords)):
            neighbor_distances = knn_distances[i]
            neighbor_indices = knn_indices[i]

            # Compute weights inversely proportional to the distance
            weights = 1 / (neighbor_distances + 1e-8) ** w_power
            normalized_weights = weights / np.sum(weights)

            # Calculate the weighted average of the neighbors
            predictions[i] = np.average(
                coords[neighbor_indices], axis=0, weights=normalized_weights
            )

        return predictions

    def fit_gamma_mle(self, D_statistic, Dgeo, sig_level, initial_params=None):
        """Detect outliers using a Gamma distribution fitted to the Dg or Dgeo statistic.

        Args:
            D_statistic (np.array): Dg or Dgeo statistic for each sample.
            Dgeo (np.array): For determining initial_shape and initial_rate.
            sig_level (float): Significance level for detecting outliers.
            initial_params (tuple): Initial shape and rate parameters for gamma distribution.

        Returns:
            tuple: Indices of outliers, p-values, and fitted Gamma parameters.
        """
        if initial_params:
            initial_shape, initial_rate = initial_params
        else:
            initial_shape = (np.mean(Dgeo) ** 2) / np.var(Dgeo)
            initial_rate = np.mean(Dgeo) / np.var(Dgeo)

        result = minimize(
            self.gamma_neg_log_likelihood,
            x0=(initial_shape, initial_rate),
            args=(D_statistic,),
            bounds=[(1e-8, 1e8), (1e-8, 1e8)],
            method="L-BFGS-B",
        )
        shape, rate = result.x
        scale = 1 / rate  # Convert rate back to scale for gamma distribution
        p_values = 1 - gamma.cdf(D_statistic, a=shape, scale=scale)
        outliers = np.where(p_values < sig_level)[0]
        gamma_params = (shape, scale)
        return outliers, p_values, gamma_params

    def gamma_neg_log_likelihood(self, params, data):
        """
        Negative log likelihood for gamma distribution.

        Args:
            params (tuple): Contains the shape and rate parameters for the gamma distribution.
            data (np.array): Data to fit the gamma distribution to.

        Returns:
            float: Negative log likelihood value.
        """
        shape, rate = params
        scale = 1 / rate  # Convert rate to scale for gamma distribution
        return -np.sum(gamma.logpdf(data, a=shape, scale=scale))

    def multi_stage_outlier_knn(
        self,
        geo_coords,
        gen_coords,
        idtype,
        sig_level=0.05,
        maxk=50,
        w_power=2,
        min_nn_dist=1000,
        scale_factor=100,
    ):
        """Iterative Outlier Detection via KNN for genetic and geographic data."""
        min_nn_dist /= scale_factor

        time_durations, optk_gen, optk_geo = self.search_nn_optk(
            geo_coords, gen_coords, maxk, w_power, min_nn_dist, scale_factor
        )

        if self.verbose >= 1:
            self.logger.info(
                f"Optimal K for Genetic Regression: {optk_gen}, Time taken: {time_durations['find_optimal_k_genetic']} seconds"
            )
            self.logger.info(
                f"Optimal K for Geographic Regression: {optk_geo}, Time taken: {time_durations['find_optimal_k_geographic']} seconds"
            )
        outlier_flags_geo = np.zeros(len(geo_coords), dtype=bool)
        outlier_flags_gen = np.zeros(len(gen_coords), dtype=bool)

        iteration = 0
        allow_geo = True
        allow_gen = True
        while True:
            iteration += 1
            new_outliers_detected = False

            if self.verbose >= 1:
                self.logger.info(
                    f"\nIteration {iteration} of multi-stage outlier detection.\n"
                )

            # if idtype == "genetic":
            #     filtered_gen_coords = gen_coords[~outlier_flags_gen]
            #     gen_dist = self.calculate_genetic_distances(filtered_gen_coords)
            #     current_outliers_gen = self.detect_genetic_outliers(
            #         filtered_gen_coords, gen_dist, maxk, sig_level
            #     )

            #     if current_outliers_gen.size:
            #         new_outliers_detected = True
            #         outlier_flags_gen[~outlier_flags_gen][current_outliers_gen] = True

            # if idtype == "geographic":
            #     filtered_geo_coords = geo_coords[~outlier_flags_geo]
            #     geo_dist = self.calculate_geographic_distances(
            #         filtered_geo_coords, scale_factor, min_nn_dist
            #     )
            #     current_outliers_geo = self.detect_geographic_outliers(
            #         filtered_geo_coords, geo_dist, maxk, sig_level
            #     )

            #     if current_outliers_geo.size:
            #         new_outliers_detected = True
            #         outlier_flags_geo[~outlier_flags_geo][current_outliers_geo] = True

            (
                time_durations,
                d_stats,
                gamma_params,
                p_values,
                current_outliers_geo,
                current_outliers_gen,
                filtered_indices,
            ) = self.do_composite(
                geo_coords,
                gen_coords,
                sig_level,
                min_nn_dist,
                scale_factor,
                optk_gen,
                optk_geo,
                outlier_flags_geo,
                outlier_flags_gen,
                time_durations,
            )

            # Map indices from filtered to original
            if current_outliers_geo is not None and current_outliers_geo.size > 0:
                actual_outlier_indices_geo = filtered_indices[current_outliers_geo]

                # Update the flags in the original arrays
                outlier_flags_geo[actual_outlier_indices_geo] = True
                new_outliers_detected = True
            else:
                allow_geo = False

            if current_outliers_gen is not None and current_outliers_gen.size > 0:
                actual_outlier_indices_gen = filtered_indices[current_outliers_gen]
                outlier_flags_gen[actual_outlier_indices_gen] = True
                new_outliers_detected = True
            else:
                allow_gen = False

            if not new_outliers_detected:
                if self.verbose >= 1:
                    self.logger.info("No new outliers detected. Terminating iteration.")
                break

            if not allow_gen and not allow_geo:
                if self.verbose >= 1:
                    self.logger.info("No new outliers detected. Terminating iteration.")
                break

            if self.verbose >= 2:
                self.logger.info(f"Finished iteration {iteration}.")

        if self.verbose >= 1:
            self.logger.info("Completed multi-stage outlier detection.")

        # Optional: Plotting Gamma distribution
        start_time = time.time()
        self.plot_gamma_dist(sig_level, d_stats, gamma_params)
        end_time = time.time()
        time_durations["plot_gamma_distribution"] = end_time - start_time

        if self.verbose >= 2:
            self.logger.info(
                f"Plotted Gamma distributions. Time taken: {time_durations['plotting_gamma_distribution']} seconds"
            )
        # Return the indices of the detected outliers
        if idtype == "composite":
            return np.where(outlier_flags_geo)[0], np.where(outlier_flags_gen)[0]
        elif idtype == "genetic":
            return np.where(outlier_flags_gen)[0]
        elif idtype == "geographic":
            return np.where(outlier_flags_geo)[0]

    def do_composite(
        self,
        geo_coords,
        gen_coords,
        sig_level,
        min_nn_dist,
        scale_factor,
        optk_gen,
        optk_geo,
        outlier_flags_geo,
        outlier_flags_gen,
        time_durations,
    ):
        non_outlier_mask = ~(outlier_flags_geo | outlier_flags_gen)

        # Keep track of original indices
        original_indices = np.arange(len(geo_coords))
        filtered_indices = original_indices[non_outlier_mask]

        filtered_geo_coords = geo_coords[non_outlier_mask]
        filtered_gen_coords = gen_coords[non_outlier_mask]

        (
            current_outliers_geo,
            current_outliers_gen,
            time_durations,
            d_stats,
            p_values,
            gamma_params,
        ) = self.detect_composite_outliers(
            filtered_geo_coords,
            filtered_gen_coords,
            optk_geo,
            optk_gen,
            time_durations,
            w_power=2,
            sig_level=sig_level,
            min_nn_dist=min_nn_dist,
            scale_factor=scale_factor,
        )

        # Return the filtered indices as well for mapping
        return (
            time_durations,
            d_stats,
            gamma_params,
            p_values,
            current_outliers_geo,
            current_outliers_gen,
            filtered_indices,
        )

    def search_nn_optk(
        self, geo_coords, gen_coords, maxk, w_power, min_nn_dist, scale_factor
    ):
        time_durations = {}

        # Step 1: Find optimal K for both genetic and geographic data
        start_time = time.time()
        optk_gen = self.find_optimal_k(
            gen_coords,
            (2, maxk),  # does k + 1
            w_power,
            min_nn_dist,
            is_genetic=True,
            scale_factor=scale_factor,
        )

        end_time = time.time()
        time_durations["find_optimal_k_genetic"] = end_time - start_time

        start_time = time.time()
        optk_geo = self.find_optimal_k(
            geo_coords,
            (1, maxk),  # does k + 1
            w_power,
            min_nn_dist,
            is_genetic=False,
            scale_factor=scale_factor,
        )
        end_time = time.time()
        time_durations["find_optimal_k_geographic"] = end_time - start_time
        return time_durations, optk_gen, optk_geo

    def plot_gamma_dist(
        self, sig_level, d_stats, gamma_params, dtypes=["geographic", "genetic"]
    ):
        outdir = path.join(self.output_dir, "plots")
        for dtype, gamma, dg in zip(dtypes, gamma_params, d_stats):
            fn = path.join(outdir, f"{self.prefix}_gamma_{dtype}.png")
            self.plotting.plot_gamma_distribution(
                gamma[0],
                gamma[1],
                dg,
                sig_level,
                fn,
                f"{dtype.capitalize()} Outlier Gamma Distribution",
            )

    def detect_composite_outliers(
        self,
        geo_coords,
        gen_coords,
        optk_geo,
        optk_gen,
        time_durations,
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
            optk_geo (int): Optimal K for nearest neigbbors (geographic).
            optk_gen (int): Optimal K for nearest neighbors (genetic).
            time_durations (dict): Dictionary storing run times for each method.
            w_power (float): Power of distance weight in KNN prediction.
            sig_level (float): Significance level for detecting outliers.
            min_nn_dist (int): Minimum distance required to consider points.
            scale_factor (int): Scaling factor for geo coordinates.

        Returns:
            tuple: Indices of detected outliers, p-values, and gamma parameters for geographic and genetic outliers.

        Returns:
            tuple: Indices of detected outliers, p-values, and gamma parameters for geographic and genetic outliers.
        """
        # Step 2: Find KNN based on both distances
        start_time = time.time()
        knn_indices_geo, geo_dist = self.find_geo_knn(geo_coords, optk_geo, min_nn_dist)
        knn_indices_gen, gen_dist = self.find_gen_knn(gen_coords, optk_gen)
        end_time = time.time()
        time_durations["find_knn"] = end_time - start_time

        if self.verbose >= 2:
            self.logger.info(
                f"Finding KNN indices, Time taken: {time_durations['find_knn']} seconds"
            )

        # Step 3: Predict using weighted KNN
        start_time = time.time()
        predicted_gen_data = self.predict_coords_knn(
            gen_coords, geo_dist, knn_indices_geo, w_power
        )
        predicted_geo_data = self.predict_coords_knn(
            geo_coords, gen_dist, knn_indices_gen, w_power
        )
        end_time = time.time()
        time_durations["predict_knn"] = end_time - start_time

        if self.verbose >= 2:
            self.logger.info(
                f"Predicting using weighted KNN, Time taken: {time_durations['predict_knn']} seconds"
            )

        # Step 4: Calculate D statistics and detect outliers
        start_time = time.time()
        dgen = self.calculate_statistic(
            predicted_gen_data, gen_coords, is_genetic=True, scale_factor=scale_factor
        )
        dgeo = self.calculate_statistic(
            predicted_geo_data, geo_coords, is_genetic=False, scale_factor=scale_factor
        )
        end_time = time.time()
        time_durations["calculate_statistic"] = end_time - start_time

        if self.verbose >= 2:
            self.logger.info(
                f"Calculating D statistics, Time taken: {time_durations['calculate_statistic']} seconds"
            )

        # Log r-squared values for predictions
        r2_gen = calculate_r2_knn(predicted_gen_data, gen_coords)
        r2_geo = calculate_r2_knn(predicted_geo_data, geo_coords)

        if self.verbose >= 1:
            self.logger.info(f"r-squared for genetic outlier detection: {r2_gen}")
            self.logger.info(f"r-squared for geographic outlier detection: {r2_geo}")

        # Step 5: Fit Gamma distribution and detect outliers
        start_time = time.time()
        outliers_gen, p_value_gen, gamma_params_gen = self.fit_gamma_mle(
            dgen, dgeo, sig_level
        )
        end_time = time.time()
        time_durations["fit_gamma_genetic"] = end_time - start_time

        if self.verbose >= 2:
            self.logger.info(
                f"Fitting gamma distribution for genetic outliers, Time taken: {time_durations['fit_gamma_genetic']} seconds"
            )

        start_time = time.time()
        outliers_geo, p_value_geo, gamma_params_geo = self.fit_gamma_mle(
            dgeo, dgeo, sig_level
        )
        end_time = time.time()
        time_durations["fit_gamma_geographic"] = end_time - start_time

        if self.verbose >= 2:
            self.logger.info(
                f"Fitted gamma distribution for geographic outliers. Time taken: {time_durations['fit_gamma_geographic']} seconds"
            )

        # Return the results and the timing data
        return (
            outliers_geo,
            outliers_gen,
            time_durations,
            [dgeo, dgen],
            [p_value_geo, p_value_gen],
            [gamma_params_geo, gamma_params_gen],
        )

    def composite_outlier_detection(self, sig_level=0.05, maxk=50, min_nn_dist=100):
        self.logger.info("Starting composite outlier detection...")

        dgen = self.genetic_data
        dgeo = self.geographic_data
        outliers = {}
        outliers["geographic"], outliers["genetic"] = self.multi_stage_outlier_knn(
            dgeo,
            dgen,
            "composite",
            sig_level,
            maxk=maxk,
            w_power=2,
            min_nn_dist=min_nn_dist,
            scale_factor=100,
        )
        self.logger.info("Outlier detection completed.")
        return outliers

    def compute_pairwise_p(self, distmat, gamma_params, k):
        """Compute p-values for each pair of sample and K nearest neighbors."""
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
