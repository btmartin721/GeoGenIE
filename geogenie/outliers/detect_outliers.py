import logging
import os
import time
from os import path

import numpy as np
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

        self.logger.info("Estimating genetic distance matrix...")

        # Calculate the Euclidean distance matrix
        dist_matrix = cdist(gen_coords, gen_coords, metric="euclidean")

        # Add a small value to diagonal to avoid zeros (if required)
        np.fill_diagonal(dist_matrix, 1e-8)
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

        # Find optimal K using genetic distances to predict geo coords
        optk = self.find_optimal_k(
            geo_coords, gen_dist_matrix, (2, maxk), w_power, min_nn_dist=min_nn_dist
        )

        knn_indices = self.find_gen_knn(gen_dist_matrix, optk)

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

    def find_gen_knn(self, dist_matrix, k):
        """Find K-nearest neighbors for geographic data considering minimum neighbor distance.

        Args:
            dist_matrix (np.array): Distance matrix.
            k (int): Number of neighbors.

        Returns:
            np.array: Indices of K-nearest neighbors.
        """
        neighbors = NearestNeighbors(
            n_neighbors=k, metric="precomputed", n_jobs=self.n_jobs
        )
        neighbors.fit(dist_matrix)
        _, indices = neighbors.kneighbors(dist_matrix)
        return indices

    def find_geo_knn(self, dist_matrix, k, min_nn_dist):
        """Find K-nearest neighbors for geographic data considering minimum neighbor distance.

        Args:
            dist_matrix (np.array): Distance matrix.
            k (int): Number of neighbors.
            min_nn_dist (float): Minimum distance to consider for neighbors.

        Returns:
            np.array: Indices of K-nearest neighbors.
        """
        n_samples = dist_matrix.shape[0]
        neighbors = NearestNeighbors(
            n_neighbors=k + 1, metric="precomputed", n_jobs=self.n_jobs
        )
        neighbors.fit(dist_matrix)
        distances, indices = neighbors.kneighbors(dist_matrix)

        # Exclude neighbors that are too close (less than min_nn_dist)
        for i in range(n_samples):
            valid_neighbors = distances[i] >= min_nn_dist
            valid_neighbor_indices = indices[i, valid_neighbors]

            # In case there are fewer than k valid neighbors, truncate or pad
            # with -1
            if len(valid_neighbor_indices) > k:
                indices[i] = valid_neighbor_indices[
                    1 : k + 1
                ]  # Skip the first one as it's the point itself
            else:
                indices[i, : len(valid_neighbor_indices)] = valid_neighbor_indices[1:]
                # Pad with -1 if fewer than k valid neighbors
                indices[i, len(valid_neighbor_indices) :] = -1

        return indices

    def find_optimal_k(
        self,
        coords,
        dist_matrix,
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
                knn_indices = self.find_geo_knn(dist_matrix, k, min_nn_dist)

                # Genetic predictions.
                predictions = self.predict_coords_knn(
                    coords, dist_matrix, knn_indices, w_power
                )

                # Geographic error.
                D_statistic = self.calculate_statistic(
                    predictions, coords, is_genetic, scale_factor=scale_factor
                )
            else:
                # Geo coords, genetic dist matrix.
                knn_indices = self.find_gen_knn(dist_matrix, k)

                # Geographic predictions.
                predictions = self.predict_coords_knn(
                    coords, dist_matrix, knn_indices, w_power
                )

                # Geographic error.
                D_statistic = self.calculate_statistic(
                    predictions, coords, is_genetic, scale_factor=scale_factor
                )

            all_D.append(np.sum(D_statistic))
        optimal_k = min_k + np.argmin(all_D)  # Adjust index to actual K value
        self.logger.info("Completed optimal K search.")
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
        min_nn_dist=100,
        scale_factor=100,
    ):
        """Iterative Outlier Detection via KNN for genetic and geographic data."""
        outlier_flags_geo = np.zeros(len(geo_coords), dtype=bool)
        outlier_flags_gen = np.zeros(len(gen_coords), dtype=bool)
        original_indices = np.arange(len(geo_coords))

        iteration = 0
        while True:
            iteration += 1
            self.logger.info(f"Iteration {iteration} in multi-stage outlier detection.")

            if idtype == "genetic":
                # Apply KNN regression and outlier detection for genetic data
                filtered_gen_coords = gen_coords[~outlier_flags_gen]
                gen_dist = self.calculate_genetic_distances(filtered_gen_coords)
                current_outliers_gen = self.detect_genetic_outliers(
                    filtered_gen_coords, gen_dist, maxk, sig_level
                )

                # Update genetic outlier flags
                outlier_flags_gen[original_indices[current_outliers_gen]] = True

            elif idtype == "geographic":
                # Apply KNN regression and outlier detection for geographic data
                filtered_geo_coords = geo_coords[~outlier_flags_geo]
                geo_dist = self.calculate_geographic_distances(
                    filtered_geo_coords, scale_factor, min_nn_dist
                )
                current_outliers_geo = self.detect_geographic_outliers(
                    filtered_geo_coords, geo_dist, maxk, sig_level
                )

                # Update geographic outlier flags
                outlier_flags_geo[original_indices[current_outliers_geo]] = True

            elif idtype == "composite":
                # Apply KNN regression and outlier detection for composite data
                mask = np.logical_or(~outlier_flags_geo, ~outlier_flags_gen)
                filtered_geo_coords = geo_coords[mask]
                filtered_gen_coords = gen_coords[mask]
                geo_dist = self.calculate_geographic_distances(
                    filtered_geo_coords, scale_factor, min_nn_dist
                )
                gen_dist = self.calculate_genetic_distances(filtered_gen_coords)

                (
                    current_outliers_geo,
                    current_outliers_gen,
                    p_values,
                    gamma_params,
                ) = self.detect_composite_outliers(
                    filtered_geo_coords,
                    filtered_gen_coords,
                    geo_dist,
                    gen_dist,
                    maxk,
                    sig_level,
                    min_nn_dist,
                )

                # Update composite outlier flags
                outlier_flags_geo[mask][current_outliers_geo] = True
                outlier_flags_gen[mask][current_outliers_gen] = True

            # Check for termination condition
            if idtype != "composite":
                if not any(
                    current_outliers_gen
                    if idtype == "genetic"
                    else current_outliers_geo
                ):
                    break
            else:
                if not any(current_outliers_geo) and not any(current_outliers_gen):
                    break

        # Return the indices of the detected outliers
        if idtype == "composite":
            return np.where(outlier_flags_geo)[0], np.where(outlier_flags_gen)[0]
        elif idtype == "genetic":
            return np.where(outlier_flags_gen)[0]
        elif idtype == "geographic":
            return np.where(outlier_flags_geo)[0]

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
            geo_dist (np.array): Geographic distance matrix.
            gen_dist (np.array): Genetic distance matrix.
            maxk (int): Maximum number of neighbors to consider.
            w_power (float): Power of distance weight in KNN prediction.
            sig_level (float): Significance level for detecting outliers.
            min_nn_dist (int): Minimum distance required to consider points.
            scale_factor (int): Scaling factor for geo coordinates.

        Returns:
            tuple: Indices of detected outliers, p-values, and gamma parameters for geographic and genetic outliers.

        Returns:
            tuple: Indices of detected outliers, p-values, and gamma parameters for geographic and genetic outliers.
        """

        # Initialize a dictionary to store time durations
        time_durations = {}

        # Step 1: Find optimal K for both genetic and geographic data
        start_time = time.time()
        optk_gen = self.find_optimal_k(
            gen_coords,
            geo_dist,
            (2, maxk),
            w_power,
            min_nn_dist,
            is_genetic=True,
            scale_factor=scale_factor,
        )
        optk_geo = self.find_optimal_k(
            geo_coords,
            gen_dist,
            (2, maxk),
            w_power,
            min_nn_dist,
            is_genetic=False,
            scale_factor=scale_factor,
        )
        end_time = time.time()
        time_durations["find_optimal_k"] = end_time - start_time
        self.logger.info(
            f"Optimal K for Genetic Regression: {optk_gen}, Time taken: {time_durations['find_optimal_k']} seconds"
        )
        self.logger.info(
            f"Optimal K for Geographic Regression: {optk_geo}, Time taken: {time_durations['find_optimal_k']} seconds"
        )

        # Step 2: Find KNN based on both distances
        start_time = time.time()
        knn_indices_geo = self.find_geo_knn(geo_dist, optk_geo, min_nn_dist)
        knn_indices_gen = self.find_gen_knn(gen_dist, optk_gen)
        end_time = time.time()
        time_durations["find_knn"] = end_time - start_time
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
        self.logger.info(
            f"Calculating D statistics, Time taken: {time_durations['calculate_statistic']} seconds"
        )

        # Log r-squared values for predictions
        r2_gen = calculate_r2_knn(predicted_gen_data, gen_coords)
        r2_geo = calculate_r2_knn(predicted_geo_data, geo_coords)
        self.logger.info(f"r-squared for genetic outlier detection: {r2_gen}")
        self.logger.info(f"r-squared for geographic outlier detection: {r2_geo}")

        # Step 5: Fit Gamma distribution and detect outliers
        start_time = time.time()
        outliers_gen, p_value_gen, gamma_params_gen = self.fit_gamma_mle(
            dgen, dgeo, sig_level
        )
        end_time = time.time()
        time_durations["fit_gamma_genetic"] = end_time - start_time
        self.logger.info(
            f"Fitting gamma distribution for genetic outliers, Time taken: {time_durations['fit_gamma_genetic']} seconds"
        )

        start_time = time.time()
        outliers_geo, p_value_geo, gamma_params_geo = self.fit_gamma_mle(
            dgeo, dgeo, sig_level
        )
        end_time = time.time()
        time_durations["fit_gamma_geographic"] = end_time - start_time
        self.logger.info(
            f"Fitting gamma distribution for geographic outliers, Time taken: {time_durations['fit_gamma_geographic']} seconds"
        )

        # Optional: Plotting Gamma distribution
        start_time = time.time()
        outdir = path.join(self.output_dir, "plots")
        for dtype, gamma_params, d_statistic in zip(
            ["genetic", "geographic"],
            [gamma_params_gen, gamma_params_geo],
            [dgen, dgeo],
        ):
            fn = path.join(outdir, f"{self.prefix}_gamma_{dtype}.png")
            self.plotting.plot_gamma_distribution(
                gamma_params[0],
                gamma_params[1],
                d_statistic,
                sig_level,
                fn,
                f"{dtype.capitalize()} Outlier Gamma Distribution",
            )
        end_time = time.time()
        time_durations["plotting_gamma_distribution"] = end_time - start_time
        self.logger.info(
            f"Plotting Gamma distributions, Time taken: {time_durations['plotting_gamma_distribution']} seconds"
        )

        # Return the results and the timing data
        return (
            outliers_geo,
            outliers_gen,
            (p_value_geo, p_value_gen),
            (gamma_params_geo, gamma_params_gen),
        )

    def composite_outlier_detection(self, sig_level=0.05, maxk=50, min_nn_dist=100):
        self.logger.info("Starting composite outlier detection...")

        dgen = self.genetic_data
        dgeo = self.geographic_data
        outliers = {}
        for idtype in ["composite"]:
            outliers["geographic"], outliers["genetic"] = self.multi_stage_outlier_knn(
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
