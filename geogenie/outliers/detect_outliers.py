import logging
import time
from os import path
from pathlib import Path

import numpy as np
from pynndescent import NNDescent
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import gamma

from geogenie.plotting.plotting import PlotGenIE
from geogenie.utils.scorers import calculate_r2_knn, haversine_distance
from geogenie.utils.utils import geo_coords_is_valid


class GeoGeneticOutlierDetector:
    """A class to detect outliers based on genomic SNPs and geographic coordinates.

    This class uses a K-nearest neighbors (KNN) approach to detect outliers based on the difference between predicted and actual data.

    Attributes:
        args (argparse.Namespace): Command line arguments.
        genetic_data (pd.DataFrame): The SNP data as a DataFrame.
        geographic_data (np.array): geographic coordinates as 2D array.
        output_dir (str): Output directory.
        prefix (str): Prefix for output files.
        n_jobs (int): Number of parallel jobs to run.
        seed (int): Random seed to use.
        url (str): url to download base map for plots.
        buffer (float): Buffer to put around sampling area on base map when plotting.
        show_plots (bool): Whether to show plots in-line.
        debug (bool): If True, writes genetic_data and geographic_data to file.
        verbose (int): Verbosity setting (0-2), least to most verbose.
        logger (logging.Logger): Logger object.
        plotting (PlotGenIE): Plotting object.
    """

    def __init__(
        self,
        args,
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

        This class uses a K-nearest neighbors (KNN) approach to detect outliers based on the difference between predicted and actual data.

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
            data_dir = Path(output_dir, "data")
            fn = data_dir / "debug_eigen.csv"
            genetic_data.to_csv(fn, header=False, index=False)
            tmp = geographic_data.reset_index()
            tmp.columns = ["sampleID", "x", "y"]

            fn2 = data_dir / "train_test_coords.csv"
            tmp.to_csv(fn2, header=True, index=False)

        self.genetic_data = genetic_data.to_numpy()
        self.geographic_data = geographic_data.to_numpy()

        geo_coords_is_valid(self.geographic_data)

        self.args = args
        self.output_dir = output_dir
        self.prefix = prefix
        self.n_jobs = n_jobs
        self.seed = seed
        self.url = url
        self.buffer = buffer
        self.show_plots = show_plots
        self.debug = debug
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        self.plotting = PlotGenIE(
            "cpu",
            self.args.output_dir,
            self.args.prefix,
            self.args.basemap_fips,
            self.args.highlight_basemap_counties,
            self.args.shapefile,
            show_plots=False,
            fontsize=self.args.fontsize,
            filetype=self.args.filetype,
            dpi=self.args.plot_dpi,
            remove_splines=self.args.remove_splines,
        )

        # Ensure correct coordinate projection and CRS.
        df_geo = self.plotting.processor.to_geopandas(self.geographic_data)
        self.geographic_data = self.plotting.processor.to_numpy(df_geo)
        geo_coords_is_valid(self.geographic_data)

    def calculate_dgeo(self, pred_geo_coords, geo_coords, scalar):
        """Calculate the Dgeo statistic for geographic coordinates.

        This method calculates the geographic distance between predicted and actual geographic coordinates.

        Args:
            pred_geo_coords (np.array): Predicted geographic coordinates.
            geo_coords (np.array): Actual geographic coordinates.
            scalar (float): Scalar value to divide the distance.

        Returns:
            np.array: Calculated Dgeo values.
        """
        if self.verbose >= 2 or self.debug:
            msg = "Estimating Dgeo statistic..."
            self.logger.debug(msg) if self.debug else self.logger.info(msg)

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

        if self.verbose >= 2 or self.debug:
            msg = "Finished estimating Dgeo statistic."
            self.logger.debug(msg) if self.debug else self.logger.info(msg)
        return distances

    def calculate_statistic(
        self, predicted_data, actual_data, is_genetic, min_nn_dist, scale_factor
    ):
        """Calculate the Dg or Dgeo statistic based on the difference between predicted and actual data.

        This method calculates the Dg or Dgeo statistic based on the mean squared error or geographic distance between predicted and actual data.

        Args:
            predicted_data (np.array): Predicted data from KNN.
            actual_data (np.array): Actual data.
            is_genetic (bool): Flag to determine if the calculation is for genetic data.
            min_nn_dist (float): The minimum distance to consider between geographic points.
            scale_factor (float): Scaling factor for geo_coords.

        Returns:
            np.array: Dg or Dgeo statistic for each sample.
        """
        if is_genetic:
            # Dg calculation (mean squared error).
            d = np.mean((predicted_data - actual_data) ** 2, axis=1)
            eps = 1e-6
        else:
            # Dgeo calculation (geographical distance)
            d = self.calculate_dgeo(predicted_data, actual_data, scale_factor)
            eps = 1e-8
            min_nn_dist, scale_factor, recalculate = self.rescale_statistic(
                d, scale_factor, min_nn_dist
            )

            if recalculate:
                # Recalculate Dgeo again with new scale.
                d = self.calculate_dgeo(predicted_data, actual_data, scale_factor)

        zero_indices = np.isclose(d, 0)
        if np.any(zero_indices):
            d[zero_indices] = eps  # To avoid zeros in D statistic.

        pred_r2 = calculate_r2_knn(predicted_data, actual_data)
        return d, min_nn_dist, scale_factor, pred_r2

    def rescale_statistic(self, Dgeo, s, orig_min_nn_dist, max_threshold=20):
        """Rescales the Dgeo array to avoid large values that might cause errors in maximum likelihood estimation.

        This method rescales the Dgeo array to avoid large values that might cause errors in maximum likelihood estimation.

        Args:
            Dgeo (np.ndarray): An array representing geographic distances or differences.
            s (float): A scalar value used in calculations.
            orig_min_nn_dist (float): Original minimum nearest neighbor distance.
            max_threshold (int): Maximum Dgeo value to trigger rescaling.

        Returns:
            float, float: adjusted scalar value, and adjusted minimum nearest neighbor distance.
        """
        min_nn_dist = orig_min_nn_dist
        recalculate = False
        maxD = np.max(Dgeo)
        if maxD > max_threshold:
            recalculate = True
            tmps = 1
            while maxD > 20:
                tmps *= 10
                maxD /= 10
            s *= tmps

            # new min_nn_dist adjusted according to new s
            min_nn_dist = orig_min_nn_dist / s
        return min_nn_dist, s, recalculate

    def find_gen_knn(self, coords, k, scale_factor):
        """Find K-nearest neighbors for genetic data using PyNNDescent.

        This method finds the K-nearest neighbors for genetic data using PyNNDescent.

        Args:
            dist_matrix (np.array): Distance matrix.
            k (int): Number of neighbors.

        Returns:
            np.array: Indices of K-nearest neighbors.
        """

        nnd = NNDescent(
            coords,
            metric="euclidean",
            n_neighbors=k,
            n_jobs=self.n_jobs,
        )
        indices, distances = nnd.neighbor_graph

        # Exclude the first column which is the point itself
        return indices[:, 1 : k + 1], distances[:, 1 : k + 1]

    def find_geo_knn(self, coords, k, min_nn_dist):
        """Find K-nearest neighbors for geographic data considering minimum neighbor distance using PyNNDescent.

        This method finds the K-nearest neighbors for geographic data using PyNNDescent.

        Args:
            dist_matrix (np.array): Distance matrix.
            k (int): Number of neighbors.
            min_nn_dist (float): Minimum distance to consider for neighbors.

        Returns:
            np.array: Indices of K-nearest neighbors.
        """
        geo_coords_is_valid(coords)

        # PyNNDescent takes coordinates ordered: lat, lon
        geo_coords = np.flip(coords)

        # Initialize NNDescent with the precomputed distance matrix
        nnd = NNDescent(
            geo_coords,
            metric="haversine",
            n_neighbors=k,
            n_jobs=self.n_jobs,
        )
        indices, distances = nnd.neighbor_graph

        # Exclude neighbors that are too close (less than min_nn_dist)
        valid_indices = distances >= min_nn_dist
        k_indices = np.where(valid_indices, indices, -1)

        # Truncate or pad the result_indices to ensure exactly k neighbors
        # (excluding the point itself which is at index 0)
        return k_indices[:, 1 : k + 1], distances[:, 1 : k + 1]

    def find_optimal_k(
        self,
        geo_coords,
        gen_coords,
        klim,
        w_power,
        min_nn_dist,
        is_genetic,
        scale_factor,
    ):
        """Find optimal number of nearest neighbors for KNN.

        This method finds the optimal number of nearest neighbors for KNN based on the Dg or Dgeo statistic.

        Args:
            geo_coords (np.array): Geographic coordinates.
            gen_coords (np.array): Genetic coordinatees.
            dist_matrix (np.array): Distance matrix.
            klim (tuple): Range of K values to search (min_k, max_k).
            w_power (float): Power for distance weighting.
            min_nn_dist (int): Minimum nearest neighbor distance to consider points.
            is_genetic (bool): Flag to determine if the calculation is for genetic data as distance matrix.
            scale_factor (float): Factor to scale geo_coords by.

        Returns:
            int: Optimal K value.
        """
        if self.verbose >= 1:
            self.logger.info("Finding optimal K for Nearest Neighbors...")

        min_k, max_k = klim
        all_D = []
        for k in range(min_k, max_k + 1):
            if is_genetic:
                # Genetic coords, geo dist matrix.
                knn_indices, distances = self.find_geo_knn(geo_coords, k, min_nn_dist)

                # Genetic predictions.
                predictions = self.predict_coords_knn(
                    gen_coords, distances, knn_indices, w_power
                )

                # Genetic error.
                (
                    D_statistic,
                    min_nn_dist,
                    scale_factor,
                    _,
                ) = self.calculate_statistic(
                    predictions, gen_coords, is_genetic, min_nn_dist, scale_factor
                )
            else:
                # Geo coords, genetic dist matrix.
                knn_indices, distances = self.find_gen_knn(gen_coords, k, scale_factor)

                # Geographic predictions.
                predictions = self.predict_coords_knn(
                    geo_coords, distances, knn_indices, w_power
                )

                # Geographic error.
                (
                    D_statistic,
                    min_nn_dist,
                    scale_factor,
                    _,
                ) = self.calculate_statistic(
                    predictions, geo_coords, is_genetic, min_nn_dist, scale_factor
                )

            all_D.append(np.sum(D_statistic))
        optimal_k = min_k + np.argmin(all_D)  # Adjust index to actual K value

        if self.verbose >= 1:
            self.logger.info("Completed optimal K search.")

        return optimal_k

    def predict_coords_knn(self, coords, knn_distances, knn_indices, w_power):
        """Predict coordinates data using weighted KNN.

        This method predicts coordinates data using weighted KNN based on the distances to the K-nearest neighbors.

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

    def fit_gamma_mle(self, D_statistic, sig_level):
        """Detect outliers using a Gamma distribution fitted to the Dg or Dgeo statistic.

        This method detects outliers using a Gamma distribution fitted to the Dg or Dgeo statistic.

        Args:
            D_statistic (np.array): Dg or Dgeo statistic for each sample.
            Dgeo (np.array): For determining initial_shape and initial_rate.
            sig_level (float): Significance level for detecting outliers.

        Returns:
            tuple: Indices of outliers, p-values, and fitted Gamma parameters.
        """
        initial_shape = (np.mean(D_statistic) ** 2) / np.var(D_statistic)
        initial_rate = np.mean(D_statistic) / np.var(D_statistic)

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
        """Negative log likelihood for gamma distribution.

        This method calculates the negative log likelihood value for the gamma distribution.

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
        analysis_type="composite",
        sig_level=0.05,
        maxk=50,
        w_power=2,
        min_nn_dist=1000,
        scale_factor=100,
    ):
        """Iterative Outlier Detection via KNN for genetic and geographic data.

        This method performs iterative outlier detection using KNN for genetic and geographic data.

        Args:
            geo_coords (np.array): Array of geographic coordinates.
            gen_coords (np.array): Array of genetic data coordinates.
            analysis_type (str): Type of analysis to perform (genetic, geographic, composite).
            sig_level (float): Significance level for detecting outliers.
            maxk (int): Maximum number of nearest neighbors to consider.
            w_power (float): Power of distance weight in KNN prediction.
            min_nn_dist (int): Minimum distance required to consider points.
            scale_factor (int): Scaling factor for geo coordinates.

        Returns:
            tuple: Indices of detected outliers for geographic and genetic data.
        """

        if len(geo_coords) != len(gen_coords):
            msg = f"geographic and genetic coordinates must have the same number of samples: {len(geo_coords)}, {len(gen_coords)}"
            self.logger.error(msg)
            raise ValueError(msg)

        max_iter = len(geo_coords) // 2
        self.orig_min_nn_dist = min_nn_dist
        self.orig_scale_factor = scale_factor
        min_nn_dist /= scale_factor

        time_durations, optk_gen, optk_geo = self.search_nn_optk(
            geo_coords,
            gen_coords,
            maxk,
            w_power,
            min_nn_dist,
            scale_factor,
            analysis_type,
        )

        opt_ks = {"genetic": optk_gen, "geographic": optk_geo}

        if self.verbose >= 1:
            self.logger.info(
                f"Optimal K for Genetic Regression: {optk_gen}, Time taken: {time_durations['find_optimal_k_genetic']} seconds"
            )
            self.logger.info(
                f"Optimal K for Geographic Regression: {optk_geo}, Time taken: {time_durations['find_optimal_k_geographic']} seconds"
            )

        if analysis_type == "composite":
            res = {}
            for at in ["genetic", "geographic"]:
                res = self.analysis(
                    geo_coords,
                    gen_coords,
                    at,
                    sig_level,
                    min_nn_dist,
                    scale_factor,
                    max_iter,
                    opt_ks,
                    res,
                    at,
                    time_durations,
                )
        elif analysis_type == "genetic":
            at = "genetic"
            res = self.analysis(
                geo_coords,
                gen_coords,
                analysis_type,
                sig_level,
                min_nn_dist,
                scale_factor,
                max_iter,
                opt_ks,
                res,
                at,
                time_durations,
            )
        elif analysis_type == "geographic":
            res = self.analysis(
                geo_coords,
                gen_coords,
                analysis_type,
                sig_level,
                min_nn_dist,
                scale_factor,
                max_iter,
                opt_ks,
                res,
                at,
                time_durations,
            )

        # Return the indices of the detected outliers
        return (
            np.where(res["geographic"]["outlier_flags"])[0],
            np.where(res["genetic"]["outlier_flags"])[0],
            res,
        )

    def analysis(
        self,
        geo_coords,
        gen_coords,
        analysis_type,
        sig_level,
        min_nn_dist,
        scale_factor,
        max_iter,
        opt_ks,
        res,
        at,
        time_duration,
    ):
        """Perform outlier detection analysis for genetic or geographic data.

        This method performs outlier detection analysis for genetic or geographic data.

        Args:
            geo_coords (np.array): Array of geographic coordinates.
            gen_coords (np.array): Array of genetic data coordinates.
            analysis_type (str): Type of analysis to perform (genetic, geographic, composite).
            sig_level (float): Significance level for detecting outliers.
            min_nn_dist (int): Minimum distance required to consider points.
            scale_factor (int): Scaling factor for geo coordinates.
            max_iter (int): Maximum number of iterations.
            opt_ks (dict): Optimal K values for genetic and geographic data.
            res (dict): Dictionary to store results.
            at (str): Analysis type (genetic, geographic).
            time_duration (dict): Dictionary to store run times for each method.
        """
        (
            time_duration,
            outlier_flags,
            d_stats,
            p_values,
            r2_values,
            gamma_params,
        ) = self.run_multistage(
            geo_coords,
            gen_coords,
            sig_level,
            min_nn_dist,
            scale_factor,
            max_iter,
            opt_ks[at],
            analysis_type,
            time_duration,
        )

        if self.verbose >= 1:
            self.logger.info("Completed multi-stage outlier detection.")

            # Plotting Gamma distribution
        start_time = time.time()

        self.plot_gamma_dist(sig_level, d_stats, gamma_params, at)

        end_time = time.time()
        time_duration[f"plot_gamma_distribution_{at}"] = end_time - start_time

        if self.verbose >= 2 or self.debug:
            key = f"plot_gamma_distribution_{at}"
            msg = f"Plotted Gamma distributions in {time_duration[key]} seconds"
            self.logger.debug(msg) if self.debug else self.logger.info(msg)

        res[at] = {
            "time_duration": time_duration,
            "outlier_flags": outlier_flags,
            "d_stats": d_stats,
            "p_values": p_values,
            "r2_values": r2_values,
            "gamma_params": gamma_params,
        }

        return res

    def run_multistage(
        self,
        geo_coords,
        gen_coords,
        sig_level,
        min_nn_dist,
        scale_factor,
        max_iter,
        optk,
        analysis_type,
        time_durations,
    ):
        """Run multi-stage outlier detection using KNN.

        This method runs multi-stage outlier detection using KNN for genetic and geographic data.

        Args:
            geo_coords (np.array): Array of geographic coordinates.
            gen_coords (np.array): Array of genetic data coordinates.
            sig_level (float): Significance level for detecting outliers.
            min_nn_dist (int): Minimum distance required to consider points.
            scale_factor (int): Scaling factor for geo coordinates.
            max_iter (int): Maximum number of iterations.
            optk (int): Optimal K value for nearest neighbors.
            analysis_type (str): Type of analysis to perform (genetic, geographic, composite).
            time_durations (dict): Dictionary to store run times for each method.

        Returns:
            tuple: Time durations, outlier flags, D statistics, p-values, r-squared values, and gamma parameters
        """

        outlier_flags = np.zeros(len(gen_coords), dtype=bool)
        allow = True
        break_msg = "No new outliers detected. Terminating iteration."

        iteration = 0
        while iteration < max_iter:
            iteration += 1
            new_outliers_detected = False

            if self.verbose >= 1:
                self.logger.info(
                    f"Iteration {iteration} of multi-stage outlier detection.\n\n"
                )

            (
                time_durations,
                current_outliers,
                d_stats,
                p_values,
                r2_values,
                gamma_params,
                filtered_indices,
            ) = self.filter_and_detect(
                geo_coords,
                gen_coords,
                sig_level,
                min_nn_dist,
                scale_factor,
                optk,
                outlier_flags,
                time_durations,
                analysis_type,
            )

            # Map indices from filtered to original
            if current_outliers is not None and current_outliers.size > 0:
                if np.all(p_values > sig_level):
                    if self.verbose >= 1:
                        self.logger.info(break_msg)
                    break

                actual_outlier_indices = filtered_indices[current_outliers]

                # Update the flags in the original arrays
                outlier_flags[actual_outlier_indices] = True
                new_outliers_detected = True
            else:
                allow = False

            if not new_outliers_detected or not allow:
                if self.verbose >= 1:
                    self.logger.info(break_msg)
                break

            # No significant outliers.
            if np.all(p_values) > sig_level:
                if self.verbose >= 1:
                    self.logger.info(break_msg)
                break

            if self.verbose >= 2 or self.debug:
                msg = f"Finished iteration {iteration}."
                self.logger.debug(msg) if self.debug else self.logger.info(msg)

        return (
            time_durations,
            outlier_flags,
            d_stats,
            p_values,
            r2_values,
            gamma_params,
        )

    def filter_and_detect(
        self,
        geo_coords,
        gen_coords,
        sig_level,
        min_nn_dist,
        scale_factor,
        optk,
        outlier_flags,
        time_durations,
        analysis_type,
    ):
        """Filter outliers and detect new outliers using KNN.

        This method filters outliers and detects new outliers using KNN for genetic and geographic data.

        Args:
            geo_coords (np.array): Array of geographic coordinates.
            gen_coords (np.array): Array of genetic data coordinates.
            sig_level (float): Significance level for detecting outliers.
            min_nn_dist (int): Minimum distance required to consider points.
            scale_factor (int): Scaling factor for geo coordinates.
            optk (int): Optimal K value for nearest neighbors.
            outlier_flags (np.array): Array of outlier flags.
            time_durations (dict): Dictionary to store run times for each method.
            analysis_type (str): Type of analysis to perform (genetic, geographic, composite).

        Returns:
            tuple: Time durations, outlier flags, D statistics, p-values, r-squared values, gamma parameters, and filtered indices
        """
        non_outlier_mask = ~outlier_flags

        # Keep track of original indices
        original_indices = np.arange(len(geo_coords))
        filtered_indices = original_indices[non_outlier_mask]

        # Get only non-outliers.
        filtered_geo_coords = geo_coords[non_outlier_mask]
        filtered_gen_coords = gen_coords[non_outlier_mask]

        (
            time_durations,
            current_outliers,
            d_stats,
            p_values,
            r2_values,
            gamma_params,
        ) = self.detect_outliers(
            filtered_geo_coords,
            filtered_gen_coords,
            optk,
            time_durations,
            w_power=2,
            sig_level=sig_level,
            min_nn_dist=min_nn_dist,
            scale_factor=scale_factor,
            analysis_type=analysis_type,
        )

        return (
            time_durations,
            current_outliers,
            d_stats,
            p_values,
            r2_values,
            gamma_params,
            filtered_indices,
        )

    def search_nn_optk(
        self,
        geo_coords,
        gen_coords,
        maxk,
        w_power,
        min_nn_dist,
        scale_factor,
        analysis_type,
    ):
        """Search for optimal K for nearest neighbors.

        This method searches for the optimal K for nearest neighbors based on the Dg or Dgeo statistic.

        Args:
            geo_coords (np.array): Array of geographic coordinates.
            gen_coords (np.array): Array of genetic data coordinates.
            maxk (int): Maximum number of nearest neighbors to consider.
            w_power (float): Power of distance weight in KNN prediction.
            min_nn_dist (int): Minimum distance required to consider points.
            scale_factor (int): Scaling factor for geo coordinates.
            analysis_type (str): Type of analysis to perform (genetic, geographic, composite).

        Returns:
            tuple: Time durations, optimal K for genetic data, and optimal K for geographic data.
        """
        time_durations = {}

        optk_gen = None
        optk_geo = None

        if analysis_type in ["genetic", "composite"]:
            # Step 1: Find optimal K for both genetic and geographic data
            start_time = time.time()
            optk_gen = self.find_optimal_k(
                geo_coords,
                gen_coords,
                (2, maxk),
                w_power,
                min_nn_dist,
                is_genetic=True,
                scale_factor=scale_factor,
            )

        end_time = time.time()
        time_durations["find_optimal_k_genetic"] = end_time - start_time

        if analysis_type in ["geographic", "composite"]:
            start_time = time.time()
            optk_geo = self.find_optimal_k(
                geo_coords,
                gen_coords,
                (2, maxk),
                w_power,
                min_nn_dist,
                is_genetic=False,
                scale_factor=scale_factor,
            )
            end_time = time.time()
            time_durations["find_optimal_k_geographic"] = end_time - start_time
        return time_durations, optk_gen, optk_geo

    def plot_gamma_dist(self, sig_level, d_stats, gamma_params, dtype):
        """Plot the gamma distribution for detected outliers.

        This method plots the gamma distribution for detected outliers based on the Dg or Dgeo statistic.

        Args:
            sig_level (float): Significance level for detecting outliers.
            d_stats (np.array): Dg or Dgeo statistic for each sample.
            gamma_params (tuple): Parameters for the gamma distribution.
            dtype (str): Type of data (genetic, geographic
        """
        outdir = path.join(self.output_dir, "plots")
        fn = path.join(outdir, f"{self.prefix}_gamma_{dtype}.png")
        self.plotting.plot_gamma_distribution(
            gamma_params[0],
            gamma_params[1],
            d_stats,
            sig_level,
            fn,
            f"{dtype.capitalize()} Outlier Gamma Distribution",
        )

    def detect_outliers(
        self,
        geo_coords,
        gen_coords,
        optk,
        time_durations,
        w_power=2,
        sig_level=0.05,
        min_nn_dist=1000,
        scale_factor=100,
        analysis_type="genetic",
    ):
        """Detect outliers based on composite data using the KNN approach.

        This method detects outliers based on composite data using the KNN approach.

        Args:
            geo_coords (np.array): Array of geographic coordinates.
            gen_coords (np.array): Array of genetic data coordinates.
            optk (int): Optimal K for nearest neigbbors (geographic).
            time_durations (dict): Dictionary storing run times for each method.
            w_power (float): Power of distance weight in KNN prediction.
            sig_level (float): Significance level for detecting outliers.
            min_nn_dist (int): Minimum distance required to consider points.
            scale_factor (int): Scaling factor for geo coordinates.
            analysis_type (str): Either 'genetic' or 'geographic'.

        Returns:
            tuple: Indices of detected outliers, p-values, and gamma parameters for geographic and genetic outliers.

        Returns:
            tuple: Indices of detected outliers, p-values, and gamma parameters for geographic and genetic outliers.
        """
        if analysis_type == "genetic":
            dependent_data = gen_coords
            independent_data = geo_coords
            knn_func = self.find_geo_knn
            is_genetic = True
        elif analysis_type == "geographic":
            dependent_data = geo_coords
            independent_data = gen_coords
            knn_func = self.find_gen_knn
            is_genetic = False
        else:
            msg = f"Invalid 'analysis_type' parameter provided: {analysis_type}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Step 2: Find KNN based on distances
        start_time = time.time()
        knn_indices, dist = knn_func(independent_data, optk, min_nn_dist)
        end_time = time.time()
        time_durations[f"find_knn_{analysis_type}"] = end_time - start_time

        if self.verbose >= 2 or self.debug:
            msg = f"Found KNN indices in: {time_durations[key]} seconds"
            key = f"find_knn_{analysis_type}"
            self.logger.debug(msg) if self.debug else self.logger.info(msg)

        # Step 3: Predict using weighted KNN
        start_time = time.time()
        predictions = self.predict_coords_knn(
            dependent_data, dist, knn_indices, w_power
        )
        end_time = time.time()
        time_durations[f"predict_knn_{analysis_type}"] = end_time - start_time

        if self.verbose >= 2 or self.debug:
            msg = f"Predicted using weighted KNN: {time_durations[key]} seconds"
            key = f"predict_knn_{analysis_type}"
            self.logger.debug(msg) if self.debug else self.logger.info(msg)

        # Step 4: Calculate D statistics and detect outliers
        start_time = time.time()
        d, min_nn_dist, scale_factor, r2_vals = self.calculate_statistic(
            predictions,
            dependent_data,
            is_genetic=is_genetic,
            min_nn_dist=min_nn_dist,
            scale_factor=scale_factor,
        )
        end_time = time.time()
        time_durations[f"calculate_statistic_{analysis_type}"] = end_time - start_time

        if self.verbose >= 2 or self.debug:
            msg = f"Calculated D statistics in: {time_durations[key]} seconds"
            key = f"calculate_statistic_{analysis_type}"
            self.logger.debug(msg) if self.debug else self.logger.info(msg)

        if self.verbose >= 1:
            self.logger.info(
                f"r-squared for {analysis_type} outlier detection: {r2_vals}"
            )

        # Step 5: Fit Gamma distribution and detect outliers
        start_time = time.time()
        outliers, p_values, gamma_params = self.fit_gamma_mle(d, sig_level)
        end_time = time.time()
        time_durations[f"fit_gamma_{analysis_type}"] = end_time - start_time

        if self.verbose >= 2 or self.debug:
            key = f"fit_gamma_{analysis_type}"
            msg = f"Fitted gamma distribution for {analysis_type} outliers in: {time_durations[key]} seconds"
            self.logger.debug(msg) if self.debug else self.logger.info(msg)

        return (
            time_durations,
            outliers,
            d,
            p_values,
            r2_vals,
            gamma_params,
        )

    def composite_outlier_detection(
        self, sig_level=0.05, maxk=50, min_nn_dist=1000, scale_factor=100, w_power=2
    ):
        """Perform composite outlier detection using the KNN approach.

        This method performs composite outlier detection using the KNN approach.

        Args:
            sig_level (float): Significance level for detecting outliers.
            maxk (int): Maximum number of nearest neighbors to consider.
            min_nn_dist (int): Minimum distance required to consider points.
            scale_factor (int): Scaling factor for geo coordinates.
            w_power (float): Power of distance weight in KNN prediction.

        Returns:
            dict: Detected outliers for geographic and genetic data.
        """
        if self.verbose >= 1:
            self.logger.info("Starting composite outlier detection...")

        dgen = self.genetic_data
        dgeo = self.geographic_data
        outliers = {}
        (
            outliers["geographic"],
            outliers["genetic"],
            _,
        ) = self.multi_stage_outlier_knn(
            dgeo,
            dgen,
            analysis_type="composite",
            sig_level=sig_level,
            maxk=maxk,
            w_power=w_power,
            min_nn_dist=min_nn_dist,
            scale_factor=scale_factor,
        )

        if self.verbose >= 1:
            self.logger.info("Outlier detection completed.")
        return outliers
