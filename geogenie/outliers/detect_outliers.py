import logging
import time
from os import path
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import gamma

from geogenie.plotting.plotting import PlotGenIE
from geogenie.utils.utils import geo_coords_is_valid

# Mean Earth radius in kilometers (IUGG).
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lonlat1, lonlat2):
    """Great-circle distance in kilometers between (longitude, latitude) points.

    Args:
        lonlat1 (np.ndarray): Array of shape (..., 2); column 0 = longitude, column 1 = latitude, in degrees.
        lonlat2 (np.ndarray): Array broadcastable against ``lonlat1`` with the same column convention.

    Returns:
        np.ndarray: Distances in kilometers with the broadcast shape of the inputs (without the last axis).
    """
    lonlat1 = np.asarray(lonlat1, dtype=float)
    lonlat2 = np.asarray(lonlat2, dtype=float)
    lon1, lat1 = np.radians(lonlat1[..., 0]), np.radians(lonlat1[..., 1])
    lon2, lat2 = np.radians(lonlat2[..., 0]), np.radians(lonlat2[..., 1])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def r_squared(predicted_data, actual_data):
    """Mean per-column coefficient of determination of predictions.

    R^2 is computed for each column as ``1 - SS_res / SS_tot`` over the rows where the prediction is finite, and averaged over the columns with non-zero variance (loci for genetic data, longitude/latitude for geographic data).

    Args:
        predicted_data (np.ndarray): Predicted values, shape (n_samples, n_features).
        actual_data (np.ndarray): Observed values, shape (n_samples, n_features).

    Returns:
        float: Mean R^2 over columns, or NaN if it cannot be computed.
    """
    predicted_data = np.asarray(predicted_data, dtype=float)
    actual_data = np.asarray(actual_data, dtype=float)
    rows = np.all(np.isfinite(predicted_data), axis=1)
    if rows.sum() < 2:
        return np.nan
    pred, act = predicted_data[rows], actual_data[rows]
    ss_res = np.sum((act - pred) ** 2, axis=0)
    ss_tot = np.sum((act - act.mean(axis=0)) ** 2, axis=0)
    cols = ss_tot > 0
    if not np.any(cols):
        return np.nan
    return float(np.mean(1.0 - ss_res[cols] / ss_tot[cols]))


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p-values (q-values).

    Args:
        p_values (np.ndarray): 1D array of p-values.

    Returns:
        np.ndarray: Adjusted p-values in the original order.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = p_values.size
    if m == 0:
        return p_values.copy()
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


class GeoGeneticOutlierDetector:
    """A class to detect outliers based on genomic SNPs and geographic coordinates.

    This class uses a K-nearest neighbors (KNN) approach to detect outliers based on the difference between predicted and actual data.

    Nearest neighbors are found exactly (full pairwise distances, processed in row chunks), with self excluded by index and ties broken by sample index. Results therefore do not depend on ``n_jobs`` or on the global random state. Geographic distances are great-circle distances in kilometers; ``geographic_data`` columns are (longitude, latitude) in degrees.

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

        # Row-wise great-circle distance (km) between each sample and its prediction.
        distances = haversine_km(geo_coords, pred_geo_coords) / scalar

        if self.verbose >= 2 or self.debug:
            msg = "Finished estimating Dgeo statistic."
            self.logger.debug(msg) if self.debug else self.logger.info(msg)
        return distances

    def calculate_statistic(
        self,
        predicted_data,
        actual_data,
        is_genetic,
        min_nn_dist,
        scale_factor,
        rescale=True,
    ):
        """Calculate the Dg or Dgeo statistic based on the difference between predicted and actual data.

        This method calculates the Dg or Dgeo statistic based on the mean squared error or geographic distance between predicted and actual data. Samples without a prediction (no valid neighbor) get NaN.

        Args:
            predicted_data (np.array): Predicted data from KNN.
            actual_data (np.array): Actual data.
            is_genetic (bool): Flag to determine if the calculation is for genetic data.
            min_nn_dist (float): The minimum distance (km) to consider between geographic points. Returned unchanged.
            scale_factor (float): Scaling factor for Dgeo (Dgeo = distance in km / scale_factor).
            rescale (bool): If True, increase ``scale_factor`` when Dgeo values are large (see ``rescale_statistic``). Use False to keep a common scale, e.g. when comparing sums of Dgeo between K values.

        Returns:
            tuple: Dg or Dgeo statistic for each sample, min_nn_dist, the (possibly increased) scale_factor, and the mean per-column R^2 of the predictions.
        """
        if is_genetic:
            # Dg calculation (mean squared error).
            d = np.mean((predicted_data - actual_data) ** 2, axis=1)
            eps = 1e-6
        else:
            # Dgeo calculation (geographical distance)
            d = self.calculate_dgeo(predicted_data, actual_data, scale_factor)
            eps = 1e-8

            if rescale:
                min_nn_dist, scale_factor, recalculate = self.rescale_statistic(
                    d, scale_factor, min_nn_dist
                )

                if recalculate:
                    # Recalculate Dgeo again with new scale.
                    d = self.calculate_dgeo(predicted_data, actual_data, scale_factor)

        zero_indices = np.isclose(d, 0)
        if np.any(zero_indices):
            d[zero_indices] = eps  # To avoid zeros in D statistic.

        pred_r2 = r_squared(predicted_data, actual_data)
        return d, min_nn_dist, scale_factor, pred_r2

    def rescale_statistic(self, Dgeo, s, orig_min_nn_dist, max_threshold=20):
        """Rescales the Dgeo array to avoid large values that might cause errors in maximum likelihood estimation.

        This method rescales the Dgeo array to avoid large values that might cause errors in maximum likelihood estimation. The gamma distribution is a scale family, so this changes the fitted scale but not the p-values.

        Args:
            Dgeo (np.ndarray): An array representing geographic distances or differences.
            s (float): A scalar value used in calculations.
            orig_min_nn_dist (float): Minimum nearest neighbor distance (km). It is a neighbor-search distance, independent of the Dgeo scale, and is returned unchanged.
            max_threshold (int): Maximum Dgeo value to trigger rescaling.

        Returns:
            tuple: minimum nearest neighbor distance, adjusted scalar value, and whether Dgeo must be recalculated.
        """
        min_nn_dist = orig_min_nn_dist
        recalculate = False
        maxD = np.nanmax(Dgeo) if np.any(np.isfinite(Dgeo)) else 0.0
        if maxD > max_threshold:
            recalculate = True
            tmps = 1
            while maxD > max_threshold:
                tmps *= 10
                maxD /= 10
            s *= tmps
        return min_nn_dist, s, recalculate

    def _exact_knn(self, coords, k, metric, min_dist=0.0, chunk_size=512):
        """Exact K-nearest neighbors with self excluded and a deterministic tie-break.

        Distances from each sample to all samples are computed in row chunks. The sample itself is excluded by index (not by position, so exact duplicates are handled), neighbors closer than ``min_dist`` are excluded, and the remaining candidates are ordered by (distance, index) with a stable sort.

        Args:
            coords (np.ndarray): Array of shape (n_samples, n_features). For ``metric="haversine"``, columns are (longitude, latitude) in degrees.
            k (int): Number of neighbors. Capped at n_samples - 1.
            metric (str): "euclidean" or "haversine" (great-circle distance in km).
            min_dist (float): Neighbors with distance < min_dist are excluded. Same units as the metric.
            chunk_size (int): Number of rows processed at once.

        Returns:
            tuple: (indices, distances), each of shape (n_samples, k). Indices are row positions in ``coords``. Where fewer than k valid neighbors exist, the remaining entries have index -1 and distance np.inf; these must be masked, never used to index.
        """
        coords = np.asarray(coords, dtype=float)
        n = coords.shape[0]
        k = int(max(0, min(k, n - 1)))
        indices = np.full((n, k), -1, dtype=int)
        distances = np.full((n, k), np.inf)
        if k == 0:
            return indices, distances

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            rows = np.arange(start, stop)
            if metric == "haversine":
                d = haversine_km(coords[rows, None, :], coords[None, :, :])
            elif metric == "euclidean":
                d = cdist(coords[rows], coords, metric="euclidean")
            else:
                msg = f"Unsupported metric for nearest neighbors: {metric}"
                self.logger.error(msg)
                raise ValueError(msg)

            d[np.arange(stop - start), rows] = np.inf  # Exclude self by index.
            if min_dist is not None and min_dist > 0:
                d[d < min_dist] = np.inf  # Exclude neighbors that are too close.

            # Stable sort: ties are broken by neighbor index.
            order = np.argsort(d, axis=1, kind="stable")[:, :k]
            dist = np.take_along_axis(d, order, axis=1)
            valid = np.isfinite(dist)
            indices[rows] = np.where(valid, order, -1)
            distances[rows] = dist
        return indices, distances

    def find_gen_knn(self, coords, k, scale_factor=None):
        """Find exact K-nearest neighbors in genotype space (Euclidean distance).

        Args:
            coords (np.array): Genetic data, shape (n_samples, n_loci).
            k (int): Number of neighbors (excluding the sample itself).
            scale_factor (float): Unused; kept for call compatibility.

        Returns:
            tuple: Indices (row positions in ``coords``) and Euclidean distances of the K nearest neighbors, each of shape (n_samples, k).
        """
        return self._exact_knn(coords, k, metric="euclidean")

    def find_geo_knn(self, coords, k, min_nn_dist):
        """Find exact K-nearest geographic neighbors, excluding neighbors closer than ``min_nn_dist``.

        Args:
            coords (np.array): Geographic coordinates, shape (n_samples, 2); column 0 = longitude, column 1 = latitude, in degrees.
            k (int): Number of neighbors (excluding the sample itself).
            min_nn_dist (float): Minimum great-circle distance in kilometers; closer neighbors are skipped and the next ones are used instead.

        Returns:
            tuple: Indices (row positions in ``coords``, or -1 where fewer than k neighbors are at least ``min_nn_dist`` away) and great-circle distances in km (np.inf for -1 entries), each of shape (n_samples, k).
        """
        geo_coords_is_valid(coords)
        return self._exact_knn(coords, k, metric="haversine", min_dist=min_nn_dist)

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
        max_k = min(max_k, len(gen_coords) - 1)
        if max_k < min_k:
            msg = f"Too few samples ({len(gen_coords)}) to search K in {klim}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Neighbors are ordered by (distance, index), so the K nearest
        # neighbors are the first K columns of the max_k neighbor graph.
        if is_genetic:
            # Genetic predictions from geographic neighbors.
            all_indices, all_distances = self.find_geo_knn(
                geo_coords, max_k, min_nn_dist
            )
            target = gen_coords
        else:
            # Geographic predictions from genetic neighbors.
            all_indices, all_distances = self.find_gen_knn(
                gen_coords, max_k, scale_factor
            )
            target = geo_coords

        all_D = []
        for k in range(min_k, max_k + 1):
            predictions = self.predict_coords_knn(
                target, all_distances[:, :k], all_indices[:, :k], w_power
            )

            # Same scale for every K so the sums are comparable.
            D_statistic, _, _, _ = self.calculate_statistic(
                predictions,
                target,
                is_genetic,
                min_nn_dist,
                scale_factor,
                rescale=False,
            )

            all_D.append(np.nansum(D_statistic))
        optimal_k = min_k + int(np.argmin(all_D))  # Adjust index to actual K value

        if self.verbose >= 1:
            self.logger.info("Completed optimal K search.")

        return optimal_k

    def predict_coords_knn(self, coords, knn_distances, knn_indices, w_power):
        """Predict coordinates data using weighted KNN.

        This method predicts coordinates data using weighted KNN based on the distances to the K-nearest neighbors.

        Args:
            coords (np.array): Array of genetic or geographic coordinates.
            knn_distances (np.array): Distances to K-nearest neighbors.
            knn_indices (np.array): Indices of K-nearest neighbors. Entries < 0 mark missing neighbors and are ignored.
            w_power (float): Power of distance weight in prediction.

        Returns:
            np.array: Predicted coordinates using weighted KNN. Rows without any valid neighbor are NaN.
        """
        predictions = np.full(np.shape(coords), np.nan, dtype=float)

        for i in range(len(coords)):
            valid = knn_indices[i] >= 0
            if not np.any(valid):
                continue
            neighbor_indices = knn_indices[i][valid]
            neighbor_distances = knn_distances[i][valid]

            zero = neighbor_distances <= 0
            if np.any(zero):
                # Limit of inverse-distance weights as d -> 0: equal weight
                # on the zero-distance neighbors only.
                weights = zero.astype(float)
            else:
                # Compute weights inversely proportional to the distance
                weights = 1 / neighbor_distances**w_power

            # Calculate the weighted average of the neighbors
            predictions[i] = np.average(
                coords[neighbor_indices], axis=0, weights=weights
            )

        return predictions

    def fit_gamma_mle(self, D_statistic, sig_level, p_adjust=None):
        """Detect outliers using a Gamma distribution fitted to the Dg or Dgeo statistic.

        This method detects outliers using a Gamma distribution fitted to the Dg or Dgeo statistic. Non-finite values (samples without a prediction) are left out of the fit, get a NaN p-value and are never flagged.

        Args:
            D_statistic (np.array): Dg or Dgeo statistic for each sample.
            sig_level (float): Significance level for detecting outliers.
            p_adjust (str | None): Multiple-testing correction applied before comparing with ``sig_level``. None (default): raw p-values, each sample tested at ``sig_level``. "fdr_bh": Benjamini-Hochberg adjusted p-values.

        Returns:
            tuple: Indices of outliers, (unadjusted) p-values, and fitted Gamma parameters (shape, scale).
        """
        if p_adjust not in (None, "none", "fdr_bh"):
            msg = f"Invalid 'p_adjust' parameter provided: {p_adjust}"
            self.logger.error(msg)
            raise ValueError(msg)

        finite = np.isfinite(D_statistic)
        d_fit = D_statistic[finite]
        initial_shape = (np.mean(d_fit) ** 2) / np.var(d_fit)
        initial_rate = np.mean(d_fit) / np.var(d_fit)

        result = minimize(
            self.gamma_neg_log_likelihood,
            x0=(initial_shape, initial_rate),
            args=(d_fit,),
            bounds=[(1e-8, 1e8), (1e-8, 1e8)],
            method="L-BFGS-B",
        )
        shape, rate = result.x
        scale = 1 / rate  # Convert rate back to scale for gamma distribution
        p_values = np.full(len(D_statistic), np.nan)
        p_values[finite] = gamma.sf(d_fit, a=shape, scale=scale)

        p_test = p_values.copy()
        if p_adjust == "fdr_bh":
            p_test[finite] = benjamini_hochberg(p_values[finite])
        outliers = np.where(p_test < sig_level)[0]
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
        p_adjust=None,
    ):
        """Iterative Outlier Detection via KNN for genetic and geographic data.

        This method performs iterative outlier detection using KNN for genetic and geographic data. In each iteration the test is run on the samples not yet flagged, and newly flagged samples are added; iteration stops when an iteration flags no new samples.

        Args:
            geo_coords (np.array): Array of geographic coordinates; column 0 = longitude, column 1 = latitude, in degrees.
            gen_coords (np.array): Array of genetic data coordinates.
            analysis_type (str): Type of analysis to perform (genetic, geographic, composite).
            sig_level (float): Significance level for detecting outliers.
            maxk (int): Maximum number of nearest neighbors to consider.
            w_power (float): Power of distance weight in KNN prediction.
            min_nn_dist (float): Minimum distance in meters between a sample and its geographic neighbors (genetic test). Converted to km internally.
            scale_factor (float): Dgeo is the prediction error in km divided by this factor.
            p_adjust (str | None): Multiple-testing correction (None or "fdr_bh"). Default None: raw p-values.

        Returns:
            tuple: Indices of detected geographic outliers, indices of detected genetic outliers (row positions in ``geo_coords``/``gen_coords``; empty for a test that was not run), and the results dictionary.
        """

        if len(geo_coords) != len(gen_coords):
            msg = f"geographic and genetic coordinates must have the same number of samples: {len(geo_coords)}, {len(gen_coords)}"
            self.logger.error(msg)
            raise ValueError(msg)

        if analysis_type == "composite":
            analysis_types = ["genetic", "geographic"]
        elif analysis_type in ["genetic", "geographic"]:
            analysis_types = [analysis_type]
        else:
            msg = f"Invalid 'analysis_type' parameter provided: {analysis_type}"
            self.logger.error(msg)
            raise ValueError(msg)

        max_iter = len(geo_coords) // 2
        self.orig_min_nn_dist = min_nn_dist
        self.orig_scale_factor = scale_factor

        # min_nn_dist is given in meters; neighbor distances are in km.
        min_nn_dist = min_nn_dist / 1000.0

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
            for at in analysis_types:
                self.logger.info(
                    f"Optimal K for {at.capitalize()} Regression: {opt_ks[at]}, Time taken: {time_durations[f'find_optimal_k_{at}']} seconds"
                )

        res = {}
        for at in analysis_types:
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
                p_adjust=p_adjust,
            )

        def _flagged(at):
            if at not in res:
                return np.array([], dtype=int)
            return np.where(res[at]["outlier_flags"])[0]

        # Return the indices of the detected outliers
        return _flagged("geographic"), _flagged("genetic"), res

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
        p_adjust=None,
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
            p_adjust (str | None): Multiple-testing correction (None or "fdr_bh").

        Returns:
            dict: ``res`` with ``res[at]`` set. ``outlier_flags`` is aligned with the input rows; ``d_stats`` and ``p_values`` are from the final iteration and refer to the samples not flagged.
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
            p_adjust=p_adjust,
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
        p_adjust=None,
    ):
        """Run multi-stage outlier detection using KNN.

        This method runs multi-stage outlier detection using KNN for genetic and geographic data. Each iteration re-runs the test on the samples that have not been flagged yet (they are also removed as neighbors and from the gamma fit). Iteration stops when an iteration flags no new samples, or after ``max_iter`` iterations.

        Args:
            geo_coords (np.array): Array of geographic coordinates.
            gen_coords (np.array): Array of genetic data coordinates.
            sig_level (float): Significance level for detecting outliers.
            min_nn_dist (float): Minimum distance (km) required to consider points.
            scale_factor (int): Scaling factor for geo coordinates.
            max_iter (int): Maximum number of iterations.
            optk (int): Optimal K value for nearest neighbors.
            analysis_type (str): Type of analysis to perform (genetic, geographic).
            time_durations (dict): Dictionary to store run times for each method.
            p_adjust (str | None): Multiple-testing correction (None or "fdr_bh").

        Returns:
            tuple: Time durations, outlier flags (aligned with the input rows), D statistics, p-values, r-squared values, and gamma parameters. D statistics and p-values are from the final iteration.
        """

        outlier_flags = np.zeros(len(gen_coords), dtype=bool)

        iteration = 0
        while iteration < max_iter:
            iteration += 1

            if self.verbose >= 1:
                self.logger.info(
                    f"Iteration {iteration} of multi-stage {analysis_type} outlier detection."
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
                p_adjust=p_adjust,
            )

            if current_outliers is None or current_outliers.size == 0:
                if self.verbose >= 1:
                    self.logger.info(
                        f"No new {analysis_type} outliers detected in iteration {iteration}. Terminating iteration with {int(outlier_flags.sum())} {analysis_type} outliers in total."
                    )
                break

            # Map indices from filtered to original
            actual_outlier_indices = filtered_indices[current_outliers]

            # Update the flags in the original arrays
            outlier_flags[actual_outlier_indices] = True

            if self.verbose >= 1:
                self.logger.info(
                    f"Iteration {iteration}: flagged {len(actual_outlier_indices)} new {analysis_type} outliers ({int(outlier_flags.sum())} in total)."
                )

            if (~outlier_flags).sum() <= optk + 1:
                self.logger.warning(
                    f"Too few samples remain for {analysis_type} outlier detection with K={optk}. Terminating iteration with {int(outlier_flags.sum())} {analysis_type} outliers in total."
                )
                break
        else:
            if self.verbose >= 1:
                self.logger.info(
                    f"Reached the maximum of {max_iter} iterations with {int(outlier_flags.sum())} {analysis_type} outliers in total."
                )

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
        p_adjust=None,
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
            analysis_type (str): Type of analysis to perform (genetic, geographic).
            p_adjust (str | None): Multiple-testing correction (None or "fdr_bh").

        Returns:
            tuple: Time durations, indices of new outliers (positions in the filtered arrays), D statistics, p-values, r-squared values, gamma parameters, and filtered indices (positions of the filtered samples in the input arrays)
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
            p_adjust=p_adjust,
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
            d_stats[np.isfinite(d_stats)],
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
        p_adjust=None,
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
            min_nn_dist (float): Minimum distance (km) between a sample and its geographic neighbors (genetic test only).
            scale_factor (int): Scaling factor for geo coordinates.
            analysis_type (str): Either 'genetic' or 'geographic'.
            p_adjust (str | None): Multiple-testing correction (None or "fdr_bh").

        Returns:
            tuple: Time durations, indices of detected outliers (row positions in the given arrays), D statistics, p-values, r-squared value, and gamma parameters.
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
            key = f"find_knn_{analysis_type}"
            msg = f"Found KNN indices in: {time_durations[key]} seconds"
            self.logger.debug(msg) if self.debug else self.logger.info(msg)

        # Step 3: Predict using weighted KNN
        start_time = time.time()
        predictions = self.predict_coords_knn(
            dependent_data, dist, knn_indices, w_power
        )
        end_time = time.time()
        time_durations[f"predict_knn_{analysis_type}"] = end_time - start_time

        if self.verbose >= 2 or self.debug:
            key = f"predict_knn_{analysis_type}"
            msg = f"Predicted using weighted KNN: {time_durations[key]} seconds"
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
            key = f"calculate_statistic_{analysis_type}"
            msg = f"Calculated D statistics in: {time_durations[key]} seconds"
            self.logger.debug(msg) if self.debug else self.logger.info(msg)

        if self.verbose >= 1:
            self.logger.info(
                f"Mean per-column R-squared of KNN predictions for {analysis_type} outlier detection: {r2_vals}"
            )

        # Step 5: Fit Gamma distribution and detect outliers
        start_time = time.time()
        outliers, p_values, gamma_params = self.fit_gamma_mle(
            d, sig_level, p_adjust=p_adjust
        )
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
        self,
        sig_level=0.05,
        maxk=50,
        min_nn_dist=1000,
        scale_factor=None,
        w_power=2,
        p_adjust=None,
    ):
        """Perform composite outlier detection using the KNN approach.

        This method performs composite outlier detection using the KNN approach.

        Args:
            sig_level (float): Significance level for detecting outliers.
            maxk (int): Maximum number of nearest neighbors to consider.
            min_nn_dist (int): Minimum distance in meters between a sample and its geographic neighbors (CLI ``--min_nn_dist``).
            scale_factor (int | None): Scaling factor for Dgeo. If None, ``args.scale_factor`` is used (CLI ``--scale_factor``), or 100 if not set.
            w_power (float): Power of distance weight in KNN prediction.
            p_adjust (str | None): Multiple-testing correction. None (default): each sample is tested at ``sig_level`` with raw p-values. "fdr_bh": Benjamini-Hochberg.

        Returns:
            dict: Detected outliers, ``{"geographic": idx, "genetic": idx}``. Each ``idx`` is an integer array of row positions in the ``genetic_data``/``geographic_data`` passed to the constructor (not index labels); callers map them back to sample IDs.
        """
        if scale_factor is None:
            scale_factor = getattr(getattr(self, "args", None), "scale_factor", None)
            if scale_factor is None:
                scale_factor = 100

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
            p_adjust=p_adjust,
        )

        if self.verbose >= 1:
            self.logger.info("Outlier detection completed.")
        return outliers
