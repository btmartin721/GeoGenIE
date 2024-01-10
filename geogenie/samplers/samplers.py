import logging
import os

import jenkspy
import numpy as np
import pandas as pd
import torch
from geopy.distance import great_circle
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from scipy import optimize
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Sampler

from geogenie.utils.scorers import haversine_distances_agg
from geogenie.utils.utils import assign_to_bins, geo_coords_is_valid, validate_is_numpy

logger = logging.getLogger(__name__)

os.environ["TQDM_DISABLE"] = "true"


def synthetic_resampling(
    features,
    labels,
    sample_weights,
    n_bins,
    args,
    method="kerneldensity",
    smote_neighbors=3,
    verbose=0,
):
    """
    Performs synthetic resampling on the provided datasets using various binning methods and SMOTE (Synthetic Minority Over-sampling Technique).

    Args:
        features (DataFrame): The feature set.
        labels (DataFrame): The label set.
        sample_weights (array): Array of sample weights.
        n_bins (int): The number of bins to use for resampling.
        args (Namespace): Arguments provided for resampling configurations.
        method (str, optional): The method to use for binning. Defaults to "kmeans".
        smote_neighbors (int, optional): The number of nearest neighbors to use in SMOTE. Defaults to 5.

    Raises:
        ValueError: If an invalid resampling method is provided.

    Returns:
        tuple: A tuple containing the resampled features, labels, sample weights, centroids, DataFrame with original data, bins before resampling, centroids before resampling, and bins after resampling.

    Notes:
        - The function supports different binning methods: 'kmeans', 'optics', and 'kerneldensity'.
                - For 'kerneldensity', spatial KDE is assumed to be pre-run.
        - The function includes several preprocessing steps like bin assignment, merging of small bins, scaling, and handling single-sample bins.
        - SMOTE is used to oversample minority classes in each bin.
        - EditedNearestNeighbors is used for undersampling majority classes in each bin.
    """
    if method not in ["kmeans", "optics", "kerneldensity"]:
        if method.lower() == "none":
            return None, None, None, None, None, None, None, None
        else:
            msg = f"Invalid 'method' parameter passed to 'synthetic_resampling()': {method}"
            logger.error(msg)
            raise ValueError(msg)

    # dfX will contain sample_weights.
    dfX, dfy = setup_synth_resampling(features, labels, sample_weights)
    feature_names = dfX.columns

    centroids = None
    df = pd.concat([dfX, dfy], axis=1)

    if method == "kerneldensity":
        # Instantiate the KDE model
        # Assuming you have already run the spatial_kde function
        bins, centroids = do_kde_binning(n_bins, verbose, dfy)

    else:
        # Binning with KMeans or OPTICS
        bins, centroids = assign_to_bins(
            df, "x", "y", n_bins, args, method=method, random_state=args.seed
        )

        X = df.to_numpy()
        y = bins.copy()

        # Exclude samples with DBSCAN label -1 (noise)
        # And do standard scaling.
        y_filtered, bins_filtered, ssc, X_scaled = process_bins(X, y, bins)

        # Map resampled bins back to longitude and latitude
        # Calculate OPTICS or KMeans cluster centroids
        unique_bins = np.unique(bins_filtered)

        if len(unique_bins) == 1:
            logger.warning("Number of unique bins was only 1.")
            return None, None, None, None

        if method != "kerneldensity":
            centroids = get_centroids(centroids, y_filtered, bins_filtered, unique_bins)

        # Perform SMOTE on filtered data
        (
            features_resampled,
            labels_resampled,
            sample_weights_resampled,
            centroids_resampled,
            bins_resampled,
        ) = run_binned_smote(
            args,
            feature_names,
            ssc,
            centroids,
            bins_filtered,
            X_scaled,
            smote_neighbors,
            method,
            labels,
        )

    if all(
        [
            x is None
            for x in [
                features_resampled,
                labels_resampled,
                sample_weights_resampled,
                centroids_resampled,
                bins_resampled,
            ]
        ]
    ):
        return None, None, None, None, None, None, None, None

    return (
        features_resampled,
        labels_resampled,
        sample_weights_resampled,
        centroids_resampled,
        df,
        bins,
        centroids,
        bins_resampled,
    )


def do_kde_binning(n_bins, verbose, dfy):
    density, lon_grid, lat_grid = spatial_kde(
        dfy["x"].to_numpy(), dfy["y"].to_numpy(), bandwidth=0.04
    )

    # Define thresholds for bins
    # thresholds = define_density_thresholds(density, n_bins)
    thresholds = define_jenks_thresholds(density, n_bins)

    centroids = calculate_bin_centers(density, lon_grid, lat_grid, thresholds)

    # Assign samples to bins
    samples = np.column_stack([dfy["x"].to_numpy(), dfy["y"].to_numpy()])

    bins = assign_samples_to_bins(samples, density, lon_grid, lat_grid, thresholds)

    if len(np.unique(bins)) == 1:
        msg = "Only one bin detected with 'kerneldensity' method."
        logger.error(msg)
        raise ValueError(msg)

    small_bins = identify_small_bins(bins, n_bins, min_smote_neighbors=5)
    distance_matrix = calculate_centroid_distances(centroids)
    bins = merge_small_bins(small_bins, distance_matrix, bins)

    # Check for single-sample bins after merging
    bin_counts_after_merging = np.bincount(bins, minlength=n_bins)
    single_sample_bins = np.where(bin_counts_after_merging == 1)[0]

    if single_sample_bins.size > 0:
        if verbose >= 1:
            logger.warning(f"Single-sample bins found: {single_sample_bins}")
        # Merge single-sample bins with nearest bin
        bins = merge_single_sample_bins(
            single_sample_bins, distance_matrix, bins, verbose=verbose
        )

    return bins, centroids


def merge_single_sample_bins(
    single_sample_bins, distance_matrix, bin_indices, verbose=0
):
    """
    Merge single-sample bins with the nearest bin.

    Args:
        single_sample_bins (np.ndarray): Array of single-sample bin indices.
        distance_matrix (np.ndarray): Distance matrix between centroids.
        bin_indices (np.ndarray): Array of bin indices for each sample.
        verbose (int): Verbosity setting (0=silent, 3=most verbose).

    Returns:
        np.ndarray: Updated array of bin indices.
    """
    updated_bin_indices = bin_indices.copy()
    for single_bin in single_sample_bins:
        distances = distance_matrix[single_bin, :]
        distances[single_bin] = np.inf
        nearest_bin = np.argmin(distances)
        updated_bin_indices[bin_indices == single_bin] = nearest_bin

        if verbose >= 1:
            logger.info(f"Merged single-sample bin {single_bin} into bin {nearest_bin}")
    return updated_bin_indices


def identify_small_bins(bin_indices, num_bins, min_smote_neighbors=5):
    """
    Identify bins with a count less than or equal to smote_neighbors.

    Args:
        bin_indices (np.ndarray): Array of bin indices for each sample.
        num_bins (int): Minimum number of bins to use with np.bincount.
        min_smote_neighbors (int): Minimum number of nearest neighbors to consider. Any bins with fewer samples than ``min_smote_neighbors`` will be merged later. Defaults to 5.

    Returns:
        np.ndarray: Indices of small bins.

    """
    bin_counts = np.bincount(bin_indices, minlength=num_bins)
    small_bins = np.where(bin_counts <= min_smote_neighbors)[0]
    return small_bins


def merge_small_bins(small_bins, distance_matrix, bin_indices):
    """
    Merge small bins with the closest neighboring bin.

    Args:
        small_bins (np.ndarray): Indices of small bins.
        distance_matrix (np.ndarray): Distance matrix to determine nearest neighbors from.
        bin_indices (np.ndarray): Bin indices to compare with small_bins.

    Returns:
        np.ndarray: Updated bin indices with small bins merged into the next nearest bin.

    """
    updated_bin_indices = bin_indices.copy()
    for small_bin in small_bins:
        distances = distance_matrix[small_bin, :]
        distances[small_bin] = np.inf
        closest_bin = np.argmin(distances)
        updated_bin_indices[bin_indices == small_bin] = closest_bin
    return updated_bin_indices


def calculate_centroid_distances(centroids):
    """
    Calculate geographical distances between centroids.

    Args:
        centroids (list): List of centroids (longitude, latitude).

    Returns:
        np.ndarray: Matrix of distances between centroids.
    """
    # Convert to (latitude, longitude) for geopy
    centroids_latlon = [(lat, lon) for lon, lat in centroids]
    distance_matrix = cdist(
        centroids_latlon, centroids_latlon, lambda u, v: great_circle(u, v).kilometers
    )
    return distance_matrix


def define_jenks_thresholds(density, num_classes):
    """
    Define thresholds using Jenks Natural Breaks for binning.

    Args:
        density (np.ndarray): Density values from KDE.
        num_classes (int): Number of classes to divide the data into.

    Returns:
        np.ndarray: Threshold values for binning.
    """
    # Flatten the density grid and apply Jenks Natural Breaks
    flat_density = density.ravel()
    breaks = jenkspy.jenks_breaks(flat_density, n_classes=num_classes)

    # Return the breaks as thresholds, excluding the minimum value
    return np.array(breaks[1:])


def calculate_bin_centers(density_grid, lon_grid, lat_grid, thresholds):
    """
    Calculate the centers (centroids) of bins.

    Args:
        density_grid (np.ndarray): Grid of density values from KDE.
        lon_grid (np.ndarray): Grid of longitude values.
        lat_grid (np.ndarray): Grid of latitude values.
        thresholds (np.ndarray): Threshold values for binning.

    Returns:
        list: List of centroids (longitude, latitude) for each bin.
    """
    bin_centers = []
    for i in range(len(thresholds) + 1):
        if i == 0:
            mask = density_grid < thresholds[i]
        elif i == len(thresholds):
            mask = density_grid >= thresholds[i - 1]
        else:
            mask = (density_grid >= thresholds[i - 1]) & (density_grid < thresholds[i])

        lon_center = np.mean(lon_grid[mask])
        lat_center = np.mean(lat_grid[mask])
        bin_centers.append((lon_center, lat_center))

    return bin_centers


def assign_samples_to_bins(samples, density_grid, lon_grid, lat_grid, thresholds):
    """
    Assign samples to bins based on density thresholds.

    Args:
        samples (np.ndarray): Array of samples (longitude, latitude).
        density_grid (np.ndarray): Grid of density values from KDE.
        lon_grid (np.ndarray): Grid of longitude values.
        lat_grid (np.ndarray): Grid of latitude values.
        thresholds (np.ndarray): Threshold values for binning.

    Returns:
        np.ndarray: Bin index for each sample.
    """
    sample_bins = np.zeros(samples.shape[0], dtype=int)

    # Interpolate density for each sample
    for i, (lon, lat) in enumerate(samples):
        idx_lon = np.searchsorted(lon_grid[:, 0], lon) - 1
        idx_lat = np.searchsorted(lat_grid[0, :], lat) - 1
        sample_density = density_grid[idx_lon, idx_lat]

        # Assign bin based on thresholds
        sample_bins[i] = np.digitize(sample_density, thresholds)

    return sample_bins


def spatial_kde(longitudes, latitudes, bandwidth=0.05):
    """
    Perform spatial KDE on longitude and latitude data in decimal degrees.

    Args:
        longitudes (np.ndarray): Array of longitudes in decimal degrees.
        latitudes (np.ndarray): Array of latitudes in decimal degrees.
        bandwidth (float): Bandwidth for KDE. Defualts to 0.05.

    Returns:
        np.ndarray: Density values.
        np.ndarray: Grid of longitude values.
        np.ndarray: Grid of latitude values.
    """
    # Perform KDE
    xy = np.vstack([longitudes, latitudes])
    kde = gaussian_kde(xy, bw_method=bandwidth)

    # Create a grid
    lon_min, lon_max = longitudes.min(), longitudes.max()
    lat_min, lat_max = latitudes.min(), latitudes.max()
    lon_grid, lat_grid = np.mgrid[lon_min:lon_max:100j, lat_min:lat_max:100j]

    # Evaluate KDE on the grid
    zz = np.reshape(
        kde(np.vstack([lon_grid.ravel(), lat_grid.ravel()])), lon_grid.shape
    )

    return zz, lon_grid, lat_grid


def define_density_thresholds(density, num_bins):
    """
    Define density thresholds for binning.

    Args:
        density (np.ndarray): Density values from KDE.
        num_bins (int): Number of bins to create.

    Returns:
        np.ndarray: Threshold values for binning.
    """
    thresholds = np.quantile(density.ravel(), np.linspace(0, 1, num_bins + 1))
    return thresholds[1:-1]  # Exclude the min and max


def get_kde_bins(n_bins, dfy, bandwidth=0.04):
    """
    Calculate the 1D centroids of bins for each dimension in 2D data.

    Args:
        n_bins (int): Number of bins to divide each dimension.
        dfy (np.ndarray): 2D input data for KDE.
        bandwidth (float): Bandwidth for KDE.

    Returns:
        tuple: Two arrays containing centroids of bins for each dimension.
    """
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")

    data = dfy.to_numpy()

    centroids = []
    for dim in range(data.shape[1]):
        # Fit KDE for this dimension
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(data[:, dim].reshape(-1, 1))

        # Define bins for this dimension
        bin_edges = np.linspace(data[:, dim].min(), data[:, dim].max(), n_bins + 1)
        dim_centroids = (bin_edges[:-1] + bin_edges[1:]) / 2

        centroids.append(dim_centroids)

    return tuple(centroids)


def get_centroids(centroids, y_filtered, bins_filtered, unique_bins):
    if centroids is None:
        centroids = {
            bin: y_filtered[bins_filtered == bin].mean(axis=0)
            for bin in unique_bins
            if bin != -1
        }
    else:
        centroids = {bin: centroids[i] for i, bin in enumerate(unique_bins)}
    return centroids


def run_binned_smote(
    args,
    feature_names,
    ssc,
    centroids,
    bins_filtered,
    X_scaled,
    smote_neighbors,
    method,
    labels,
):
    """
    Runs SMOTEENN, and adjusts the number of neighbors for SMOTE based on the minimum number of samples in the least populous bin.

    Args:
        args (argparse.Namespace): All user-supplied and default arguments.
        feature_names (list): List of feature names.
        ssc (object): Some object related to the function (not specified).
        centroids (np.array): Array of centroids.
        bins_filtered (np.array): Array representing bins.
        X_scaled (np.array): Scaled feature array.
        smote_neighbors (int): Initial number of neighbors for SMOTE.
        method (str): The method used for SMOTE.
        labels (np.ndarray or pd.DataFrame): Labels to use. Only used in specific circumstances.

    Returns:
        tuple: features_resampled, labels_resampled, sammple_weights_resampled, centroids_resampled, and bins_resampled
    """
    # Count the number of occurrences for each bin
    bin_counts = np.bincount(bins_filtered)

    # Find the count of the least populous bin
    # Ensure no zero count bins are considered
    least_populous_bin_count = np.min(bin_counts[bin_counts > 0])

    # Adjust smote_neighbors to be at most one less than the count in the least populous bin
    smote_neighbors = min(least_populous_bin_count - 1, smote_neighbors)

    # Ensure smote_neighbors is not less than 1
    smote_neighbors = max(smote_neighbors, 1)

    smt = SMOTE(random_state=args.seed, k_neighbors=smote_neighbors)
    enn = EditedNearestNeighbours(n_neighbors=smote_neighbors)
    smote = SMOTEENN(random_state=args.seed, smote=smt, enn=enn)

    if X_scaled.shape[1] == len(feature_names):
        if isinstance(X_scaled, np.ndarray):
            X_scaled = np.hstack((X_scaled, labels))
        else:
            X_scaled = pd.concat([X_scaled, labels], axis=1)

    try:
        features_resampled, bins_resampled = smote.fit_resample(X_scaled, bins_filtered)
    except ValueError:
        return None, None, None, None, None

    if method != "kerneldensity":
        targets_resampled = np.array(
            [centroids[bin] for bin in bins_resampled if bin in centroids]
        )
    else:
        targets_resampled = np.array([centroids[bin] for bin in bins_resampled])

    try:
        features_resampled = pd.DataFrame(
            ssc.inverse_transform(features_resampled),
            columns=list(feature_names) + ["x", "y"],
        )
    except (ValueError, KeyError) as e:
        if features_resampled.shape[1] == len(list(feature_names)) + 3:
            features_resampled = pd.DataFrame(
                ssc.inverse_transform(features_resampled),
                columns=list(feature_names) + ["x", "y"] + ["sample_weights"],
            )
            features_resampled.drop(["sample_weights"], axis=1, inplace=True)
        elif features_resampled.shape[1] == len(list(feature_names)) + 2:
            features_resampled = pd.DataFrame(
                ssc.inverse_transform(features_resampled),
                columns=list(feature_names) + ["x", "y"],
            )
    labels_resampled = features_resampled[["x", "y"]]
    features_resampled.drop(labels=["x", "y"], axis=1, inplace=True)
    centroids_resampled = targets_resampled.copy()

    sample_weights_resampled = None
    if "sample_weights" in feature_names:
        try:
            sample_weights_resampled = features_resampled["sample_weights"].to_numpy()
            sample_weights_resampled = torch.tensor(
                sample_weights_resampled, dtype=torch.float32
            )
            features_resampled.drop(labels=["sample_weights"], axis=1, inplace=True)
        except KeyError:
            sample_weights_resampled = torch.ones(
                features_resampled.shape[0], dtype=torch.float32
            )

    features_resampled = torch.tensor(
        features_resampled.to_numpy(), dtype=torch.float32
    )
    labels_resampled = torch.tensor(labels_resampled.to_numpy(), dtype=torch.float32)

    return (
        features_resampled,
        labels_resampled,
        sample_weights_resampled,
        centroids_resampled,
        bins_resampled,
    )


def setup_synth_resampling(features, labels, sample_weights):
    """
    Prepares data for synthetic resampling by converting feature and label arrays into DataFrames and incorporating sample weights.

    This function first validates that the input features, labels, and sample weights are NumPy arrays. It then creates pandas DataFrames for features and labels. If the sample weights are not uniform (all 1.0), they are added to the features DataFrame as a new column.

    Args:
        features (numpy.ndarray): The feature set, expected as a 2D NumPy array.
        labels (numpy.ndarray): The label set, expected as a 2D NumPy array with two columns (x, y).
        sample_weights (numpy.ndarray): An array of sample weights.

    Returns:
        tuple of DataFrame: A tuple containing two DataFrames:
                            - dfX: DataFrame of features with an additional column for sample weights if they are not uniform.
                            - dfy: DataFrame of labels.

    Notes:
        - This function is typically used as a preprocessing step before applying synthetic resampling techniques like SMOTE.
        - The function assumes that features and labels are provided as NumPy arrays and converts them into pandas DataFrames.
    """
    features, labels, sample_weights = validate_is_numpy(
        features, labels, sample_weights
    )

    dfX = pd.DataFrame(features, columns=range(features.shape[1]))
    dfy = pd.DataFrame(labels, columns=["x", "y"])

    if not np.all(sample_weights == 1.0):  # If not all 1's.
        dfX["sample_weights"] = sample_weights
    return dfX, dfy


def process_bins(X, y, bins):
    noise_filter = bins != -1
    X_filtered = X[noise_filter]  # The features + coordinates.
    y_filtered = y[noise_filter]
    bins_filtered = bins[noise_filter]
    X_filtered = np.hstack([X_filtered, np.expand_dims(y_filtered, axis=1)])

    ssc = StandardScaler()
    X_scaled = ssc.fit_transform(X_filtered)
    return y_filtered, bins_filtered, ssc, X_scaled


def custom_gpr_optimizer(obj_func, initial_theta, bounds):
    """
    Custom optimizer using scipy.optimize.minimize with increased maxiter.

    Args:
        obj_func (callable): The objective function.
        initial_theta (array-like): Initial guess for the parameters.
        bounds (list of tuples): Bounds for the parameters.

    Returns:
        tuple: Optimized parameters and function value at the optimum.
    """
    # Call scipy.optimize.minimize with the necessary parameters
    opt_res = optimize.minimize(
        obj_func,
        initial_theta,
        method="L-BFGS-B",
        jac=True,  # True if obj_func returns the gradient as well
        bounds=bounds,
        options={"maxiter": 15000, "eps": 1e-8},
    )

    # Check the result and return the optimized parameters
    if not opt_res.success:
        logger.warning(RuntimeWarning("Optimization failed: " + opt_res.message))

    theta_opt, func_min = opt_res.x, opt_res.fun
    return theta_opt, func_min


def cluster_minority_samples(minority_samples, n_clusters):
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
        verbose=0,
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

        if not use_kmeans and not use_kde and focus_regions is None:
            if objective_mode:
                use_kde = True
            else:
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
        self.verbose = verbose

        if indices is None:
            self.indices = np.arange(data.shape[0])
        else:
            self.indices = indices

        self.optimal_k_neighbors = self.find_optimal_k_neighbors()
        self.weights = self.calculate_weights()

    @staticmethod
    def estimate_density(y_true):
        """
        Estimate the density of points in the dataset.

        Args:
            y_true (np.ndarray): Truth values.

        Returns:
            float: An estimate of the density.
        """
        # Example density estimation logic
        neighbors = NearestNeighbors(
            n_neighbors=2
        )  # Using 2 to find the nearest neighbor
        neighbors.fit(y_true)
        distances, _ = neighbors.kneighbors(y_true)
        average_distance = np.mean(
            distances[:, 1]
        )  # distances[:, 0] is zero as it's the distance to itself
        return average_distance

    def define_local_region(self, y_true, point, dynamic_k):
        """
        Define a local region around a given point for GWR using dynamic nearest neighbors.

        Args:
            y_true (np.ndarray): Truth values.
            point (np.ndarray): The central point of the local region.
            dynamic_k (int): Dynamically determined number of nearest neighbors.

        Returns:
            np.ndarray: A boolean array indicating which points are in the local region.
        """
        neighbors = NearestNeighbors(n_neighbors=dynamic_k)
        neighbors.fit(y_true)
        indices = neighbors.kneighbors([point], return_distance=False)
        local_region = np.zeros(y_true.shape[0], dtype=bool)
        local_region[indices[0]] = True
        return local_region

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
        local_metrics = []

        for point in targets:
            # Determine the dynamic number of neighbors based on optimal k
            # neighbors
            dynamic_k = self.find_optimal_k_neighbors()

            # Define the local region around the current point using dynamic
            # nearest neighbors
            local_region = self.define_local_region(targets, point, dynamic_k)

            # Extract the subset of predictions and targets that fall within
            # the local region
            local_predictions = predictions[local_region]
            local_targets = targets[local_region]

            # Calculate Haversine error for each pair of points
            local_error = haversine_distances_agg(
                local_targets, local_predictions, np.array
            )

            # Adjust the local error using sample weights
            if sample_weights is not None:
                local_error_weighted = local_error * sample_weights[local_region]
                local_metric = np.mean(local_error_weighted)
            else:
                local_metric = np.mean(local_error)

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
        """Calculate sample weights for geographic coordinates in the form of longitude and latitude.

        There are several options for calculating sample weights:

        - KMeans clustering
        - Kernel Density Estimation (KDE)
        - The KDE method sets the bandwidth adaptively to obtain optimal smoothing for the kernel.
        - User-specified focal regions; users should specify min/ max longitude and latitude to create a bounding box.
        - The weights can optionally be normalized from 0 to 1.

        """
        if not self.objective_mode:
            if self.verbose >= 1:
                self.logger.info("Estimating sample weights...")

        weights = np.ones(len(self.data))

        if self.use_kmeans:
            if not self.objective_mode and self.verbose >= 1:
                self.logger.info("Estimating sample weights with KMeans...")
            n_clusters = self.find_optimal_clusters()
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
            labels = kmeans.fit_predict(self.data)
            cluster_counts = np.bincount(labels, minlength=n_clusters)
            cluster_weights = 1 / (cluster_counts[labels] + 1e-5)
            mms = MinMaxScaler(feature_range=(1, 300))
            weights *= np.squeeze(mms.fit_transform(cluster_weights.reshape(-1, 1)))
        if self.use_kde:
            if not self.objective_mode and self.verbose >= 1:
                self.logger.info("Estimating sample weights with Kernel Density...")
            adaptive_bandwidth = self.calculate_adaptive_bandwidth(
                self.optimal_k_neighbors
            )
            kde = KernelDensity(
                bandwidth=adaptive_bandwidth,
                kernel="gaussian",
                metric="haversine",
            )
            kde.fit(self.data)
            log_density = kde.score_samples(self.data)
            self.density = np.exp(log_density)

            # control aggressiveness of inverse weighting.
            weights *= 1 / np.power(self.density + 1e-5, self.w_power)

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

        if not self.objective_mode and self.verbose >= 1:
            self.logger.info("Done estimating sample weights.")

        if self.normalize:
            mms = MinMaxScaler()
            weights = np.squeeze(mms.fit_transform(weights.reshape(-1, 1)))
        return weights

    def calculate_adaptive_bandwidth(self, k_neighbors):
        """Calculate adaptive bandwidth for Kernel Density Estimation (KDE).

        Args:
            k_neighbors (int): Number of nearest neighbors to consider.

        Returns:
            float: Bandwidth to use with KDE.

        """
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, n_jobs=-1).fit(self.data)
        distances, _ = nbrs.kneighbors(self.data)

        # Exclude the first distance (self-distance)
        average_distance = np.mean(distances[:, 1:], axis=1)
        adaptive_bandwidth = np.mean(average_distance)  # Mean over all points
        return adaptive_bandwidth

    def find_optimal_k_neighbors(self):
        """Uses use the stability of the average distance as the criterion for determining optimal nearest neighbors.

        Returns:
            int: Optimal number of nearest neighbors (K).
        """
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
        """Search for the optimal number of clusters for KMeans clustering method. Uses the lowest Mean Silhouette Width (MSW) to obtain optimal K.

        Returns:
            int: Optimal number of clusters (K) for KMeans clsutering.
        """
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
        """Allows use as a PyTorch sampler."""
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
