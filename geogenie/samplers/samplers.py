# Standard library imports
import logging
import os

# Third party imports
import jenkspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from geopy.distance import great_circle
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from scipy import optimize
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Local application imports
from geogenie.utils.spatial_data_processors import SpatialDataProcessor
from geogenie.utils.utils import assign_to_bins, validate_is_numpy

logger = logging.getLogger(__name__)

processor = SpatialDataProcessor(output_dir=None, logger=logger)

os.environ["TQDM_DISABLE"] = "true"


import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


class GeographicDensitySampler:
    """Class to sample data points based on spatial density.

    This class provides a method to sample data points based on spatial density using KMeans, KDE, and DBSCAN.

    Attributes:
        data (pandas.DataFrame): DataFrame with 'longitude' and 'latitude'.
        focus_regions (list of tuples): Regions of interest.
        use_kmeans (bool): Use KMeans clustering for weights.
        use_kde (bool): Use KernelDensity Estimation for weights.
        use_dbscan (bool): Use DBSCAN clustering for weights.
        w_power (float): Aggressiveness of inverse density weighting.
        max_clusters (int): Max clusters for KMeans.
        max_neighbors (int): Max neighbors for adaptive bandwidth.
        indices (np.ndarray): Indices for sampling.
        normalize (bool): Normalize weights.
        verbose (int): Verbosity level.
        logger (logging.Logger): Logger instance.
        dtype (np.dtype): Data type for calculations.
    """

    def __init__(
        self,
        data,
        focus_regions=None,
        use_kmeans=True,
        use_kde=True,
        use_dbscan=False,
        w_power=1,
        max_clusters=10,
        max_neighbors=10,
        indices=None,
        normalize=False,
        verbose=0,
        logger=None,
        dtype=np.float32,
    ):
        """Geographic Density Sampler for weighting based on spatial density.

        This class provides a method to sample data points based on spatial density using KMeans, KDE, and DBSCAN.

        Args:
            data (pandas DataFrame): DataFrame with 'longitude' and 'latitude'.
            focus_regions (list of tuples): Regions of interest.
            use_kmeans (bool): Use KMeans clustering for weights.
            use_kde (bool): Use KernelDensity Estimation for weights.
            use_dbscan (bool): Use DBSCAN clustering for weights.
            w_power (float): Aggressiveness of inverse density weighting.
            max_clusters (int): Max clusters for KMeans.
            max_neighbors (int): Max neighbors for adaptive bandwidth.
            indices (np.ndarray): Indices for sampling.
            normalize (bool): Normalize weights.
            verbose (int): Verbosity level.
            logger (logging.Logger): Logger instance.
            dtype (np.dtype): Data type for calculations.
        """
        self.data = data
        self.focus_regions = focus_regions
        self.use_kmeans = use_kmeans
        self.use_kde = use_kde
        self.use_dbscan = use_dbscan
        self.w_power = w_power
        self.max_clusters = max_clusters
        self.max_neighbors = max_neighbors
        self.normalize = normalize
        self.verbose = verbose
        self.dtype = dtype

        if indices is None:
            self.indices = np.arange(len(data))
        else:
            self.indices = indices

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        self.density = None
        self.weights = self.calculate_weights()
        self._plot_cluster_weights(self.weights)

    def _plot_cluster_weights(self, weights):
        """Plot the sample weights.

        This method plots the sample weights on a scatter plot of the data points.

        Args:
            weights (np.ndarray): Array of sample weights.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sns.set_style("white")

        data = self.data.copy()
        data["weights"] = weights * 1000

        ax = sns.scatterplot(
            data=data,
            x="x",
            y="y",
            hue="weights",
            size="weights",
            palette="viridis",
            legend=True,
            alpha=0.8,
            ax=ax,
        )

        ax.set_title("Sample Weights", fontsize=24)
        ax.set_xlabel("Longitude", fontsize=24)
        ax.set_ylabel("Latitude", fontsize=24)
        ax.legend(
            title="Sample Weights",
            loc="upper right",
            fancybox=True,
            shadow=True,
            fontsize=24,
        )

        fig.savefig(
            "sample_weights.png", bbox_inches="tight", facecolor="white", dpi=300
        )

    def calculate_weights(self):
        """Calculate sample weights using KMeans, KDE, DBSCAN, and focus regions.

        This method calculates sample weights using KMeans, KDE, and DBSCAN clustering, and adjusts for focus regions.

        Returns:
            np.ndarray: Computed sample weights.
        """
        if self.verbose >= 1:
            self.logger.info("Calculating sample weights...")
        weights = np.ones(len(self.data))

        # KMeans-based weights
        if self.use_kmeans:
            self.logger.info("Calculating KMeans-based weights...")
            kmeans_weights = self._calculate_kmeans_weights()
            weights *= kmeans_weights

        # KDE-based weights
        if self.use_kde:
            self.logger.info("Calculating KDE-based weights...")
            kde_weights = self._calculate_kde_weights()
            weights *= kde_weights

        # DBSCAN-based weights
        if self.use_dbscan:
            self.logger.info("Calculating DBSCAN-based weights...")
            dbscan_weights = self._calculate_dbscan_weights()
            weights *= dbscan_weights

        # Focus regions adjustment
        if self.focus_regions:
            self.logger.info("Adjusting weights for focus regions...")
            weights = self._adjust_for_focus_regions(weights)

        # Normalize final weights
        if self.normalize:
            self.logger.info("Normalizing weights...")
            scaler = MinMaxScaler(feature_range=(1, 10))
            weights = scaler.fit_transform(weights.reshape(-1, 1)).squeeze()

        self.logger.info("Weight calculation complete.")
        return weights

    def _calculate_kmeans_weights(self):
        """Compute weights using KMeans clustering.

        This method uses KMeans clustering to assign weights to samples based on cluster sizes.

        Returns:
            np.ndarray: KMeans-based weights.
        """
        n_clusters = self.find_optimal_clusters()
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
        labels = kmeans.fit_predict(self.data)
        cluster_counts = np.bincount(labels, minlength=n_clusters)
        cluster_weights = 1 / (cluster_counts[labels] + 1e-5)

        # Scale weights
        if self.normalize:
            scaler = MinMaxScaler(feature_range=(1, 10))
            return scaler.fit_transform(cluster_weights.reshape(-1, 1)).squeeze()
        else:
            return cluster_weights

    def _calculate_kde_weights(self):
        """Compute weights using Kernel Density Estimation.

        This method uses Kernel Density Estimation (KDE) to assign weights to samples based on density.

        Returns:
            np.ndarray: KDE-based weights.
        """
        kde = KernelDensity(
            bandwidth=0.03,
            kernel="gaussian",
            metric="haversine",
            algorithm="ball_tree",
        )
        kde.fit(self.data)
        log_density = kde.score_samples(self.data)
        density = np.exp(log_density)
        self.density = density

        # Compute weights
        kde_weights = 1 / np.power(density + 1e-5, self.w_power)

        # Cap extreme weights
        max_weight = np.percentile(kde_weights, 95)  # Cap at 95th percentile
        kde_weights = np.clip(kde_weights, None, max_weight)

        return kde_weights

    def _determine_eps(self, data, k=5):
        """Automatically determine the epsilon parameter for DBSCAN using k-distance.

        This method automatically determines the epsilon parameter for DBSCAN using the k-distance heuristic.

        Args:
            data (array-like): The data to cluster.
            k (int): Number of nearest neighbors.

        Returns:
            float: Optimal eps for DBSCAN.
        """
        # Compute pairwise distances
        distances = pairwise_distances(data)

        # k-th nearest neighbor distances
        k_distances = np.sort(distances, axis=1)[:, k]

        # Use the 95th percentile as heuristic
        elbow_point = np.percentile(k_distances, 95)

        if self.verbose >= 1:
            self.logger.info(f"Estimated epsilon (eps) for DBSCAN: {elbow_point}")
        return elbow_point

    def _calculate_dbscan_weights(self):
        """Compute weights using DBSCAN clustering.

        This method uses DBSCAN clustering to assign weights to samples based on cluster sizes.

        Returns:
            np.ndarray: DBSCAN-based weights.
        """
        eps = self._determine_eps(self.data)

        dbscan = DBSCAN(
            eps=eps, min_samples=2, metric="haversine", algorithm="ball_tree"
        )
        labels = dbscan.fit_predict(self.data)

        weights = np.ones(len(self.data))
        unique_labels = np.unique(labels)

        # Assign weights based on cluster sizes
        for label in unique_labels:
            if label != -1:  # For valid clusters
                cluster_size = np.sum(labels == label)
                weights[labels == label] = 1 / (cluster_size + 1e-5)

        # Scale weights for non-outliers only
        scaler = MinMaxScaler(feature_range=(1, 10))
        non_outlier_weights = weights[labels != -1]
        scaled_weights = scaler.fit_transform(
            non_outlier_weights.reshape(-1, 1)
        ).squeeze()

        # Assign scaled weights back to non-outliers
        weights[labels != -1] = scaled_weights

        # Assign fixed weight for outliers
        weights[labels == -1] = scaled_weights.mean()

        return weights

    def _adjust_for_focus_regions(self, weights):
        """Adjust weights for user-specified focus regions.

        This method adjusts the weights for user-specified focus regions by doubling the weights for samples within the regions.

        Args:
            weights (np.ndarray): Current weights.

        Returns:
            np.ndarray: Adjusted weights.
        """
        for region in self.focus_regions:
            lon_min, lon_max, lat_min, lat_max = region
            in_region = (
                (self.data["longitude"] >= lon_min)
                & (self.data["longitude"] <= lon_max)
                & (self.data["latitude"] >= lat_min)
                & (self.data["latitude"] <= lat_max)
            )
            weights[in_region] *= 2  # Multiplicative adjustment
        return weights

    def calculate_adaptive_bandwidth(self):
        """Calculate adaptive bandwidth for KDE using nearest neighbors.

        This method calculates the adaptive bandwidth for KDE using the average distance to the k-th nearest neighbor.

        Returns:
            float: Adaptive bandwidth.
        """
        nbrs = NearestNeighbors(n_neighbors=self.max_neighbors + 1, n_jobs=-1).fit(
            self.data
        )
        distances, _ = nbrs.kneighbors(self.data)
        average_distance = np.mean(distances[:, 1:], axis=1)
        return np.mean(average_distance)

    def find_optimal_clusters(self):
        """
        Find optimal number of clusters for KMeans using the Elbow Method.

        Returns:
            int: Optimal number of clusters.
        """
        distortions = []
        cluster_range = range(2, self.max_clusters + 1)
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
            kmeans.fit(self.data)
            distortions.append(kmeans.inertia_)

        elbow_index = np.argmin(np.gradient(distortions))
        return cluster_range[elbow_index]

    def __iter__(self):
        """Allows use as a PyTorch sampler.

        Returns:
            iterator: An iterator for the sampler.
        """
        return (
            self.indices[i]
            for i in np.random.choice(
                self.indices,
                size=len(self.indices),
                p=self.weights / np.sum(self.weights),
            )
        )

    def __len__(self):
        """Returns the number of samples.

        Returns:
            int: The number of samples
        """
        return len(self.data)


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
    """Performs synthetic resampling on the provided datasets using various binning methods and SMOTE (Synthetic Minority Over-sampling Technique).

    Args:
        features (DataFrame): The feature set.
        labels (DataFrame): The label set.
        sample_weights (np.ndarray): Array of sample weights.
        densities (np.ndarray): Sample densities.
        n_bins (int): The number of bins to use for resampling.
        args (Namespace): Arguments provided for resampling configurations.
        method (str, optional): The method to use for binning. Defaults to "kmeans".
        smote_neighbors (int, optional): The number of nearest neighbors to use in SMOTE. Defaults to 5.
        verbose (int): Verbosity setting. Defaults to 0 (silent).

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
        bins, centroids = do_kde_binning(n_bins, verbose, dfy, sample_weights)

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


def do_kde_binning(n_bins, verbose, dfy, sample_weights):
    """Perform binning using KernelDensity Estimation (KDE).

    Args:
        n_bins (int): Number of bins to create.
        verbose (int): Verbosity setting (0=silent, 3=most verbose).
        dfy (pd.DataFrame): DataFrame of labels.
        sample_weights (np.ndarray): Array of sample weights.

    Returns:
        np.ndarray: Array of bin indices.
        list: List of centroids for each bin.
    """

    density, lon_grid, lat_grid = spatial_kde(
        dfy["x"].to_numpy(), dfy["y"].to_numpy(), sample_weights
    )

    # Define thresholds for bins
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
    """Merge single-sample bins with the nearest bin.

    This method merges single-sample bins with the nearest neighboring bin based on the distance matrix between centroids.

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
    """Identify bins with a count less than or equal to smote_neighbors.

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
    """Merge small bins with the closest neighboring bin.

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
    """Calculate geographical distances between centroids.

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
    """Define thresholds using Jenks Natural Breaks for binning.

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
    """Calculate the centers (centroids) of bins.

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
    """Assign samples to bins based on density thresholds.

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


def spatial_kde(longitudes, latitudes, sample_weights):
    """Perform spatial KDE on longitude and latitude data in decimal degrees.

    Args:
        longitudes (np.ndarray): Array of longitudes in decimal degrees.
        latitudes (np.ndarray): Array of latitudes in decimal degrees.

    Returns:
        np.ndarray: Density values.
        np.ndarray: Grid of longitude values.
        np.ndarray: Grid of latitude values.
    """
    # Perform KDE
    xy = np.vstack([longitudes, latitudes])
    kde = gaussian_kde(xy, weights=sample_weights)

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
    """Define density thresholds for binning.

    Args:
        density (np.ndarray): Density values from KDE.
        num_bins (int): Number of bins to create.

    Returns:
        np.ndarray: Threshold values for binning.
    """
    thresholds = np.quantile(density.ravel(), np.linspace(0, 1, num_bins + 1))
    return thresholds[1:-1]  # Exclude the min and max


def get_kde_bins(n_bins, dfy, bandwidth=0.04):
    """Calculate the 1D centroids of bins for each dimension in 2D data.

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
    """Get centroids for each bin.

    Args:
        centroids (np.ndarray): Array of centroids.
        y_filtered (np.ndarray): Filtered labels.
        bins_filtered (np.ndarray): Filtered bins.
        unique_bins (np.ndarray): Unique bins.

    Returns:
        dict: Dictionary of centroids for each bin.
    """
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
    """Runs SMOTEENN, and adjusts the number of neighbors for SMOTE based on the minimum number of samples in the least populous bin.

    This method first calculates the number of occurrences for each bin and then finds the count of the least populous bin. It then adjusts the number of neighbors for SMOTE to be at most one less than the count in the least populous bin. The function ensures that the number of neighbors is not less than 1. It then performs SMOTEENN resampling on the scaled feature array and labels. If the method is "kerneldensity", the function uses all bins. The function then inversely transforms the resampled features and labels and drops the "x" and "y" columns. If the "sample_weights" column is present, the function converts it to a tensor and drops it from the features DataFrame. The function then converts the DataFrames to tensors and returns the resampled features, labels, sample weights, centroids, and bins.

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
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

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
                sample_weights_resampled, dtype=dtype
            )
            features_resampled.drop(labels=["sample_weights"], axis=1, inplace=True)
        except KeyError:
            sample_weights_resampled = torch.ones(
                (features_resampled.shape[0],), dtype=dtype
            )

    features_resampled = torch.tensor(features_resampled.to_numpy(), dtype=dtype)
    labels_resampled = torch.tensor(labels_resampled.to_numpy(), dtype=dtype)

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

    dfX["sample_weights"] = np.ones(dfX.shape[0])
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
