import logging
import os
import signal
import sys
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import OPTICS, KMeans
from torch.utils.data import Dataset

from geogenie.utils.exceptions import TimeoutException

logger = logging.getLogger(__name__)


@contextmanager
def time_limit(seconds):
    """Context manager to terminate execution of anything within the context. If ``seconds`` are exceeded in terms of execution time, then the code within the context gets skipped."""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def validate_is_numpy(features, labels, sample_weights):
    """Ensure that features, labels, and sample_weights are numpy arrays and not PyTorch Tensors."""
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(sample_weights, torch.Tensor):
        sample_weights = sample_weights.numpy()
    return features, labels, sample_weights


@contextmanager
def suppress_output():
    """
    A context manager to suppress standard output and standard error.

    Usage:
        with suppress_output():
            # Code that generates output to be suppressed

    This temporarily redirects sys.stdout and sys.stderr to os.devnull.
    """

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def assign_to_bins(
    df,
    long_col,
    lat_col,
    n_bins,
    args,
    method="optics",
    min_samples=None,
    random_state=None,
):
    """
    Assign longitude and latitude coordinates to bins based on K-Means or DBSCAN clustering.

    Args:
        df (pd.DataFrame): DataFrame containing longitude and latitude columns.
        long_col (str): Name of the column containing longitude values.
        lat_col (str): Name of the column containing latitude values.
        n_bins (int): Number of bins (clusters) for KMeans or minimum samples for OPTICS.
        method (str): Clustering method ('kmeans' or 'optics'). Defaults to 'optics'.
        min_samples (int): Minimum number of samples for OPTICS. Defaults to None (4).
        random_state (int or RandomState): Random seed or state for reproducibility. Defaults to None (new seed each time).

    Returns:
        np.ndarray: Numpy array with bin indices indicating the bin assignment.
    """

    if method not in ["kmeans", "optics"]:
        msg = f"Invalid 'method' parameter passed to 'assign_to_bin()': {method}"
        logger.error(msg)
        raise ValueError(msg)

    if method == "kmeans":
        coordinates = df[[long_col, lat_col]].to_numpy()
        model = KMeans(
            n_clusters=n_bins, random_state=random_state, n_init="auto", max_iter=1000
        )
    else:
        coordinates = df[[lat_col, long_col]].to_numpy()
        if min_samples is None:
            min_samples = 4

        model = OPTICS(min_samples=min_samples, metric="haversine", n_jobs=args.n_jobs)

    model.fit(coordinates)
    bins = model.labels_  # NOTE: -1 indicates noise points with OPTICS

    if method == "kmeans":
        return bins, model.cluster_centers_
    return bins, None


class CustomDataset(Dataset):
    def __init__(self, features, labels, sample_weights=None):
        """
        Initialize custom PyTorch Dataset that incorporates sample weighting.

        Args:
            features (Tensor): Input features.
            labels (Tensor): Labels corresponding to the features.
            sample_weights (Tensor): Weights for each sample. If None, then a sample_weights tensor is still created, but all weights will be equal to 1.0 (equal weighting). Defaults to None.

        Attributes:
            features (torch.Tensor): Input features.
            labels (torch.Tensor): Labels corresponding to features.
            sample_weights (torch.Tensor): Sample weights of shape (n_samples,).
            tensors (tuple): Tuple consisting of (features, labels, sample_weights).
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)

        if sample_weights is None:
            sample_weights = torch.ones(len(features), dtype=torch.float32)
        else:
            if not isinstance(sample_weights, torch.Tensor):
                sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights

        # Store the tensors as a tuple
        self.tensors = (features, labels, sample_weights)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (feature, label, sample_weight) for the specified index.
        """

        return self.features[idx], self.labels[idx], self.sample_weights[idx]


class LongLatToCartesianTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for converting longitude and latitude to/ from Cartesian coordinates.

    Attributes:
        radius (float): The radius of the Earth in kilometers.
        placeholder (bool): If True, does not actually do transformation. Makes it to where target normalization can easily be toggled for development purposed.
    """

    def __init__(self, radius=6371, placeholder=False):
        """
        Initialize the transformer with the Earth's radius.

        Args:
            radius (float, optional): The radius of the Earth in kilometers. Defaults to 6371.
        """
        self.radius = radius
        self.placeholder = placeholder

    def fit(self, X, y=None):
        """Fit method for compatibility with scikit-learn. Does nothing."""
        return self

    def transform(self, X):
        """
        Convert latitude and longitude to Cartesian coordinates.

        Args:
            X (array-like): The longitude and latitude values. Shape (n_samples, 2).

        Returns:
            np.ndarray: Cartesian coordinates. Shape (n_samples, 3).
        """

        if self.placeholder:
            return X.copy()
        lon_rad = np.radians(X[:, 0])
        lat_rad = np.radians(X[:, 1])

        x = self.radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = self.radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = self.radius * np.sin(lat_rad)

        return np.column_stack((x, y, z))

    def inverse_transform(self, X):
        """
        Convert Cartesian coordinates back to latitude and longitude.

        Args:
            X (array-like): Cartesian coordinates. Shape (n_samples, 3).

        Returns:
            np.ndarray: Longitude and latitude values. Shape (n_samples, 2).
        """
        if self.placeholder:
            return X.copy()

        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        lat = np.degrees(np.arcsin(z / self.radius))
        lon = np.degrees(np.arctan2(y, x))

        return np.column_stack((lon, lat))


def geo_coords_is_valid(coordinates):
    """
    Validates that a given NumPy array contains valid geographic coordinates.

    Args:
        coordinates (np.ndarray): A NumPy array of shape (n_samples, 2) where the first column is longitude and the second is latitude.

    Raises:
        ValueError: If the array shape is not (n_samples, 2), or if the longitude and latitude values are not in their respective valid ranges.

    Returns:
        bool: True if the validation passes, indicating the coordinates are valid.
    """
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.numpy()

    # Check shape
    if coordinates.shape[1] != 2:
        msg = f"Array must be of shape (n_samples, 2): {coordinates.shape}"
        logger.error(msg)
        raise ValueError(msg)

    # Validate longitude and latitude ranges
    longitudes, latitudes = coordinates[:, 0], coordinates[:, 1]

    if not np.all((-180 <= longitudes) & (longitudes <= 180)):
        msg = f"Longitude values must be between -180 and 180 degrees: min, max = longitude: {np.min(longitudes)}, {np.max(longitudes)}"
        logger.error(msg)
        raise ValueError(msg)
    if not np.all((-90 <= latitudes) & (latitudes <= 90)):
        msg = f"Latitude values must be between -90 and 90 degrees: {np.min(latitudes)}, {np.max(latitudes)}"
        logger.error(msg)
        raise ValueError(msg)
    return True


def rmse_to_distance(rmse, latitudes):
    """
    Convert RMSE in decimal degrees to distance in kilometers for a batch of predictions,
    and calculate the mean, median, and standard deviation of the distance errors.

    Args:
        rmse (numpy.ndarray): RMSE values in decimal degrees, shape [batch_size, 2] ([rmse_longitude, rmse_latitude]).
        latitudes (numpy.ndarray): Latitudes at which to estimate the distances, shape (n_samples,).

    Returns:
        dict: A dictionary containing the mean, median, and standard deviation of the error distances.
    """
    km_per_degree_latitude = 111  # Approximate value; better away from poles.
    km_per_degree_longitude = km_per_degree_latitude * np.cos(np.radians(latitudes))

    # Convert RMSE from degrees to kilometers
    distance_longitude = rmse[:, 0] * km_per_degree_longitude
    distance_latitude = rmse[:, 1] * km_per_degree_latitude

    # Combine the latitudinal and longitudinal errors
    total_distance_error = np.sqrt(distance_longitude**2 + distance_latitude**2)

    # Calculate mean, median, and standard deviation
    mean_error = np.mean(total_distance_error)
    median_error = np.median(total_distance_error)
    std_dev_error = np.std(total_distance_error)

    return {
        "mean": mean_error,
        "median": median_error,
        "std_dev": std_dev_error,
        "dist_error": total_distance_error,
    }


class StreamToLogger:
    """
    Custom stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


@contextmanager
def redirect_logging(logger):
    """Redirects logging by manipulating sys.stdout and sys.stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def get_iupac_dict():
    return {
        ("A", "A"): "A",
        ("C", "C"): "C",
        ("G", "G"): "G",
        ("T", "T"): "T",
        ("A", "C"): "M",
        ("C", "A"): "M",  # A or C
        ("A", "G"): "R",
        ("G", "A"): "R",  # A or G
        ("A", "T"): "W",
        ("T", "A"): "W",  # A or T
        ("C", "G"): "S",
        ("G", "C"): "S",  # C or G
        ("C", "T"): "Y",
        ("T", "C"): "Y",  # C or T
        ("G", "T"): "K",
        ("T", "G"): "K",  # G or T
        ("A", "C", "G"): "V",
        ("C", "A", "G"): "V",
        ("G", "A", "C"): "V",
        ("G", "C", "A"): "V",
        ("C", "G", "A"): "V",
        ("A", "G", "C"): "V",  # A or C or G
        ("A", "C", "T"): "H",
        ("C", "A", "T"): "H",
        ("T", "A", "C"): "H",
        ("T", "C", "A"): "H",
        ("C", "T", "A"): "H",
        ("A", "T", "C"): "H",  # A or C or T
        ("A", "G", "T"): "D",
        ("G", "A", "T"): "D",
        ("T", "A", "G"): "D",
        ("T", "G", "A"): "D",
        ("G", "T", "A"): "D",
        ("A", "T", "G"): "D",  # A or G or T
        ("C", "G", "T"): "B",
        ("G", "C", "T"): "B",
        ("T", "C", "G"): "B",
        ("T", "G", "C"): "B",
        ("G", "T", "C"): "B",
        ("C", "T", "G"): "B",  # C or G or T
        ("A", "C", "G", "T"): "N",
        ("C", "A", "G", "T"): "N",
        ("G", "A", "C", "T"): "N",
        ("T", "A", "C", "G"): "N",
        ("T", "C", "A", "G"): "N",
        ("G", "T", "A", "C"): "N",
        ("G", "C", "T", "A"): "N",
        ("C", "G", "T", "A"): "N",
        ("T", "G", "C", "A"): "N",
        ("A", "T", "G", "C"): "N",
        ("N", "N"): "N",  # any nucleotide
    }


def base_to_int():
    return {
        "A": 0,
        "T": 1,
        "G": 2,
        "C": 3,
        "N": 4,
        "R": 5,
        "Y": 6,
        "S": 7,
        "W": 8,
        "K": 9,
        "M": 10,
        "B": 11,
        "D": 12,
        "H": 13,
        "V": 14,
        "Z": 15,
    }
