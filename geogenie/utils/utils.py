import logging
import signal
from contextlib import contextmanager
from io import StringIO

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import OPTICS, KMeans

from geogenie.utils.exceptions import TimeoutException

logger = logging.getLogger(__name__)


def check_column_dtype(df, column_name):
    """
    Check if a DataFrame column is string dtype or numeric dtype.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        column_name (str): The name of the column to check.

    Returns:
        str: 'string' if the column is of string dtype, 'numeric' if the column is of numeric dtype, otherwise 'other'.
    """
    dtype = df[column_name].dtype
    if dtype == "object" or dtype == "string":
        return "string"
    elif pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    else:
        return "other"


def detect_separator(file_path):
    """
    Detects the separator used in a CSV file by reading the first line.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        str: The detected separator (one of ',', '\t', or ' ').
    """
    with open(file_path, "r") as file:
        first_line = file.readline()

    if "," in first_line:
        return ","
    elif "\t" in first_line:
        return "\t"
    else:
        return " "


def read_csv_with_dynamic_sep(file_path):
    """
    Reads a CSV file with a dynamically detected separator, replacing spaces with tabs in-memory.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The DataFrame containing the CSV data.
    """
    sep = detect_separator(file_path)

    if sep == " ":
        with open(file_path, "r") as file:
            data = file.read().replace(" ", "\t")
        data_io = StringIO(data)
        sep = "\t"
        return pd.read_csv(data_io, sep=sep)
    else:
        return pd.read_csv(file_path, sep=sep)


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
