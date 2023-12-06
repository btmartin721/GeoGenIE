import logging
import sys
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger(__name__)


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
