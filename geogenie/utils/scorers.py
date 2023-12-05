import logging
import math
from math import asin, cos, radians, sin, sqrt

import numba
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


def calculate_r2_knn(predicted_data, actual_data):
    """
    Calculate the coefficient of determination (R^2) for predictions.

    Args:
        predicted_data (np.array): Predicted data from KNN.
        actual_data (np.array): Actual data.

    Returns:
        float: R^2 value.
    """
    correlation_matrix = np.corrcoef(predicted_data, actual_data)
    r_squared = correlation_matrix[0, 1] ** 2
    return r_squared


def calculate_r2(actual_coords, predicted_coords):
    """
    Calculate the coefficient of determination (r^2 value) for actual vs predicted coordinates.

    This method is tailored to the use case for geographic coordinates.

    Args:
    actual_coords (np.array): A 2D array of actual coordinates (longitude, latitude).
    predicted_coords (np.array): A 2D array of predicted coordinates (longitude, latitude).

    Returns:
    float: The calculated r^2 value.
    """
    # Checking if actual and predicted coordinates have the same shape
    if actual_coords.shape != predicted_coords.shape:
        logger.error(
            f"The shape of actual and predicted coordinates must be the same, "
            f"but got: {actual_coords.shape}, {predicted_coords.shape}"
        )
        raise ValueError(
            f"The shape of actual and predicted coordinates must be the same, "
            f"but got: {actual_coords.shape}, {predicted_coords.shape}"
        )

    # Mean of actual coordinates
    mean_actual = np.mean(actual_coords, axis=0)

    # Total sum of squares
    total_sum_of_squares = np.sum((actual_coords - mean_actual) ** 2)

    # Residual sum of squares
    residual_sum_of_squares = np.sum((actual_coords - predicted_coords) ** 2)

    # Calculating r^2
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2


def get_r2(y_true, y_pred, idx):
    """This is for the scikit-learn implementation of r2_score

    This is not the same as the calculate_r2 method."""
    return r2_score(y_true[:, idx], y_pred[:, idx], multioutput="variance_weighted")


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


@numba.jit(fastmath=True, nopython=True)
def haversine_distance(coord1, coord2):
    """
    Calculate the Haversine distance between two geographic coordinate points.

    Args:
        coord1, coord2 (tuple): Latitude and longitude for each point.

    Returns:
        float: Distance in kilometers.
    """
    radius = 6371  # Earth radius in kilometers
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c


def parallel_haversine(args):
    geo_chunk, pred_geo_coords, haversine_distance, scalar = args
    return (
        np.diagonal(cdist(geo_chunk, pred_geo_coords, metric=haversine_distance))
        / scalar
    )


def parallel_euclidean(args):
    gen_chunk, gen_coords = args
    return cdist(gen_chunk, gen_coords, metric="euclidean")


def parallel_haversine_sklearn(args):
    geo_chunk, sklearn_haversine, scale_factor = args
    dist_matrix_chunk = sklearn_haversine(geo_chunk) * 6371.0
    dist_matrix_chunk /= scale_factor
    return dist_matrix_chunk
