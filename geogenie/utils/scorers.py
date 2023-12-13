import logging
import math
from math import asin, cos, radians, sin, sqrt

import numba
import numpy as np
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


def haversine_distances_agg(y_true, y_pred, func):
    """
    Calculate the distance metric between y_true and y_pred using the specified aggregation function (e.g., np.mean or np.median)

    Args:
        y_true (numpy.ndarray): Array of true values (latitude, longitude).
        y_pred (numpy.ndarray): Array of predicted values (latitude, longitude).
        func (callable): Function to aggregate distances.

    Returns:
        float: Aggregated distance.
    """
    return func(
        [
            haversine(y_pred[x, 0], y_pred[x, 1], y_true[x, 0], y_true[x, 1])
            for x in range(len(y_pred))
        ]
    )


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
    return np.mean(r_squared)


def calculate_r2_sklearn(y_true, y_pred, idx):
    """This is for the scikit-learn implementation of r2_score

    This is not the same as the calculate_r2 method."""
    return r2_score(y_true[:, idx], y_pred[:, idx], multioutput="variance_weighted")


def r2_multioutput(preds, targets):
    r2_lon = np.corrcoef(preds[:, 0], targets[:, 0])[0][1] ** 2
    r2_lat = np.corrcoef(preds[:, 1], targets[:, 1])[0][1] ** 2
    return r2_lon, r2_lat


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


@numba.njit(fastmath=True)
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
