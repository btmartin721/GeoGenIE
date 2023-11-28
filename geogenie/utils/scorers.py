import logging
from math import asin, cos, radians, sin, sqrt

import numpy as np
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


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
