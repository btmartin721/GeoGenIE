import logging
import math

import numba
import numpy as np
import scipy.stats as stats
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import root_mean_squared_error

from geogenie.utils.spatial_data_processors import SpatialDataProcessor

logger = logging.getLogger(__name__)
processor = SpatialDataProcessor(output_dir=None, logger=logger)


def kstest(y_true, y_pred, sample_weight=None):
    """Perform the Kolmogorov-Smirnov test on the Haversine errors."""
    # Calculate Haversine error for each pair of points
    haversine_errors = processor.haversine_distance(y_true, y_pred)
    errors = haversine_errors.copy()

    # Statistical Distribution Analysis
    mean_error = np.mean(errors)
    std_dev_error = np.std(errors)
    skewness_error = stats.skew(errors)

    # Kolmogorov-Smirnov Test
    ks_statistic, p_value = stats.kstest(
        errors, "norm", args=(mean_error, std_dev_error)
    )

    return (
        ks_statistic,
        p_value,
        skewness_error,
    )


class LocallyLinearEmbeddingWrapper(LocallyLinearEmbedding):
    """Wrapper class for LocallyLinearEmbedding to allow for prediction."""

    def predict(self, X):
        return self.transform(X)

    @staticmethod
    def lle_reconstruction_scorer(estimator, X, y=None):
        """
        Compute the negative reconstruction error for an LLE model to use as a scorer.
        GridSearchCV assumes that higher score values are better, so the reconstruction
        error is negated.

        Args:
            estimator (LocallyLinearEmbedding): Fitted LLE model.
            X (numpy.ndarray): Original high-dimensional data.

        Returns:
            float: Negative reconstruction error.
        """
        return -estimator.reconstruction_error_


def calculate_r2_knn(predicted_data, actual_data):
    """Calculate the coefficient of determination (R^2) for predictions.

    Args:
        predicted_data (np.array): Predicted data from KNN.
        actual_data (np.array): Actual data.

    Returns:
        float: R^2 value.
    """
    correlation_matrix = np.corrcoef(predicted_data, actual_data)
    r_squared = correlation_matrix[0, 1] ** 2
    return np.mean(r_squared)


def calculate_rmse(preds, targets):
    haversine_errors = processor.haversine_distance(targets, preds)
    return root_mean_squared_error(np.zeros_like(haversine_errors), haversine_errors)


@numba.njit(fastmath=True)
def haversine_distance(coord1, coord2):
    """Calculate the Haversine distance between two geographic coordinate points.

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
