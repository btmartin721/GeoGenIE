import logging
import math

import numba
import numpy as np
import scipy.stats as stats
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import root_mean_squared_error
from statsmodels.stats.diagnostic import lilliefors

from geogenie.utils.spatial_data_processors import SpatialDataProcessor

logger = logging.getLogger(__name__)
processor = SpatialDataProcessor(output_dir=None, logger=logger)


def kstest(y_true, y_pred, sample_weight=None):
    """Test whether the Haversine errors are normally distributed.

    The normal distribution's mean and standard deviation are estimated from the same errors, so the plain Kolmogorov-Smirnov p-value would be too large. The Lilliefors correction of the Kolmogorov-Smirnov test is used instead (``statsmodels.stats.diagnostic.lilliefors``), which accounts for the estimated parameters.

    Args:
        y_true (np.ndarray): Ground truth coordinates (longitude, latitude).
        y_pred (np.ndarray): Predicted coordinates (longitude, latitude).
        sample_weight (np.ndarray, optional): Not used. Defaults to None.

    Returns:
        tuple: Kolmogorov-Smirnov (Lilliefors) statistic, its p-value, and the skewness of the errors.
    """
    # Calculate Haversine error for each pair of points
    haversine_errors = processor.haversine_distance(y_true, y_pred)
    errors = np.asarray(haversine_errors, dtype=float).copy()

    # Statistical Distribution Analysis
    skewness_error = stats.skew(errors)

    # Kolmogorov-Smirnov test with estimated parameters (Lilliefors).
    if errors.size < 4 or np.std(errors) == 0:
        return np.nan, np.nan, skewness_error
    ks_statistic, p_value = lilliefors(errors, dist="norm", pvalmethod="table")

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

    The squared Pearson correlation is computed separately for each column (e.g., longitude and latitude, or each locus) and averaged over the columns where it is defined (columns with zero variance are skipped).

    Args:
        predicted_data (np.array): Predicted data from KNN.
        actual_data (np.array): Actual data.

    Returns:
        float: R^2 value.
    """
    predicted_data = np.asarray(predicted_data, dtype=float)
    actual_data = np.asarray(actual_data, dtype=float)
    if predicted_data.ndim == 1:
        predicted_data = predicted_data[:, None]
        actual_data = actual_data[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared = np.array(
            [
                np.corrcoef(predicted_data[:, j], actual_data[:, j])[0, 1] ** 2
                for j in range(predicted_data.shape[1])
            ]
        )
    if np.all(np.isnan(r_squared)):
        return np.nan
    return np.nanmean(r_squared)


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
