import logging
import math
from math import asin, cos, radians, sin, sqrt

import numba
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import median_absolute_error, r2_score

logger = logging.getLogger(__name__)


def kstest(y_true, y_pred, sample_weight=None):
    # Calculate Haversine error for each pair of points
    haversine_errors = np.array(
        [
            haversine(act[0], act[1], pred[0], pred[1])
            for act, pred in zip(y_true, y_pred)
        ]
    )

    errors = np.array(haversine_errors)

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


# def evaluate_outliers(
#     train_samples, outlier_geo_indices, outlier_gen_indices, dt, args
# ):
#     if not isinstance(train_samples, pd.Series):
#         train_samples = pd.Series(train_samples, name="ID")

#     """Evaluate and plot results from GeoGenOutlierDetector."""
#     y_pred = process_pred(train_samples, outlier_geo_indices, outlier_gen_indices, dt)
#     y_true = process_truths(train_samples, dt)

#     plotting = PlotGenIE(
#         "cpu",
#         args.output_dir,
#         args.prefix,
#         args.show_plots,
#         args.fontsize,
#     )
#     plotting.plot_confusion_matrix(y_true["Label"], y_pred["Label"], dt)

#     logger.info(f"Accuracy: {accuracy_score(y_true['Label'], y_pred['Label'])}")
#     logger.info(f"F1 Score: {f1_score(y_true['Label'], y_pred['Label'])}")


def process_truths(train_samples):
    """Process true values."""
    truths = pd.read_csv("data/real_outliers.csv", sep=" ")
    truths["method"] = truths["method"].str.replace('"', "")
    truths["method"] = truths["method"].str.replace("'", "")
    y_true = pd.DataFrame(columns=["ID"])
    y_true["ID"] = train_samples
    y_true["Label"] = y_true["ID"].isin(truths["ID"]).astype(int)
    y_true["Type"] = "True"
    return y_true.sort_values(by=["ID"])


def process_pred(train_samples, outlier_geo_indices, outlier_gen_indices, label_type):
    """Process predictions."""
    # Create the y_pred DataFrame
    y_pred = pd.DataFrame(columns=["ID"])
    y_pred["ID"] = train_samples
    y_pred[label_type] = 0
    y_pred["Type"] = "Pred"

    # Convert outlier indices to IDs
    outlier_geo_ids = train_samples.iloc[outlier_geo_indices]
    outlier_gen_ids = train_samples.iloc[outlier_gen_indices]

    outlier_ids = pd.concat([outlier_geo_ids, outlier_gen_ids])

    # Mark the outliers in y_pred based on IDs
    y_pred["Label"] = y_pred["ID"].isin(outlier_ids).astype(int)
    return y_pred.sort_values(by=["ID"])


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
