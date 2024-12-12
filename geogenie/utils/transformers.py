import logging

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_array, check_is_fitted


class MCA(BaseEstimator, TransformerMixin):
    """Class to perform Multiple Correspondence Analayis (MCA).

    This class performs Multiple Correspondence Analysis (MCA) on the input data.

    Attributes:
        n_components (int): Number of MCA components to output.
        n_iter (int): Number of randomized SVD iterations to perform.
        check_input (bool): Whether to check input data for conformity.
        random_state (int or None): Random state for reproducibility.
        one_hot (bool): Flag for one-hot encoding the input data.
        categories (list): Possible categories in input features.
        epsilon (float): Small value to prevent division by 0.
        logger (logging.Logger): Logger object for the class
    """

    def __init__(
        self,
        n_components=2,
        n_iter=10,
        check_input=True,
        random_state=None,
        one_hot=True,
        categories=[0, 1, 2],
        epsilon=1e-5,
    ):
        """Initialize the MCA class.

        Args:
            n_components (int, optional): Number of MCA components to output. Defaults to 2.
            n_iter (int, optional): Number of randomized SVD iterations to perform. Defaults to 10.
            check_input (bool, optional): Whether to check input data for conformity. Defaults to True.
            random_state (int or None, optional): Random state for reproducibility. Defaults to None.
            one_hot (bool, optional): Flag for one-hot encoding the input data. Defaults to True.
            categories (list, optional): Possible categories in input features. Defaults to [0, 1, 2].
            epsilon (float, optional): Small value to prevent division by 0. Defaults to 1e-5.
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.check_input = check_input
        self.random_state = random_state
        self.one_hot = one_hot
        self.categories = categories
        self.epsilon = epsilon

        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        """Fit the input data.

        Args:
            X (np.ndarray): Array to fit.
            y (None, optional): Ignored. This parameter exists only for compatibility with the sklearn API.
        """
        if self.check_input:
            X = check_array(X, dtype=None, force_all_finite="allow-nan")

        if not isinstance(self.categories, list):
            raise TypeError(
                f"'categories' must be a list, got: {type(self.categories)}"
            )

        if self.one_hot:
            categories = [np.array(self.categories)] * X.shape[1]
            self.one_hot_encoder_ = OneHotEncoder(categories=categories)
            X = self.one_hot_encoder_.fit_transform(X)

        # Normalize and handle zero sums
        X = self._normalize_data(X)

        S = self._compute_S_matrix(X)

        U, Sigma, VT = randomized_svd(
            S,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        self.U_, self.Sigma_, self.VT_ = U, Sigma, VT
        self._store_results()

        return self

    def transform(self, X):
        """Transform input data X using MCA.

        Args:
            X (np.ndarray): Array to transform.
        """
        check_is_fitted(self, ["U_", "Sigma_", "VT_", "row_sums_", "col_sums_"])

        X = check_array(X, dtype=None, force_all_finite="allow-nan")

        if self.one_hot:
            X = self.one_hot_encoder_.transform(X)

        X_normalized = self._normalize_data(X)

        # Calculate inverse square root of row sums
        row_sums = X_normalized.sum(axis=1)
        row_sums_inv_sqrt = np.power(row_sums, -0.5)

        # Ensure row_sums_inv_sqrt is a 1D array
        if row_sums_inv_sqrt.ndim != 1:
            row_sums_inv_sqrt = np.ravel(row_sums_inv_sqrt)

        # Create a diagonal matrix
        row_inv = sp.diags(row_sums_inv_sqrt)

        transformed_X = row_inv @ X_normalized @ self.VT_.T

        self.logger.debug(
            f"MCA Transformed X: {transformed_X}, Shape: {transformed_X.shape}"
        )

        return transformed_X

    def _normalize_data(self, X):
        """Normalize the input data.

        Args:
            X (np.ndarray): Array to normalize.

        Returns:
            np.ndarray: Normalized array.
        """

        X_normalized = X.astype(float) / X.sum()
        row_sums = X_normalized.sum(axis=1) + self.epsilon
        col_sums = X_normalized.sum(axis=0) + self.epsilon
        self.row_sums_, self.col_sums_ = row_sums, col_sums
        return X_normalized

    def _compute_S_matrix(self, X):
        """Compute the S matrix.

        Args:
            X (np.ndarray): The input data.

        Returns:
            sp.spmatrix: The S matrix
        """
        # Convert row_sums_ and col_sums_ to 1D numpy arrays if they are not
        # already
        row_sums = np.asarray(self.row_sums_).flatten()
        col_sums = np.asarray(self.col_sums_).flatten()

        # Calculate inverses of square roots element-wise
        row_inv = sp.diags(np.power(row_sums, -0.5))
        col_inv = sp.diags(np.power(col_sums, -0.5))

        # Compute the S matrix
        S = row_inv @ (X - np.outer(row_sums, col_sums)) @ col_inv
        return S

    def _store_results(self):
        """Store the results of the MCA."""
        self.eigenvalues_ = np.square(self.Sigma_)
        total_variance = np.sum(self.eigenvalues_)
        self.explained_inertia_ = self.eigenvalues_ / total_variance
        self.cumulative_inertia_ = np.cumsum(self.explained_inertia_)


class MinMaxScalerGeo(BaseEstimator, TransformerMixin):
    """Class to scale geographic coordinates to a specified range.

    Attributes:
        lat_range (tuple): Minimum and maximum values for latitude.
        lon_range (tuple): Minimum and maximum values for longitude.
        scale_min (float): Minimum value of the scaled range.
        scale_max (float): Maximum value of the scaled range.
        logger (logging.Logger): Logger object for the class.
    """

    def __init__(
        self, lat_range=(-90, 90), lon_range=(-180, 180), scale_min=0, scale_max=1
    ):
        """Initialize the MinMaxScalerGeo with specified ranges.

        This class scales geographic coordinates to a specified range.

        Args:
            lat_range (tuple): Minimum and maximum values for latitude.
            lon_range (tuple): Minimum and maximum values for longitude.
            scale_min (float): Minimum value of the scaled range.
            scale_max (float): Maximum value of the scaled range.
        """
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        """Fit does nothing as parameters are not data-dependent.

        Args:
            X (array-like): The data to fit. Ignored. This parameter exists only for compatibility with the sklearn API.
            y (None, optional): Ignored. This parameter exists only for compatibility with the sklearn API.

        Returns:
            self: Returns the instance itself.
        """
        return self

    def transform(self, X):
        """Scale the geographic coordinates based on the provided ranges.

        Args:
            X (array-like): The input coordinates to transform. Expected shape (n_samples, 2) where
                            X[:, 0] should be longitude and X[:, 1] should be latitude.

        Returns:
            np.array: Transformed coordinates, where each feature is scaled to [scale_min, scale_max].
        """
        # Ensure input is a numpy array
        X = np.asarray(X)

        # Unpack the ranges
        lon_min, lon_max = self.lon_range
        lat_min, lat_max = self.lat_range

        # Initialize the scaled array
        X_scaled = np.empty_like(X, dtype=float)

        # Transform longitudes
        X_scaled[:, 0] = (X[:, 0] - lon_min) / (lon_max - lon_min) * (
            self.scale_max - self.scale_min
        ) + self.scale_min

        # Transform latitudes
        X_scaled[:, 1] = (X[:, 1] - lat_min) / (lat_max - lat_min) * (
            self.scale_max - self.scale_min
        ) + self.scale_min

        self.logger.debug(
            f"Transformed Coordinates: {X_scaled}, Shape: {X_scaled.shape}"
        )

        return X_scaled

    def inverse_transform(self, X_scaled):
        """Scale back the coordinates to their original range.

        Args:
            X_scaled (array-like): The scaled coordinates to revert.

        Returns:
            np.array: Original geographic coordinates.
        """
        # Ensure input is a numpy array
        if not isinstance(X_scaled, np.ndarray):
            X_scaled = np.asarray(X_scaled)

        # Initialize the original array
        X_original = np.empty_like(X_scaled, dtype=float)

        # Unpack the ranges
        lon_min, lon_max = self.lon_range
        lat_min, lat_max = self.lat_range

        # Inverse transform longitudes
        X_original[:, 0] = (X_scaled[:, 0] - self.scale_min) / (
            self.scale_max - self.scale_min
        ) * (lon_max - lon_min) + lon_min

        # Inverse transform latitudes
        X_original[:, 1] = (X_scaled[:, 1] - self.scale_min) / (
            self.scale_max - self.scale_min
        ) * (lat_max - lat_min) + lat_min

        self.logger.debug(
            f"Inverse Transformed Coordinates: {X_original}, Shape: {X_original.shape}"
        )

        return X_original
