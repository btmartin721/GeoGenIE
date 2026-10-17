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

    The scaling is fitted on the data passed to ``fit`` (the training coordinates). With ``isotropic=True`` (default),
    longitude is first multiplied by cos(mean latitude) and both axes share one scale factor, so Euclidean distance in
    the scaled space is proportional to distance on the ground; the longer axis spans [scale_min, scale_max]. With
    ``isotropic=False`` each axis is min-max scaled to [scale_min, scale_max] separately.

    Attributes:
        lat_range (tuple): Minimum and maximum latitude seen in fit.
        lon_range (tuple): Minimum and maximum longitude seen in fit.
        scale_min (float): Minimum value of the scaled range.
        scale_max (float): Maximum value of the scaled range.
        isotropic (bool): Whether both axes share one distance-preserving scale.
        logger (logging.Logger): Logger object for the class.
    """

    def __init__(
        self,
        lat_range=(-90, 90),
        lon_range=(-180, 180),
        scale_min=0,
        scale_max=1,
        isotropic=True,
    ):
        """Initialize the MinMaxScalerGeo.

        Args:
            lat_range (tuple): Initial latitude range; replaced by the data range in fit.
            lon_range (tuple): Initial longitude range; replaced by the data range in fit.
            scale_min (float): Minimum value of the scaled range.
            scale_max (float): Maximum value of the scaled range.
            isotropic (bool): Use one distance-preserving scale for both axes. Default: True.
        """
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.isotropic = isotropic

        self.logger = logging.getLogger(__name__)
        self._set_axis_spans()

    def _set_axis_spans(self):
        """Compute the per-axis multipliers and spans used by transform and inverse_transform."""
        lon_min, lon_max = self.lon_range
        lat_min, lat_max = self.lat_range
        if self.isotropic:
            self.lon_factor_ = np.cos(np.radians((lat_min + lat_max) / 2))
            span = max((lon_max - lon_min) * self.lon_factor_, lat_max - lat_min)
            self.lon_span_ = span / self.lon_factor_
            self.lat_span_ = span
        else:
            self.lon_factor_ = 1.0
            self.lon_span_ = lon_max - lon_min
            self.lat_span_ = lat_max - lat_min

    def fit(self, X, y=None):
        """Set the longitude and latitude ranges from the data.

        Args:
            X (array-like): Coordinates to fit, shape (n_samples, 2), columns (longitude, latitude).
            y (None, optional): Ignored. This parameter exists only for compatibility with the sklearn API.

        Returns:
            self: Returns the instance itself.
        """
        X = np.asarray(X, dtype=float)
        lon_min, lon_max = np.nanmin(X[:, 0]), np.nanmax(X[:, 0])
        lat_min, lat_max = np.nanmin(X[:, 1]), np.nanmax(X[:, 1])

        # Avoid a zero-width range if all samples share a coordinate.
        if lon_max - lon_min == 0:
            lon_min, lon_max = lon_min - 0.5, lon_max + 0.5
        if lat_max - lat_min == 0:
            lat_min, lat_max = lat_min - 0.5, lat_max + 0.5

        self.lon_range = (lon_min, lon_max)
        self.lat_range = (lat_min, lat_max)
        self._set_axis_spans()
        return self

    def transform(self, X):
        """Scale geographic coordinates.

        Args:
            X (array-like): Coordinates, shape (n_samples, 2), columns (longitude, latitude).

        Returns:
            np.array: Scaled coordinates.
        """
        X = np.asarray(X, dtype=float)
        width = self.scale_max - self.scale_min
        X_scaled = np.empty_like(X, dtype=float)
        X_scaled[:, 0] = (X[:, 0] - self.lon_range[0]) / self.lon_span_ * width + self.scale_min
        X_scaled[:, 1] = (X[:, 1] - self.lat_range[0]) / self.lat_span_ * width + self.scale_min

        self.logger.debug(
            f"Transformed Coordinates: {X_scaled}, Shape: {X_scaled.shape}"
        )
        return X_scaled

    def inverse_transform(self, X_scaled):
        """Scale coordinates back to longitude and latitude.

        Args:
            X_scaled (array-like): Scaled coordinates.

        Returns:
            np.array: Coordinates, columns (longitude, latitude).
        """
        X_scaled = np.asarray(X_scaled, dtype=float)
        width = self.scale_max - self.scale_min
        X_original = np.empty_like(X_scaled, dtype=float)
        X_original[:, 0] = (X_scaled[:, 0] - self.scale_min) / width * self.lon_span_ + self.lon_range[0]
        X_original[:, 1] = (X_scaled[:, 1] - self.scale_min) / width * self.lat_span_ + self.lat_range[0]

        self.logger.debug(
            f"Inverse Transformed Coordinates: {X_original}, Shape: {X_original.shape}"
        )
        return X_original
