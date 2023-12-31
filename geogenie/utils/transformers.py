import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_array, check_is_fitted

logger = logging.getLogger(__name__)


class MCA(BaseEstimator, TransformerMixin):
    """Class to perform Multiple Correspondence Analayis (MCA).

    Attributes:
        n_components (int): Number of MCA components to output.
        n_iter (int): Number of randomized SVD iterations to perform.
        check_input (bool): Whether to check input data for conformity.
        random_state (int or None): Random state for reproducibility.
        one_hot (bool): Flag for one-hot encoding the input data.
        categories (list): Possible categories in input features.
        epsilon (float): Small value to prevent division by 0.
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
        self.n_components = n_components
        self.n_iter = n_iter
        self.check_input = check_input
        self.random_state = random_state
        self.one_hot = one_hot
        self.categories = categories
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Fit the input data."""
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
        """Transform input data X using MCA."""
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
        return transformed_X

    def _normalize_data(self, X):
        X_normalized = X.astype(float) / X.sum()
        row_sums = X_normalized.sum(axis=1) + self.epsilon
        col_sums = X_normalized.sum(axis=0) + self.epsilon
        self.row_sums_, self.col_sums_ = row_sums, col_sums
        return X_normalized

    def _compute_S_matrix(self, X):
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
        self.eigenvalues_ = np.square(self.Sigma_)
        total_variance = np.sum(self.eigenvalues_)
        self.explained_inertia_ = self.eigenvalues_ / total_variance
        self.cumulative_inertia_ = np.cumsum(self.explained_inertia_)


class MortonCurveTransformer(BaseEstimator, TransformerMixin):
    """A transformer for encoding and decoding geographical coordinates (longitude, latitude) into a single integer using a Morton curve  (Z-order curve).

    Args:
        bits (int): The number of bits to use for encoding each of the longitude and latitude. Defaults to 16.

    Methods:
        fit(X, y=None): Fit the transformer on the data.
        transform(X): Transform the data using the Morton curve encoding.
        inverse_transform(X): Inverse transform the Morton curve encoded data.
        _interleave_bits(x, y): Interleave the bits of x and y.
        _deinterleave_bits(z): De-interleave the bits of z into x and y.
        _apply_interleave(longitude, latitude): Apply interleaving to longitude and latitude.
        _decode_long_lat(encoded_value): Decode an encoded value back to longitude and latitude.
    """

    def __init__(self, bits=16):
        """Initialize the MortonCurveTransformer with the specified number of bits for encoding.

        Args:
            bits (int): The number of bits to use for encoding each coordinate.
        """
        self.bits = bits
        self.max_long = None
        self.max_lat = None

    def fit(self, X, y=None):
        """Fit the transformer on the data. This method calculates and stores the maximum absolute values of longitude and latitude, which are used for normalization.

        Args:
            X (pd.DataFrame or np.ndarray): The data containing longitude and latitude columns, respectively.
            y (ignored): Not used, present here for compatibility with scikit-learn's fit method.

        Returns:
            MortonCurveTransformer: The fitted transformer.
        """
        # Extract longitude and latitude, then store scaling parameters
        if isinstance(X, pd.DataFrame):
            longitudes = X.iloc[:, 0]
            latitudes = X.iloc[:, 1]
        elif isinstance(X, np.ndarray):
            longitudes = X[:, 0]
            latitudes = X[:, 1]
        else:
            msg = f"Input must be a pandas DataFrame or a numpy array: {type(X)}"
            logger.error(msg)
            raise TypeError(msg)

        self.max_long = np.max(np.abs(longitudes))
        self.max_lat = np.max(np.abs(latitudes))
        return self

    def transform(self, X):
        """
        Transform the data using Morton curve encoding. This method encodes each pair of
        longitude and latitude coordinates into a single integer.

        Args:
            X (pd.DataFrame or np.ndarray): The data containing longitude and latitude columns.

        Returns:
            pd.Series or np.ndarray: The encoded data, where each pair of coordinates is represented by a single integer.
        """
        # Check if X is a DataFrame or NumPy array
        if isinstance(X, pd.DataFrame):
            X_encoded = X.apply(
                lambda row: self._apply_interleave(row.iloc[0], row.iloc[1]), axis=1
            )
        elif isinstance(X, np.ndarray):
            X_encoded = np.apply_along_axis(
                lambda row: self._apply_interleave(row[0], row[1]), 1, X
            )
        else:
            msg = msg = f"Input must be a pandas DataFrame or a numpy array: {type(X)}"
            logger.error(msg)
            raise TypeError(msg)
        return X_encoded

    def inverse_transform(self, X_encoded):
        """
        Inverse transform the Morton curve encoded data. This method decodes each integer back into a pair of longitude and latitude coordinates.

        Args:
            X_encoded (pd.Series or np.ndarray): The encoded data with Morton curve integers.

        Returns:
            np.ndarray: The decoded data, with each row containing a pair of longitude and latitude.
        """
        if isinstance(X_encoded, pd.Series) or isinstance(X_encoded, np.ndarray):
            func = np.vectorize(self._decode_long_lat)
            return np.array(func(X_encoded)).T
        else:
            raise TypeError("Input must be a pandas Series or a NumPy array")

    def _interleave_bits(self, x, y):
        """
        Interleave the bits of two integers. Used in the Morton curve encoding process.

        Args:
            x (int): The first integer (representing encoded longitude).
            y (int): The second integer (representing encoded latitude).

        Returns:
            int: The interleaved integer representing the Morton curve encoded value.
        """
        z = 0
        for i in range(self.bits):
            z |= (x & (1 << i)) << i | (y & (1 << i)) << (i + 1)
        return z

    def _deinterleave_bits(self, z):
        """
        De-interleave the bits of an integer into two integers. Used in the Morton curve decoding process.

        Args:
            z (int): The interleaved integer representing the Morton curve encoded value.

        Returns:
            int, int: The two de-interleaved integers representing longitude and latitude.
        """
        x = y = 0
        for i in range(self.bits):
            x |= (z & (1 << (2 * i))) >> i
            y |= (z & (1 << (2 * i + 1))) >> (i + 1)
        return x, y

    def _apply_interleave(self, longitude, latitude):
        """
        Apply interleaving to longitude and latitude to encode them into a single integer.

        Args:
            longitude (float): The longitude value.
            latitude (float): The latitude value.

        Returns:
            int: The Morton curve encoded value of the longitude and latitude.
        """
        # Normalize using stored max values
        norm_long = (longitude + self.max_long) / (2 * self.max_long)
        norm_lat = (latitude + self.max_lat) / (2 * self.max_lat)
        int_long = int(norm_long * (2**self.bits))
        int_lat = int(norm_lat * (2**self.bits))
        return self._interleave_bits(int_long, int_lat)

    def _decode_long_lat(self, encoded_value):
        """
        Decode an encoded value back to longitude and latitude.

        Args:
            encoded_value (int or float): The Morton curve encoded value.

        Returns:
            float, float: The decoded longitude and latitude values.
        """
        # Ensure encoded_value is an integer
        encoded_value = int(encoded_value)
        int_long, int_lat = self._deinterleave_bits(encoded_value)
        norm_long = int_long / (2**self.bits)
        norm_lat = int_lat / (2**self.bits)
        longitude = norm_long * (2 * self.max_long) - self.max_long
        latitude = norm_lat * (2 * self.max_lat) - self.max_lat
        return longitude, latitude
