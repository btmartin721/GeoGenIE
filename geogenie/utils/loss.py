import warnings

import torch
import torch.nn as nn


def _weighted_mean(values, sample_weight=None):
    """Mean of per-sample values, weighted by ``sample_weight`` if given.

    Args:
        values (torch.Tensor): Per-sample values, shape (n_samples,).
        sample_weight (torch.Tensor, optional): Per-sample weights, shape (n_samples,) or (n_samples, 1). Defaults to None (unweighted mean).

    Returns:
        torch.Tensor: Scalar weighted mean, ``sum(values * w) / sum(w)``.
    """
    if sample_weight is None:
        return values.mean()
    sample_weight = sample_weight.view(-1).to(values.dtype)
    return (values * sample_weight).sum() / sample_weight.sum()


class WeightedDRMSLoss(nn.Module):
    """Custom loss class to compute the Distance Root Mean Square (DRMS) for
    longitude and latitude coordinates.

    The loss is computed on the coordinates passed to it, which during training are the normalised targets produced by the target scaler (``MinMaxScalerGeo``). Haversine distance is only meaningful for coordinates in degrees, so the distance used here is the Euclidean distance in the normalised space. With a distance-preserving target scaler (longitude scaled by cos(mean latitude) and one common scale factor for both axes) this is proportional to the distance in km. The returned value is ``sqrt(sum(w * d^2) / sum(w))``.

    Attributes:
        radius (float): Kept for backwards compatibility; not used.
    """

    def __init__(self, radius=6371):
        """
        Initializes the WeightedDRMSLoss class.

        Args:
            radius (float): Kept for backwards compatibility; not used, because the loss is computed in normalised coordinate space. Default is 6371.
        """
        super(WeightedDRMSLoss, self).__init__()
        self.radius = radius

    def forward(self, preds, targets, sample_weight=None):
        """
        Forward pass to compute the Distance Root Mean Square (DRMS) loss.

        Args:
            preds (torch.Tensor): Predicted (normalised) longitude and latitude coordinates.
            targets (torch.Tensor): Actual (normalised) longitude and latitude coordinates.
            sample_weight (torch.Tensor): Sample weights to make some samples more or less important than others. Defaults to None.

        Returns:
            torch.Tensor: DRMS loss, in normalised coordinate units.
        """
        if preds.shape != targets.shape:
            raise ValueError("Predictions and targets must have the same shape")

        squared_dist = torch.sum((preds - targets) ** 2, dim=1)
        return torch.sqrt(_weighted_mean(squared_dist, sample_weight))


def weighted_rmse_loss(y_true, y_pred, sample_weight=None):
    """Custom PyTorch weighted mean Euclidean distance loss function.

    Despite its name (kept for backwards compatibility), this computes the (weighted) mean of the per-sample Euclidean distances between predictions and ground truth, not the root of the mean squared error: ``sum(w * d) / sum(w)`` with ``d = ||y_pred - y_true||``. The distance is in the units of the inputs (normalised coordinates during training).

    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predictions.
        sample_weight (torch.Tensor): Sample weights (1-dimensional).

    Returns:
        torch.Tensor: Weighted mean Euclidean distance.
    """
    dist = torch.sqrt(torch.sum((y_pred - y_true) ** 2, axis=1))
    if sample_weight is not None:
        # Weight each sample's error, then take the weighted mean.
        return (dist * sample_weight).sum() / sample_weight.sum()
    return dist.mean()


class WeightedHuberLoss(nn.Module):
    """Custom loss class to compute the Weighted Huber Loss.

    The Huber loss is applied to each coordinate error. ``delta`` is in the units of the targets passed to the loss, which during training are the normalised coordinates produced by the target scaler, not degrees or km.

    Attributes:
        delta (float): The threshold for the Huber loss, in normalised target units. Default is 1.0.
        smoothing_factor (float): Deprecated and ignored. Default is 0.0.
    """

    def __init__(self, delta=1.0, smoothing_factor=0.0):
        """Instantiate the WeightedHuberLoss class.

        Args:
            delta (float, optional): Threshold between the quadratic and linear parts of the loss, in normalised target units. Defaults to 1.0.
            smoothing_factor (float, optional): Deprecated and ignored. Earlier versions blended the targets toward the batch mean, which biased predictions toward the centroid of the training data. Defaults to 0.0.
        """
        super(WeightedHuberLoss, self).__init__()
        self.delta = delta
        self.smoothing_factor = smoothing_factor
        if smoothing_factor:
            warnings.warn(
                "WeightedHuberLoss 'smoothing_factor' is deprecated and ignored; targets are not smoothed.",
                UserWarning,
            )

    def forward(self, input, target, sample_weight=None):
        """Forward pass to compute the Weighted Huber Loss."""
        assert (
            input.shape == target.shape
        ), f"Shape mismatch: {input.shape} vs {target.shape}"
        if torch.any(torch.isnan(input)) or torch.any(torch.isnan(target)):
            if torch.any(torch.isnan(input)) and torch.any(torch.isnan(target)):
                raise ValueError("Both the input and target contain NaN values")

            if torch.any(torch.isnan(input)):
                raise ValueError("Inputs contain NaN values")

            if torch.any(torch.isnan(target)):
                raise ValueError("Targets contain NaN values")

        if torch.any(torch.isinf(input)) or torch.any(torch.isinf(target)):
            if torch.any(torch.isinf(input)) and torch.any(torch.isinf(target)):
                raise ValueError("Both the input and target contain INF values")

            if torch.any(torch.isinf(input)):
                raise ValueError("Inputs contain INF values")

            if torch.any(torch.isinf(target)):
                raise ValueError("Targets contain INF values")

        error = torch.abs(input - target)
        is_small_error = error < self.delta

        small_error_loss = 0.5 * (error**2)
        large_error_loss = self.delta * (error - 0.5 * self.delta)

        loss = torch.where(is_small_error, small_error_loss, large_error_loss)

        if sample_weight is not None:
            if torch.any(torch.isnan(sample_weight)) or torch.any(
                torch.isinf(sample_weight)
            ):
                raise ValueError("Sample weight contains NaN or Inf values")
        # Mean over coordinates, then (weighted) mean over samples.
        return _weighted_mean(loss.mean(dim=1), sample_weight)
