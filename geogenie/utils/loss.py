import torch
import torch.nn as nn


class WeightedDRMSLoss(nn.Module):
    """Custom loss class to compute the Distance Root Mean Square (DRMS) for
    longitude and latitude coordinates.

    Attributes:
        radius (float): Radius of the Earth in kilometers. Default is 6371 km.
    """

    def __init__(self, radius=6371):
        """
        Initializes the WeightedDRMSLoss class with the Earth's radius.

        Args:
            radius (float): Radius of the Earth in kilometers. Default is 6371 km.
        """
        super(WeightedDRMSLoss, self).__init__()
        self.radius = radius

    def forward(self, preds, targets, sample_weight=None):
        """
        Forward pass to compute the Distance Root Mean Square (DRMS) loss.

        Args:
            preds (torch.Tensor): Predicted longitude and latitude coordinates.
            targets (torch.Tensor): Actual longitude and latitude coordinates.
            sample_weight (torch.Tensor): Sample weights to make some samples more or less important than others. Defaults to None.

        Returns:
            torch.Tensor: DRMS loss.
        """
        if preds.shape != targets.shape:
            raise ValueError("Predictions and targets must have the same shape")

        lon1, lat1 = preds[:, 0], preds[:, 1]
        lon2, lat2 = targets[:, 0], targets[:, 1]

        lon1, lat1, lon2, lat2 = map(torch.deg2rad, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(
            dlon / 2
        ).pow(2)
        c = 2 * torch.asin(torch.clamp(torch.sqrt(a), min=-1, max=1))

        km_dist = self.radius * c
        squared_dist = km_dist.pow(2)

        if sample_weight is not None:
            weighted_squared_dist = squared_dist * sample_weight
            mean_weighted_squared_dist = torch.mean(weighted_squared_dist)
        else:
            mean_weighted_squared_dist = torch.mean(squared_dist)

        return torch.sqrt(mean_weighted_squared_dist)


def weighted_rmse_loss(y_true, y_pred, sample_weight=None):
    """Custom PyTorch weighted RMSE loss function.

    This method computes the weighted RMSE loss between the ground truth and predictions.

    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predictions.
        sample_weight (torch.Tensor): Sample weights (1-dimensional).

    Returns:
        float: Weighted RMSE loss.
    """
    rmse = torch.sqrt(torch.sum((y_pred - y_true) ** 2, axis=1)).mean()
    if sample_weight is not None:
        rmse_scaled = rmse * sample_weight
        return rmse_scaled.mean()
    return rmse


class WeightedHuberLoss(nn.Module):
    """Custom loss class to compute the Weighted Huber Loss.

    Attributes:
        delta (float): The threshold for the Huber loss. Default is 1.0.
        smoothing_factor (float): The smoothing factor for the target. Default is 0.1.
    """

    def __init__(self, delta=1.0, smoothing_factor=0.1):
        """Instantiate the WeightedHuberLoss class.

        Args:
            delta (float, optional): _description_. Defaults to 1.0.
            smoothing_factor (float, optional): _description_. Defaults to 0.1.
        """
        super(WeightedHuberLoss, self).__init__()
        self.delta = delta
        self.smoothing_factor = smoothing_factor

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

        target_smoothed = (
            1 - self.smoothing_factor
        ) * target + self.smoothing_factor * torch.mean(target, dim=0)

        error = torch.abs(input - target_smoothed)
        is_small_error = error < self.delta

        small_error_loss = 0.5 * (error**2)
        large_error_loss = self.delta * (error - 0.5 * self.delta)

        loss = torch.where(is_small_error, small_error_loss, large_error_loss)

        if sample_weight is not None:
            if torch.any(torch.isnan(sample_weight)) or torch.any(
                torch.isinf(sample_weight)
            ):
                raise ValueError("Sample weight contains NaN or Inf values")
            sample_weight = sample_weight.view(-1, 1)
            weighted_loss = loss * sample_weight
            return weighted_loss.mean()
        return loss.mean()
