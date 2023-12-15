import logging
import math
import sys

import torch
import torch.nn as nn


class MultiobjectiveHaversineLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, composite_loss=False):
        super(MultiobjectiveHaversineLoss, self).__init__()
        self.alpha = alpha  # Weight for Haversine loss
        self.beta = beta  # Weight for mean and stddev
        self.gamma = gamma  # Weight for R-squared
        self.logger = logging.getLogger(__name__)
        self.composite_loss = composite_loss

    def haversine_loss(self, predictions, targets, sample_weight):
        R = 6371.0  # Earth radius in kilometers
        lat1, lon1 = torch.deg2rad(targets[:, 1]), torch.deg2rad(targets[:, 0])
        lat2, lon2 = torch.deg2rad(predictions[:, 1]), torch.deg2rad(predictions[:, 0])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            torch.sin(dlat / 2.0) ** 2
            + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
        )
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        distance = R * c
        weighted_distance = distance * sample_weight
        # weighted_distance = torch.log1p(weighted_distance)  # lognormal
        return weighted_distance.mean()

    def pearson_correlation(self, x, y):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x - mean_x
        ym = y - mean_y
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm**2) * torch.sum(ym**2))
        r = r_num / r_den

        # To handle cases where r_den is very small or zero leading to NaNs
        r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
        return r

    def robust_scale(self, tensor):
        median = torch.median(tensor)
        q75, q25 = torch.quantile(tensor, 0.75), torch.quantile(tensor, 0.25)
        iqr = q75 - q25

        if iqr <= 0:
            iqr = 1e-3

        scaled = (tensor - median) / iqr
        return scaled

    def forward(self, predictions, targets, sample_weight):
        lon_pred, lat_pred = predictions[:, 0], predictions[:, 1]
        lon_true, lat_true = targets[:, 0], targets[:, 1]

        if sample_weight is None:
            self.logger.warning(
                "sample_weight was NoneType. Setting all weights to 1.0"
            )
            sample_weight = torch.ones(len(predictions))

        transformed_haversine_loss = self.haversine_loss(
            predictions, targets, sample_weight
        )

        if self.composite_loss:
            # Mean and standard deviation for longitude and latitude
            # Returns a tuple of (stddev, mean).
            std_lon, mean_lon = torch.std_mean(lon_pred)
            std_lat, mean_lat = torch.std_mean(lat_pred)

            # Robust scaling of the mean and standard deviation
            scaled_mean_lon = self.robust_scale(mean_lon)
            scaled_std_lon = self.robust_scale(std_lon)
            scaled_mean_lat = self.robust_scale(mean_lat)
            scaled_std_lat = self.robust_scale(std_lat)

            # R-squared for longitude and latitude
            r_squared_lon = self.pearson_correlation(lon_pred, lon_true) ** 2
            r_squared_lat = self.pearson_correlation(lat_pred, lat_true) ** 2

            # Combine the metrics for longitude and latitude
            # Note: You might want to transform these as well, depending on your specific requirements
            custom_loss_lon = (
                self.beta * (scaled_mean_lon + scaled_std_lon)
                - self.gamma * r_squared_lon
            )
            custom_loss_lat = (
                self.beta * (scaled_mean_lat + scaled_std_lat)
                - self.gamma * r_squared_lat
            )

            # Combine transformed Haversine loss with other components
            total_loss = (
                self.alpha * transformed_haversine_loss
                + (1 - self.alpha) * (custom_loss_lon + custom_loss_lat) / 2
            )

        else:
            total_loss = transformed_haversine_loss
        return total_loss


class HybridHaversineLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Initializes the custom Haversine loss class for PyTorch.

        :param alpha: Float, weighting factor for combining Haversine loss and standard deviation.
        """
        super(HybridHaversineLoss, self).__init__()
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculates the Haversine distance between two sets of points.

        :param lat1, lon1: Latitude and longitude of the first set of points.
        :param lat2, lon2: Latitude and longitude of the second set of points.
        :return: Haversine distance.
        """
        R = 6371  # Earth radius in kilometers

        phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
        delta_phi = torch.deg2rad(lat2 - lat1)
        delta_lambda = torch.deg2rad(lon2 - lon1)

        a = (
            torch.sin(delta_phi / 2) ** 2
            + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda / 2) ** 2
        )
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        return R * c

    def _weighted_haversine_loss(self, y_true, y_pred, sample_weight):
        """
        Calculates the weighted Haversine loss.

        :param y_true: Tensor of true values (latitude, longitude).
        :param y_pred: Tensor of predicted values (latitude, longitude).
        :param sample_weight: Tensor of sample weights.
        :return: Weighted Haversine loss.
        """
        lon_true, lat_true = y_true[:, 0], y_true[:, 1]
        lon_pred, lat_pred = y_pred[:, 0], y_pred[:, 1]

        haversine_loss = self._haversine_distance(
            lat_true, lon_true, lat_pred, lon_pred
        )
        weighted_loss = haversine_loss * sample_weight

        # Returns reduction of lon, lat separately.
        # Calculates both std dev and mean, returns tuple in that order.
        return weighted_loss.std_mean(weighted_loss, dim=0)

    def forward(self, predictions, targets, sample_weight=None):
        """
        Calculates the custom Haversine loss metric in PyTorch.

        :param targets: Tensor of true values.
        :param predictions: Tensor of predicted values.
        :param sample_weight: Tensor of sample weights. If None, equal weighting is assumed.
        :return: Custom Haversine loss metric value.
        """
        if not isinstance(predictions, torch.Tensor):
            msg = f"Expected predictions to be a torch.Tensor, but got {type(predictions)}"
            self.logger.error(msg)
            raise TypeError(msg)
        if not isinstance(targets, torch.Tensor):
            msg = f"Expected targets to be a torch.Tensor, but got {type(targets)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if sample_weight is None:
            sample_weight = torch.ones_like(targets)

        sample_weight = sample_weight.view(1, -1)

        weighted_std_dev, weighted_mean = self._weighted_haversine_loss(
            targets, predictions, sample_weight
        )

        return self.alpha * weighted_mean + (1 - self.alpha) * weighted_std_dev


class HybridRMSELoss(nn.Module):
    def __init__(self, use_mae=False, alpha=0.5):
        """
        Initializes the custom error metric class for PyTorch.

        :param use_mae: Boolean, if True use MAE, else use RMSE.
        :param alpha: Float, weighting factor for combining MAE/RMSE and standard deviation.
        """
        super(HybridRMSELoss, self).__init__()

        self.use_mae = use_mae
        self.alpha = alpha

    def _weighted_error(self, y_true, y_pred, sample_weight):
        """
        Calculates the weighted error (either MAE or RMSE) in PyTorch.

        :param y_true: Tensor of true values.
        :param y_pred: Tensor of predicted values.
        :param sample_weight: Tensor of sample weights.
        :return: Weighted MAE or RMSE.
        """
        error = y_pred - y_true
        error = error.mean(dim=1)
        weighted_error = error * sample_weight
        if self.use_mae:
            return torch.mean(torch.abs(weighted_error))
        else:
            return torch.sqrt(torch.mean(weighted_error**2))

    def _weighted_std_dev(self, y_true, y_pred, sample_weight):
        """
        Calculates the weighted standard deviation of the errors in PyTorch.

        :param y_true: Tensor of true values.
        :param y_pred: Tensor of predicted values.
        :param sample_weight: Tensor of sample weights.
        :return: Weighted standard deviation of errors.
        """
        error = y_pred - y_true
        error = error.mean(dim=1)
        weighted_error = error * sample_weight
        mean_weighted_error = torch.mean(weighted_error)
        variance = torch.mean((weighted_error - mean_weighted_error) ** 2)
        return torch.sqrt(variance)

    def forward(self, predictions, targets, sample_weight=None):
        """
        Calculates the custom error metric in PyTorch.

        :param targets: Tensor of true values.
        :param predictions: Tensor of predicted values.
        :param sample_weight: Tensor of sample weights. If None, equal weighting is assumed.
        :return: Custom error metric value.
        """
        # Ensure predictions and targets are tensors
        if not isinstance(predictions, torch.Tensor):
            raise TypeError(
                f"Expected predictions to be a torch.Tensor, but got {type(predictions)}"
            )
        if not isinstance(targets, torch.Tensor):
            raise TypeError(
                f"Expected targets to be a torch.Tensor, but got {type(targets)}"
            )

        if sample_weight is None:
            sample_weight = torch.ones(len(targets))

        weighted_error = self._weighted_error(targets, predictions, sample_weight)
        weighted_std_dev = self._weighted_std_dev(targets, predictions, sample_weight)

        return self.alpha * weighted_error + (1 - self.alpha) * weighted_std_dev


class WeightedRMSELoss(nn.Module):
    def __init__(self):
        super(WeightedRMSELoss, self).__init__()

    def forward(self, predictions, targets, sample_weight=None):
        """
        Calculate the Weighted Root Mean Squared Error.

        Args:
            predictions (Tensor): Predicted values with shape (batch_size, 2).
            targets (Tensor): Actual values with shape (batch_size, 2).
            sample_weight (Tensor, optional): Weights for each sample in the batch. Shape should be (batch_size,).

        Returns:
            Tensor: Weighted Root Mean Squared Error.
        """
        # Ensure predictions and targets are tensors
        if not isinstance(predictions, torch.Tensor):
            raise TypeError(
                f"Expected predictions to be a torch.Tensor, but got {type(predictions)}"
            )
        if not isinstance(targets, torch.Tensor):
            raise TypeError(
                f"Expected targets to be a torch.Tensor, but got {type(targets)}"
            )

        # Calculating the squared difference between targets and predictions
        loss = (predictions - targets) ** 2

        # If sample_weight is provided, ensure it's a tensor and apply it
        if sample_weight is not None:
            if not isinstance(sample_weight, torch.Tensor):
                sample_weight = torch.tensor(
                    sample_weight, dtype=loss.dtype, device=loss.device
                )
            sample_weight = sample_weight.view(-1, 1)
            loss = loss * sample_weight

        # Mean over all dimensions except the batch dimension
        loss = loss.mean(dim=1)

        # Finally, compute the square root
        return torch.sqrt(loss.mean())


class WeightedHaversineLoss(nn.Module):
    def __init__(self, eps=1e-6, earth_radius=6371):
        super(WeightedHaversineLoss, self).__init__()
        self.eps = eps
        self.earth_radius = earth_radius
        self.rad_factor = math.pi / 180  # radians conversion factor

    def forward(self, predictions, targets, sample_weight=None):
        predictions = predictions * self.rad_factor
        targets = targets * self.rad_factor

        dlon = predictions[:, 0] - targets[:, 0]
        dlat = predictions[:, 1] - targets[:, 1]
        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(targets[:, 0])
            * torch.cos(predictions[:, 0])
            * torch.sin(dlon / 2) ** 2
        )
        c = 2 * torch.atan2(torch.sqrt(a + self.eps), torch.sqrt(1 - a + self.eps))
        loss = self.earth_radius * c
        loss = loss.view(-1, 1)

        if sample_weight is not None:
            sample_weight = sample_weight.view(-1, 1)
            loss *= sample_weight

        return loss.mean()
