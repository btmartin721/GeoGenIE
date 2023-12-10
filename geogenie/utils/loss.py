import math

import torch
import torch.nn as nn


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
            loss *= sample_weight

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
