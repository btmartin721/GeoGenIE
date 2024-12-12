import logging

import numpy as np
import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """Define PyTorch MLP Model.

    This class defines a PyTorch Multi-Layer Perceptron (MLP) model for regression tasks.

    Attributes:
        input_size (int): The number of input features.
        width (int): The width of the hidden layers.
        nlayers (int): The number of hidden layers.
        dropout_prop (float): The dropout proportion.
        device (str): The device to run the model on.
        dtype (torch.dtype): The data type of the model.
        batch_size (int): The batch size for the model.
        seqmodel (torch.nn.Sequential): The sequential model
    """

    def __init__(
        self,
        input_size,
        width=256,
        nlayers=10,
        dropout_prop=0.25,
        device="cpu",
        output_width=2,
        dtype=torch.float32,
        batch_size=32,
    ):
        """Instantiate the MLPRegressor class.

        Args:
            input_size (int): The number of input features.
            width (int): The width of the hidden layers.
            nlayers (int): The number of hidden layers.
            dropout_prop (float): The dropout proportion.
            device (str): The device to run the model on.
            dtype (torch.dtype): The data type of the model.
            output_width (int): The number of output features.
            batch_size (int): The batch size for the model.
        """
        super(MLPRegressor, self).__init__()
        self.device = device
        self.dtype = dtype

        self.logger = logging.getLogger(__name__)

        initial_width = width
        if width >= input_size:
            self.logger.warning(
                "Provided hidden layer width is >= number of input features. Reducing initial layer width."
            )

        while width >= input_size:
            width *= 0.8
            width = int(width)

        if initial_width >= input_size:
            self.logger.warning(f"Reduced initial hidden layer width: {width}")

        self.seqmodel = self._define_model(
            input_size, width, nlayers, dropout_prop, output_width
        )

    def _define_model(self, input_size, width, nlayers, dropout_prop, output_width):
        """Method to define the neural network model.

        Args:
            input_size (int): The number of input features.
            width (int): The width of the hidden layers.
            nlayers (int): The number of hidden layers.
            dropout_prop (float): The dropout proportion.
            output_width (int): The number of output features.

        Returns:
            torch.nn.Sequential: The sequential model.
        """
        # Start with a Linear layer and BatchNorm1d
        layers = [
            nn.BatchNorm1d(input_size, dtype=self.dtype),
            nn.Linear(input_size, width, dtype=self.dtype),
            nn.ELU(),
        ]

        # Add the first half of the layers
        for _ in range(int(np.floor(nlayers / 2)) - 1):
            layers.append(nn.Linear(width, width, dtype=self.dtype))
            layers.append(nn.ELU())

        # Add dropout layer
        layers.append(nn.Dropout(dropout_prop))

        # Add the second half of the layers
        for _ in range(int(np.ceil(nlayers / 2))):
            layers.append(nn.Linear(width, width, dtype=self.dtype))
            layers.append(nn.ELU())

        # Add output layers
        layers.append(nn.Linear(width, output_width, dtype=self.dtype))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor to the neural network.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        if x.size(0) == 1:  # Check if batch size is 1
            return self.seqmodel[1:](x)  # Skip the first layer (BatchNorm)
        else:
            # Pass the input 'x' through the sequential model
            return self.seqmodel(x)
