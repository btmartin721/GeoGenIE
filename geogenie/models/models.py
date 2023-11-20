import torch.nn as nn


class MLPRegressor(nn.Module):
    """Define PyTorch MLP Model."""

    def __init__(
        self,
        input_size,
        width=256,
        nlayers=10,
        dropout_prop=0.2,
        device="cpu",
        **kwargs,
    ):
        super(MLPRegressor, self).__init__()
        self.device = device
        self.input_layer = nn.Linear(input_size, width, device=self.device)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_prop)

        # Creating blocks of layers with residual connections
        self.blocks = nn.ModuleList()
        for _ in range(nlayers // 2):  # Handles even numbers of layers
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(width, width, device=self.device),
                    nn.BatchNorm1d(width, device=self.device),
                    nn.ELU(),
                    nn.Dropout(dropout_prop),
                    nn.Linear(width, width, device=self.device),
                    nn.BatchNorm1d(width, device=self.device),
                )
            )

        # Adding an additional layer if nlayers is odd
        self.extra_layer = None
        if nlayers % 2 != 0:
            self.extra_layer = nn.Sequential(
                nn.Linear(width, width, device=self.device),
                nn.BatchNorm1d(width, device=self.device),
                nn.ELU(),
                nn.Dropout(dropout_prop),
            )

        self.output_layer = nn.Linear(width, 2, device=self.device)

    def forward(self, x):
        """Forward pass through network."""
        x = self.elu(self.input_layer(x))
        # Applying residual blocks
        for block in self.blocks:
            residual = x
            # Add the residual (skip connection)
            x = self.elu(block(x) + residual)
        if self.extra_layer is not None:
            x = self.extra_layer(x)
        return self.output_layer(x)
