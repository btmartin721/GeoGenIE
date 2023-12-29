import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import mean as torchmean
from torch_geometric.nn import GCNConv

from geogenie.utils.utils import base_to_int


class MLPRegressor(nn.Module):
    """Define PyTorch MLP Model."""

    def __init__(
        self,
        input_size,
        width=256,
        nlayers=10,
        dropout_prop=0.2,
        device="cpu",
        output_width=2,
        min_width=3,
        factor=0.5,
        **kwargs,
    ):
        super(MLPRegressor, self).__init__()
        self.device = device

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

        self.input_layer = nn.Linear(input_size, width, device=self.device)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_prop)

        if nlayers < 2:
            raise ValueError(
                "Number of layers must be at least 2 to calculate a scaling factor."
            )

        new_width = width
        old_width = width
        self.blocks = nn.ModuleList()
        for _ in range(nlayers // 2):  # Handles even numbers of layers
            new_width *= factor
            new_width = int(new_width)
            if new_width >= min_width:
                self.blocks.append(
                    nn.Sequential(
                        nn.Linear(old_width, new_width, device=self.device),
                        nn.BatchNorm1d(new_width, device=self.device),
                        nn.ELU(),
                        nn.Dropout(dropout_prop),
                        nn.Linear(new_width, new_width, device=self.device),
                        nn.BatchNorm1d(new_width, device=self.device),
                    )
                )
                old_width = new_width
                final_width = new_width
            else:
                final_width = old_width
                break

        # Adding an additional layer if nlayers is odd
        self.extra_layer = None
        if nlayers % 2 != 0:
            self.extra_layer = nn.Sequential(
                nn.Linear(final_width, final_width, device=self.device),
                nn.BatchNorm1d(final_width, device=self.device),
                nn.ELU(),
                nn.Dropout(dropout_prop),
            )
        self.output_layer = nn.Linear(final_width, output_width, device=device)

    def forward(self, x):
        """Forward pass through network."""
        x = self.elu(self.input_layer(x))
        for block in self.blocks:
            residual = x  # Storing residual
            x = block(x)  # Applying block

            # Adjust residual if necessary
            residual = self.adjust_residual(residual, block)
            x += residual
            x = self.elu(x)

        if self.extra_layer is not None:
            x = self.extra_layer(x)
        return self.output_layer(x)

    def adjust_residual(self, residual, block):
        """Adjust the residual to match the output dimension of the block."""
        # Assuming block[0] is the first Linear layer in the block
        out_features = block[0].out_features
        if residual.shape[-1] != out_features:
            # Dynamically adjust the residual adjuster to match the output dimensions
            self.residual_adjuster = nn.Linear(
                residual.shape[-1], out_features, device=self.device
            )
            residual = self.residual_adjuster(residual)
        return residual


class SNPTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        width=256,
        nlayers=3,
        dropout_prop=0.1,
        device="cpu",
        embedding_dim=256,
        nhead=8,
        dim_feedforward=1024,
        **kwargs,
    ):
        super(SNPTransformer, self).__init__()
        self.device = device
        self.d_model = embedding_dim
        num_bases = len(base_to_int())  # Number of unique bases
        self.embedding = nn.Embedding(
            num_embeddings=num_bases, embedding_dim=embedding_dim, device=device
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prop,
            device=self.device,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=nlayers
        )
        self.fc_out = nn.Linear(embedding_dim, 2, device=self.device)

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = src.permute(1, 0, 2)  # expects seq_len, batch, input_size
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        output = F.relu(output.mean(dim=1))  # Aggregating over the sequence
        return self.fc_out(output)


class GeoRegressionGNN(nn.Module):
    def __init__(
        self,
        input_size,
        width=16,
        nlayers=3,
        dropout_prop=0.1,
        device="cpu",
        **kwargs,
    ):
        super(GeoRegressionGNN, self).__init__()
        self.width = width
        self.nlayers = nlayers
        self.dropout_prop = dropout_prop
        self.device = device
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_size, width, device=self.device))

        for _ in range(nlayers - 1):
            self.layers.append(GCNConv(width, width, device=self.device))

        self.fc = nn.Linear(width, 2, device=self.device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer in self.layers:
            x = F.elu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout_prop, training=self.training)

        x = torchmean(x, dim=0)
        return self.fc(x)
