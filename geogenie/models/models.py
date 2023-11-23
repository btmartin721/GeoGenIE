import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch import mean as torchmean
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
