"""Spatial Graph Attention Network autoencoder for spatial transcriptomics.

Implements a PyTorch Geometric GAT autoencoder that learns spatial-aware
latent embeddings from Visium spot expression profiles, using the tissue
spatial connectivity graph as message-passing topology.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

__all__ = [
    "prepare_pyg_data",
    "SpatialGATAutoencoder",
    "train_gat_autoencoder",
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_pyg_data(
    adata: ad.AnnData,
    feature_key: str = "highly_variable",
) -> Data:
    """Convert an AnnData object into a PyG :class:`Data` graph.

    Extracts the expression matrix (optionally subsetted to highly
    variable genes) as node features and the spatial connectivity
    matrix (``adata.obsp['spatial_connectivities']``) as edge indices.

    Args:
        adata: AnnData with ``obsp['spatial_connectivities']``
            (run ``squidpy.gr.spatial_neighbors`` first) and, when
            *feature_key* is ``"highly_variable"``, a boolean
            ``var["highly_variable"]`` column.
        feature_key: ``"highly_variable"`` to subset to HVGs, or
            ``"all"`` to use every gene.

    Returns:
        A :class:`torch_geometric.data.Data` object with ``x``
        (node features) and ``edge_index`` (COO edge tensor).

    Raises:
        KeyError: If the spatial connectivity matrix or HVG
            annotation is missing.
    """
    if "spatial_connectivities" not in adata.obsp:
        raise KeyError(
            "adata.obsp['spatial_connectivities'] not found. "
            "Run squidpy.gr.spatial_neighbors() first."
        )

    # --- node features ---
    if feature_key == "highly_variable":
        if "highly_variable" not in adata.var.columns:
            raise KeyError(
                "adata.var['highly_variable'] not found. "
                "Run sc.pp.highly_variable_genes() first."
            )
        hvg_mask = adata.var["highly_variable"].values
        X = adata.X[:, hvg_mask]
    else:
        X = adata.X

    if sp.issparse(X):
        X = X.toarray()

    x = torch.tensor(X, dtype=torch.float32)

    # --- edge index from spatial adjacency ---
    adj = adata.obsp["spatial_connectivities"]
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    coo = adj.tocoo()
    edge_index = torch.tensor(
        np.stack([coo.row, coo.col], axis=0),
        dtype=torch.long,
    )

    return Data(x=x, edge_index=edge_index)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SpatialGATAutoencoder(nn.Module):
    """Two-layer GAT encoder with a linear decoder for reconstruction.

    The encoder passes node features through two :class:`GATConv`
    layers with ELU activation, producing a low-dimensional latent
    embedding.  The decoder reconstructs the original features via a
    single linear projection, trained with MSE loss.

    Args:
        in_channels: Dimensionality of input node features.
        hidden_channels: Width of the first GAT hidden layer.
        out_channels: Dimensionality of the latent space.
        heads: Number of attention heads in the first GAT layer.
        dropout: Dropout probability applied during training.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 30,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        # Encoder
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        # Decoder
        self.decoder = nn.Linear(out_channels, in_channels)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        """Run the two-layer GAT encoder.

        Args:
            x: Node feature matrix ``[N, in_channels]``.
            edge_index: COO edge index ``[2, E]``.
            return_attention_weights: If ``True``, return GAT attention
                weights from both layers.

        Returns:
            Tuple of ``(z, attn_weights)`` where *z* is the latent
            embedding ``[N, out_channels]`` and *attn_weights* is
            ``None`` or a list of attention-weight tuples.
        """
        attn_weights: list[Any] = []

        if return_attention_weights:
            h, attn1 = self.gat1(
                x, edge_index, return_attention_weights=True,
            )
            h = torch.nn.functional.elu(h)
            h = torch.nn.functional.dropout(
                h, p=self.dropout, training=self.training,
            )
            z, attn2 = self.gat2(
                h, edge_index, return_attention_weights=True,
            )
            attn_weights = [attn1, attn2]
        else:
            h = self.gat1(x, edge_index)
            h = torch.nn.functional.elu(h)
            h = torch.nn.functional.dropout(
                h, p=self.dropout, training=self.training,
            )
            z = self.gat2(h, edge_index)

        return z, attn_weights if return_attention_weights else None

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input features from latent embedding.

        Args:
            z: Latent embedding ``[N, out_channels]``.

        Returns:
            Reconstructed feature matrix ``[N, in_channels]``.
        """
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Full forward pass: encode → decode.

        Args:
            x: Node feature matrix ``[N, in_channels]``.
            edge_index: COO edge index ``[2, E]``.
            return_attention_weights: Whether to return attention
                weights from the encoder.

        Returns:
            Tuple of ``(x_hat, z, attn_weights)`` where *x_hat* is
            the reconstructed input, *z* the latent embedding, and
            *attn_weights* is ``None`` or a list of attention tuples.
        """
        z, attn = self.encode(
            x, edge_index,
            return_attention_weights=return_attention_weights,
        )
        x_hat = self.decode(z)
        return x_hat, z, attn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_gat_autoencoder(
    model: SpatialGATAutoencoder,
    data: Data,
    *,
    epochs: int = 200,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    verbose: bool = True,
) -> list[float]:
    """Train the GAT autoencoder with MSE reconstruction loss.

    Args:
        model: A :class:`SpatialGATAutoencoder` instance.
        data: PyG :class:`Data` with ``x`` and ``edge_index``.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        weight_decay: L2 regularisation coefficient.
        verbose: Print loss every 20 epochs.

    Returns:
        List of per-epoch training losses.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    losses: list[float] = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        x_hat, _z, _attn = model(data.x, data.edge_index)
        loss = criterion(x_hat, data.x)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>4d}/{epochs}  loss = {loss.item():.6f}")

    return losses
