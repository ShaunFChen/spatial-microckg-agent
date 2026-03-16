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
    "benchmark_clustering",
    "extract_attention_weights",
    "plot_gat_benchmark",
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


# ---------------------------------------------------------------------------
# Benchmarking & Analysis
# ---------------------------------------------------------------------------


def benchmark_clustering(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    embedding: np.ndarray | None = None,
    name_a: str = "Baseline",
    name_b: str = "GAT",
) -> dict[str, Any]:
    """Compute quantitative clustering comparison metrics.

    Computes Adjusted Rand Index (ARI), Normalized Mutual Information
    (NMI), and optionally Silhouette Score for two clustering
    assignments.  Returns a dictionary suitable for display as a
    formatted table.

    Args:
        labels_a: Cluster labels from method A (e.g. baseline Leiden).
        labels_b: Cluster labels from method B (e.g. GAT Leiden).
        embedding: Optional feature matrix ``[N, D]`` for Silhouette
            computation.  When ``None``, Silhouette is skipped.
        name_a: Display name for method A.
        name_b: Display name for method B.

    Returns:
        Dictionary with keys ``ari``, ``nmi``, ``silhouette_a``,
        ``silhouette_b``, ``n_clusters_a``, ``n_clusters_b``, and
        ``method_names``.
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    ari = adjusted_rand_score(labels_a, labels_b)
    nmi = normalized_mutual_info_score(labels_a, labels_b)

    result: dict[str, Any] = {
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "n_clusters_a": int(len(set(labels_a))),
        "n_clusters_b": int(len(set(labels_b))),
        "method_names": (name_a, name_b),
    }

    if embedding is not None:
        # Subsample for speed on large datasets (Silhouette is O(n²))
        n = len(labels_a)
        max_sample = 10_000
        if n > max_sample:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_sample, replace=False)
            emb_sub = embedding[idx]
            la_sub = np.asarray(labels_a)[idx]
            lb_sub = np.asarray(labels_b)[idx]
        else:
            emb_sub = embedding
            la_sub = np.asarray(labels_a)
            lb_sub = np.asarray(labels_b)

        result["silhouette_a"] = round(
            silhouette_score(emb_sub, la_sub), 4,
        )
        result["silhouette_b"] = round(
            silhouette_score(emb_sub, lb_sub), 4,
        )
    else:
        result["silhouette_a"] = None
        result["silhouette_b"] = None

    print(f"  Clustering benchmark ({name_a} vs {name_b}):")
    print(f"    ARI = {result['ari']:.4f}")
    print(f"    NMI = {result['nmi']:.4f}")
    print(f"    Clusters: {result['n_clusters_a']} ({name_a}), "
          f"{result['n_clusters_b']} ({name_b})")
    if result["silhouette_a"] is not None:
        print(f"    Silhouette: {result['silhouette_a']:.4f} ({name_a}), "
              f"{result['silhouette_b']:.4f} ({name_b})")

    return result


def extract_attention_weights(
    model: SpatialGATAutoencoder,
    data: Data,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-edge attention weights from a trained GAT model.

    Runs a single forward pass with ``return_attention_weights=True``
    through both GAT layers and returns the attention coefficients.

    Args:
        model: A trained :class:`SpatialGATAutoencoder`.
        data: PyG :class:`Data` with ``x`` and ``edge_index``.

    Returns:
        Tuple of ``(edge_index_attn, attention_scores)`` from the
        **first** GAT layer (multi-head; shape ``[E, heads]``).
    """
    device = next(model.parameters()).device
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        _, attn_weights = model.encode(
            data.x, data.edge_index, return_attention_weights=True,
        )

    if attn_weights and len(attn_weights) >= 1:
        edge_idx, attn_coeff = attn_weights[0]
        return edge_idx.cpu(), attn_coeff.cpu()

    raise RuntimeError(
        "No attention weights returned. Ensure model has GATConv layers."
    )


def plot_gat_benchmark(
    adata: Any,
    metrics: dict[str, Any],
    sample_id: str | None = None,
    save_dir: str | None = None,
) -> list:
    """Generate journal-quality GAT vs baseline benchmark visualisation.

    Creates a multi-panel figure with:

    - **Panel A/B:** Side-by-side spatial scatter (baseline vs GAT)
    - **Panel C:** Grouped bar chart of clustering metrics

    Args:
        adata: AnnData with ``leiden``, ``gat_leiden`` columns in
            ``.obs`` and ``spatial`` in ``.obsm``.
        metrics: Dictionary from :func:`benchmark_clustering`.
        sample_id: Sample ID to plot.  If ``None``, uses the sample
            with the most spots.
        save_dir: Directory to save plots.  If ``None``, plots are
            not saved.

    Returns:
        List of saved file paths (empty if *save_dir* is ``None``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    saved: list = []
    name_a, name_b = metrics["method_names"]

    # --- Select sample ---
    if sample_id is None and "sample_id" in adata.obs.columns:
        sample_id = adata.obs["sample_id"].value_counts().idxmax()
    if sample_id is not None and "sample_id" in adata.obs.columns:
        mask = adata.obs["sample_id"] == sample_id
        adata_sub = adata[mask].copy()
    else:
        adata_sub = adata

    # =====================================================================
    # Figure 1: Side-by-side spatial scatter
    # =====================================================================
    coords = adata_sub.obsm["spatial"]
    x_c, y_c = coords[:, 0], coords[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("white")

    for ax, col, title in [
        (axes[0], "leiden", f"{name_a} Leiden (Expression Only)"),
        (axes[1], "gat_leiden", f"{name_b} Leiden (Expression + Topology)"),
    ]:
        cats = adata_sub.obs[col].astype("category")
        n_cats = len(cats.cat.categories)
        cmap = plt.cm.get_cmap("tab20", max(n_cats, 1))
        colors = [cmap(int(c) % 20) for c in cats.values]

        ax.scatter(x_c, y_c, c=colors, s=12, alpha=0.85, linewidths=0)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Spatial X")
        ax.set_ylabel("Spatial Y")
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()

        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=cmap(i % 20), markersize=8, label=f"C{c}",
            )
            for i, c in enumerate(cats.cat.categories)
        ]
        ax.legend(
            handles=handles, fontsize=7, ncol=2, loc="upper right",
            frameon=False, handletextpad=0.3, columnspacing=0.5,
        )

    title_suffix = f" — Sample {sample_id}" if sample_id else ""
    fig.suptitle(
        f"Clustering Benchmark{title_suffix}",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if save_dir:
        p = Path(save_dir) / "benchmark_leiden_vs_gat.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        saved.append(p)
        print(f"  Saved benchmark spatial → {p}")
    plt.close(fig)

    # =====================================================================
    # Figure 2: Metrics comparison bar chart
    # =====================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    metric_names = []
    vals_a = []
    vals_b = []

    metric_names.append("ARI\n(agreement)")
    vals_a.append(metrics["ari"])
    vals_b.append(metrics["ari"])  # ARI is pairwise, same value

    metric_names.append("NMI\n(information)")
    vals_a.append(metrics["nmi"])
    vals_b.append(metrics["nmi"])

    if metrics.get("silhouette_a") is not None:
        metric_names.append(f"Silhouette\n({name_a})")
        vals_a.append(metrics["silhouette_a"])
        vals_b.append(metrics["silhouette_a"])

        metric_names.append(f"Silhouette\n({name_b})")
        vals_a.append(metrics["silhouette_b"])
        vals_b.append(metrics["silhouette_b"])

    colors = ["#4A90D9", "#4A90D9", "#D4A574", "#27AE60"][:len(metric_names)]
    bars = ax.bar(
        range(len(metric_names)), vals_a[:len(metric_names)],
        color=colors, edgecolor="white", linewidth=1.5, width=0.6,
    )

    # Value labels on bars
    for bar, val in zip(bars, vals_a[:len(metric_names)]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Clustering Quality Metrics",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_ylim(0, max(max(vals_a), 1.0) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#CCCCCC", linewidth=0.8)
    plt.tight_layout()

    if save_dir:
        p = Path(save_dir) / "benchmark_metrics.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        saved.append(p)
        print(f"  Saved metrics chart → {p}")
    plt.close(fig)

    return saved
