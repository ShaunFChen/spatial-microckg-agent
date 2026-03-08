"""Spatial statistics and cell-cell communication analysis.

Wraps squidpy spatial graph functions to compute:
- Spatial autocorrelation (Moran's I) for gene expression patterns
- Ligand-receptor interaction analysis
- Neighbourhood enrichment across cell types
- Co-occurrence scores between cell types
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import squidpy as sq

__all__ = [
    "compute_spatial_neighbors",
    "compute_spatial_autocorr",
    "run_ligrec_analysis",
    "run_nhood_enrichment",
    "run_co_occurrence",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MORAN_PVAL_THRESHOLD: float = 0.05
_LIGREC_N_PERMS: int = 1000
_LIGREC_PVAL_THRESHOLD: float = 0.01


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_spatial_neighbors(
    adata: ad.AnnData,
    *,
    coord_type: str = "generic",
    n_neighs: int = 6,
) -> ad.AnnData:
    """Build a spatial connectivity graph and store in ``adata.obsp``.

    Args:
        adata: AnnData with ``adata.obsm['spatial']`` coordinates.
        coord_type: Coordinate type — ``"generic"`` or ``"grid"``.
        n_neighs: Number of nearest neighbours.

    Returns:
        The same AnnData object (modified in-place) with spatial
        connectivity and distances stored in ``obsp``.
    """
    sq.gr.spatial_neighbors(
        adata,
        coord_type=coord_type,
        n_neighs=n_neighs,
    )
    print(f"  Spatial neighbours computed (n_neighs={n_neighs})")
    return adata


def compute_spatial_autocorr(
    adata: ad.AnnData,
    genes: list[str] | None = None,
    *,
    mode: str = "moran",
    n_perms: int = 100,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute spatial autocorrelation (Moran's I or Geary's C).

    Args:
        adata: AnnData with spatial graph in ``obsp`` (call
            :func:`compute_spatial_neighbors` first).
        genes: Subset of genes to test.  ``None`` tests all.
        mode: ``"moran"`` or ``"geary"``.
        n_perms: Number of permutations for p-value estimation.
        n_jobs: Number of parallel jobs.

    Returns:
        DataFrame indexed by gene with columns ``I`` (or ``C``),
        ``pval_norm``, ``var_norm``, etc.
    """
    sq.gr.spatial_autocorr(
        adata,
        mode=mode,
        genes=genes,
        n_perms=n_perms,
        n_jobs=n_jobs,
    )
    result_key = "moranI" if mode == "moran" else "gearyC"
    df: pd.DataFrame = adata.uns[result_key].copy()
    n_sig = (df["pval_norm"] < _MORAN_PVAL_THRESHOLD).sum()
    print(f"  Spatial autocorrelation ({mode}): {n_sig}/{len(df)} genes significant (p < {_MORAN_PVAL_THRESHOLD})")
    return df


def run_ligrec_analysis(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    *,
    n_perms: int = _LIGREC_N_PERMS,
) -> dict[str, pd.DataFrame]:
    """Run ligand-receptor interaction analysis across cell types.

    Args:
        adata: AnnData with spatial graph and cluster annotations.
        cluster_key: Column in ``adata.obs`` with cell-type labels.
        n_perms: Number of permutations for significance testing.

    Returns:
        Dictionary with ``"means"`` and ``"pvalues"`` DataFrames
        (multi-index: ligand-receptor pairs × cluster pairs).
    """
    result = sq.gr.ligrec(
        adata,
        cluster_key=cluster_key,
        n_perms=n_perms,
        use_raw=False,
        copy=True,
    )
    means_df: pd.DataFrame = result["means"]
    pvals_df: pd.DataFrame = result["pvalues"]

    # Count significant interactions
    n_sig = int((pvals_df < _LIGREC_PVAL_THRESHOLD).sum().sum())
    n_pairs = int(pvals_df.notna().sum().sum())
    print(f"  Ligand-receptor analysis: {n_sig}/{n_pairs} significant interactions (p < {_LIGREC_PVAL_THRESHOLD})")

    return {"means": means_df, "pvalues": pvals_df}


def run_nhood_enrichment(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    *,
    n_perms: int = 1000,
) -> dict[str, np.ndarray]:
    """Compute neighbourhood enrichment z-scores between clusters.

    Tests whether pairs of cell types are closer or further apart than
    expected under spatial randomness.

    Args:
        adata: AnnData with spatial graph.
        cluster_key: Column in ``adata.obs`` with cell-type labels.
        n_perms: Number of permutations.

    Returns:
        Dictionary with ``"zscore"`` and ``"count"`` arrays (cluster ×
        cluster matrices).  Also stored in ``adata.uns``.
    """
    sq.gr.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        n_perms=n_perms,
    )
    result = adata.uns[f"{cluster_key}_nhood_enrichment"]
    zscore = result["zscore"]
    n_enriched = int((zscore > 1.96).sum())
    n_depleted = int((zscore < -1.96).sum())
    n_total = zscore.shape[0] * zscore.shape[1]
    print(f"  Neighbourhood enrichment: {n_enriched} enriched, {n_depleted} depleted (of {n_total} pairs)")
    return {"zscore": zscore, "count": result["count"]}


def run_co_occurrence(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    *,
    n_steps: int = 50,
) -> dict[str, Any]:
    """Compute spatial co-occurrence scores between cell types.

    Args:
        adata: AnnData with spatial coordinates.
        cluster_key: Column in ``adata.obs`` with cell-type labels.
        n_steps: Number of distance interval steps.

    Returns:
        Dictionary with ``"occ"`` (co-occurrence ratio array) and
        ``"interval"`` (distance intervals).  Also stored in
        ``adata.uns``.
    """
    sq.gr.co_occurrence(
        adata,
        cluster_key=cluster_key,
        n_steps=n_steps,
    )
    result = adata.uns[f"{cluster_key}_co_occurrence"]
    occ = result["occ"]
    print(f"  Co-occurrence computed: {occ.shape[0]} clusters × {occ.shape[2]} distance intervals")
    return {"occ": occ, "interval": result["interval"]}
