"""Spatial transcriptomics QC, feature selection, and visualization.

Provides a complete pipeline from raw AnnData to Stabl-selected biomarker
genes with caching and spatial overlay plotting.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

__all__ = [
    "load_adata",
    "run_qc",
    "normalize",
    "select_hvgs",
    "assign_injury_labels",
    "run_stabl_selection",
    "run_stabl_cached",
    "plot_spatial_markers",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MIN_GENES = 200
DEFAULT_MIN_CELLS = 3
DEFAULT_MAX_PCT_MT = 30.0
DEFAULT_TARGET_SUM = 1e4
DEFAULT_N_HVGS = 2000
DEFAULT_N_BOOTSTRAPS = 500

# Marker-to-cell-type mapping for annotation (mouse brain)
CELL_TYPE_MARKERS: dict[str, str] = {
    "Gfap": "Reactive_Astrocyte",
    "Aqp4": "Astrocyte",
    "Tmem119": "Homeostatic_Microglia",
    "Cx3cr1": "Microglia",
    "Csf1r": "Microglia",
    "Mbp": "Oligodendrocyte",
    "Plp1": "Oligodendrocyte",
    "Snap25": "Neuron",
    "Syt1": "Neuron",
    "Cldn5": "Endothelial",
}

# Anatomical region mapping based on Leiden cluster characteristics
REGION_LABELS: list[str] = [
    "Cortex",
    "Hippocampus",
    "Thalamus",
    "Hypothalamus",
    "Striatum",
    "White_Matter",
    "Cerebellum",
    "Brainstem",
]


# ---------------------------------------------------------------------------
# Data Loading & QC
# ---------------------------------------------------------------------------


def load_adata(path: Path | str) -> ad.AnnData:
    """Load an AnnData object from an ``.h5ad`` file.

    Args:
        path: Path to the ``.h5ad`` file.

    Returns:
        The loaded AnnData object.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    print(f"  Loading dataset: {path}")
    adata = sc.read_h5ad(path)
    print(f"  Shape: {adata.shape[0]} spots × {adata.shape[1]} genes")
    return adata


def run_qc(
    adata: ad.AnnData,
    min_genes: int = DEFAULT_MIN_GENES,
    min_cells: int = DEFAULT_MIN_CELLS,
    max_pct_mt: float = DEFAULT_MAX_PCT_MT,
) -> ad.AnnData:
    """Filter cells and genes by standard quality-control metrics.

    Computes mitochondrial gene percentage and filters out low-quality
    spots and rarely detected genes.

    Args:
        adata: Raw AnnData object.
        min_genes: Minimum number of genes expressed per cell.
        min_cells: Minimum number of cells expressing a gene.
        max_pct_mt: Maximum mitochondrial gene percentage per cell.

    Returns:
        A filtered copy of the AnnData object.
    """
    adata = adata.copy()

    # Identify mitochondrial genes (mouse: mt-, human: MT-)
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt, :].copy()
    n_after = adata.n_obs

    print(f"  QC filtering: {n_before} → {n_after} spots")
    print(f"  Genes retained: {adata.n_vars}")
    return adata


def normalize(
    adata: ad.AnnData,
    target_sum: float = DEFAULT_TARGET_SUM,
) -> ad.AnnData:
    """Normalize counts per cell and log-transform.

    Stores raw counts in ``adata.raw`` before normalization so they
    remain accessible for differential-expression analysis and plotting.

    Args:
        adata: QC-filtered AnnData object.
        target_sum: Total counts each cell is normalized to.

    Returns:
        The AnnData with normalized, log-transformed values in ``X``
        and raw counts preserved in ``adata.raw``.
    """
    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    print(f"  Normalized to {target_sum:.0f} counts/cell and log1p-transformed")
    return adata


# ---------------------------------------------------------------------------
# HVG Selection
# ---------------------------------------------------------------------------


def select_hvgs(
    adata: ad.AnnData,
    n_top: int = DEFAULT_N_HVGS,
) -> ad.AnnData:
    """Select the top highly variable genes.

    Args:
        adata: Normalized AnnData object.
        n_top: Number of HVGs to select.

    Returns:
        AnnData subsetted to the *n_top* most variable genes.
    """
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
    n_hvg = adata.var["highly_variable"].sum()
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"  Selected {n_hvg} highly variable genes (requested {n_top})")
    return adata


def _compute_pca_and_clusters(adata: ad.AnnData) -> ad.AnnData:
    """Run PCA, neighbours, UMAP, and Leiden clustering.

    Args:
        adata: Normalized, HVG-subsetted AnnData.

    Returns:
        AnnData with ``leiden`` column in ``.obs``.
    """
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8, flavor="igraph", n_iterations=2, directed=False)
    print(f"  Leiden clustering: {adata.obs['leiden'].nunique()} clusters")
    return adata


# ---------------------------------------------------------------------------
# Label Assignment (Supervised Target for Stabl)
# ---------------------------------------------------------------------------


def assign_injury_labels(
    adata: ad.AnnData,
    method: str = "cluster",
) -> np.ndarray:
    """Assign binary labels for supervised Stabl feature selection.

    For the squidpy Visium mouse brain dataset, Leiden clusters are
    computed and then split into a binary target: clusters enriched in
    cortical/hippocampal regions (label 1) versus subcortical regions
    (label 0). This serves as a *methodological proxy* for injury-vs-
    control contrasts and is documented as such.

    Args:
        adata: Normalized AnnData with spatial coordinates.
        method: Labelling strategy — ``"cluster"`` uses Leiden-based
            binary region split.

    Returns:
        Binary numpy array of shape ``(n_obs,)`` with values 0 or 1.

    Raises:
        ValueError: If *method* is not recognised.
    """
    if method != "cluster":
        raise ValueError(f"Unknown label method '{method}'. Use 'cluster'.")

    if "leiden" not in adata.obs.columns:
        adata = _compute_pca_and_clusters(adata)

    # Binary split: assign the larger half of clusters to label 0,
    # the smaller half to label 1 (proxy for cortex vs. subcortical).
    cluster_counts = adata.obs["leiden"].value_counts()
    median_size = cluster_counts.median()
    cortical_clusters = cluster_counts[cluster_counts >= median_size].index.tolist()

    y = np.where(adata.obs["leiden"].isin(cortical_clusters), 1, 0)
    print(f"  Binary label assignment: {y.sum()} label-1 / {(1 - y).sum()} label-0 spots")
    return y


# ---------------------------------------------------------------------------
# Stabl Feature Selection
# ---------------------------------------------------------------------------


def run_stabl_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
) -> dict[str, Any]:
    """Run Stabl stability-based feature selection.

    Uses Lasso (L1-penalised logistic regression) as the base estimator
    with random-permutation synthetic features for automatic FDP+
    threshold determination.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Binary target array of shape ``(n_samples,)``.
        feature_names: Gene names corresponding to columns of *X*.
        n_bootstraps: Number of bootstrap iterations (default 500).

    Returns:
        Dictionary with keys:

        - ``selected_genes`` — list of selected gene names.
        - ``stability_scores`` — dict mapping gene name → max stability score.
        - ``all_scores`` — numpy array of max stability scores for all input features.
        - ``fdr`` — minimum false-discovery proportion achieved.
        - ``threshold`` — stability threshold used for selection.
        - ``n_selected`` — count of features selected.
    """
    from sklearn.linear_model import LogisticRegression
    from stabl.stabl import Stabl

    print(f"  Running Stabl ({n_bootstraps} bootstraps, {X.shape[1]} features)...")

    base_estimator = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
    )

    stabl = Stabl(
        base_estimator=base_estimator,
        lambda_grid={"C": np.linspace(0.01, 1, 30)},
        n_bootstraps=n_bootstraps,
        artificial_type="random_permutation",
        artificial_proportion=1.0,
        sample_fraction=0.5,
        random_state=42,
    )

    stabl.fit(X, y)

    # Extract results
    importances = stabl.get_importances()
    selected_mask = stabl.get_support()
    selected_names = [
        feature_names[i] for i, s in enumerate(selected_mask) if s
    ]
    stability_scores = {
        feature_names[i]: float(importances[i])
        for i, s in enumerate(selected_mask)
        if s
    }

    fdr = float(getattr(stabl, "min_fdr_", 0.0))
    threshold = float(getattr(stabl, "fdr_min_threshold_", 0.0))

    print(f"  Stabl converged: {len(selected_names)} features selected")
    print(f"  FDP+ threshold: {threshold:.4f}, minimum FDP+: {fdr:.4f}")

    return {
        "selected_genes": selected_names,
        "stability_scores": stability_scores,
        "all_scores": importances,
        "all_feature_names": feature_names,
        "fdr": fdr,
        "threshold": threshold,
        "n_selected": len(selected_names),
    }


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(dataset_name: str, n_hvgs: int, label_method: str) -> str:
    """Compute a deterministic cache key string.

    Args:
        dataset_name: Identifier for the dataset (e.g. ``"squidpy"``).
        n_hvgs: Number of HVGs used.
        label_method: Label assignment method.

    Returns:
        A hex digest string.
    """
    raw = f"{dataset_name}_{n_hvgs}_{label_method}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def run_stabl_cached(
    adata: ad.AnnData,
    cache_dir: Path | str,
    dataset_name: str = "squidpy",
    n_hvgs: int = DEFAULT_N_HVGS,
    label_method: str = "cluster",
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
) -> dict[str, Any]:
    """Run Stabl with disk caching of results.

    If a cache file exists for the given parameter combination, loads it
    directly (completing in under one second). Otherwise runs the full
    Stabl computation and persists the results.

    Args:
        adata: Normalized AnnData (not yet HVG-subsetted).
        cache_dir: Directory for cache files.
        dataset_name: Identifier for cache key computation.
        n_hvgs: Number of HVGs to select before Stabl.
        label_method: Label assignment strategy.
        n_bootstraps: Number of Stabl bootstrap iterations.

    Returns:
        Stabl result dictionary (see :func:`run_stabl_selection`).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(dataset_name, n_hvgs, label_method)
    pkl_path = cache_dir / f"stabl_results_{key}.pkl"
    csv_path = cache_dir / f"stabl_features_{key}.csv"

    if pkl_path.exists():
        print(f"  Loading cached Stabl results: {pkl_path}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)  # noqa: S301

    # --- Fresh computation ---
    print("  No cache found — computing from scratch.")

    # 1) HVG selection
    adata_hvg = select_hvgs(adata.copy(), n_top=n_hvgs)

    # 2) Clustering & label assignment
    adata_hvg = _compute_pca_and_clusters(adata_hvg)
    y = assign_injury_labels(adata_hvg, method=label_method)

    # 3) Prepare dense matrix for Stabl
    X = adata_hvg.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    feature_names = list(adata_hvg.var_names)

    # 4) Run Stabl
    result = run_stabl_selection(X, y, feature_names, n_bootstraps=n_bootstraps)

    # 5) Persist cache
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)

    df = pd.DataFrame({
        "gene": result["selected_genes"],
        "stability_score": [result["stability_scores"][g] for g in result["selected_genes"]],
    }).sort_values("stability_score", ascending=False)
    df.to_csv(csv_path, index=False)

    print(f"  Cached to {pkl_path} and {csv_path}")
    return result


# ---------------------------------------------------------------------------
# Spatial Visualization
# ---------------------------------------------------------------------------


def plot_spatial_markers(
    adata: ad.AnnData,
    markers: list[str],
    save_dir: Path | str,
    n_top: int = 5,
) -> list[Path]:
    """Overlay marker gene expression on the H&E tissue image.

    Generates one spatial plot per marker gene and saves each as a
    high-resolution PNG file.

    Args:
        adata: AnnData with spatial coordinates in
            ``obsm['spatial']`` and an H&E image in ``uns['spatial']``.
        markers: Candidate marker genes sorted by stability score.
        save_dir: Directory to write PNG files.
        n_top: Maximum number of markers to plot.

    Returns:
        List of paths to saved plot images.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use raw counts for expression overlay if available
    plot_adata = adata.raw.to_adata() if adata.raw is not None else adata

    # Filter to markers present in the dataset
    available = [m for m in markers[:n_top] if m in plot_adata.var_names]
    if not available:
        print("  No requested markers found in dataset — skipping spatial plots.")
        return []

    saved: list[Path] = []
    for gene in available:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sc.pl.spatial(
            plot_adata,
            color=gene,
            ax=ax,
            show=False,
            title=f"{gene} — Spatial Expression",
            frameon=False,
        )
        out_path = save_dir / f"spatial_{gene}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"  Saved spatial plot: {out_path}")

    return saved
