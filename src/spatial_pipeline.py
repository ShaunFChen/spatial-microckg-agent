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
    "select_de_genes",
    "compute_clusters",
    "annotate_clusters",
    "assign_condition_labels",
    "stratified_downsample",
    "batch_correct",
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
DEFAULT_N_BOOTSTRAPS = 200
DEFAULT_SPOTS_PER_SAMPLE = 1000
DEFAULT_FDR_ALPHA = 0.01
DEFAULT_MIN_LOG2FC = 0.5

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

# Brain-region marker panels for spatial cluster annotation
REGION_MARKER_PANELS: dict[str, list[str]] = {
    "Cortex": ["Slc17a7", "Satb2", "Cux2", "Tbr1", "Foxp2"],
    "Hippocampus": ["Prox1", "Fibcd1", "Dkk3", "C1ql2"],
    "Thalamus": ["Gbx2", "Tcf7l2", "Prkcd"],
    "Hypothalamus": ["Oxt", "Avp", "Agrp", "Pomc"],
    "Striatum": ["Ppp1r1b", "Drd1", "Drd2", "Rarb"],
    "White_Matter": ["Mbp", "Plp1", "Mag", "Mog"],
    "Cerebellum": ["Pcp2", "Calb1", "Calb2"],
    "Brainstem": ["Slc6a5", "Lhx3", "Chat"],
}

# Neuroinflammation signature genes for injury labelling
NEUROINFLAMMATION_MARKERS: list[str] = [
    "Gfap", "Aif1", "Cd68", "Tnf", "Il1b", "Cxcl10",
    "C1qa", "C1qb", "C3", "Trem2", "Tyrobp", "Csf1r",
]

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


def select_de_genes(
    adata: ad.AnnData,
    groupby: str = "condition",
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    min_log2fc: float = DEFAULT_MIN_LOG2FC,
) -> ad.AnnData:
    """Pre-filter genes by t-test differential expression with FDR correction.

    Runs a Welch t-test between groups defined by *groupby*, applies
    Benjamini-Hochberg FDR correction, and retains genes that pass both
    the adjusted p-value and fold-change thresholds.  This produces a
    biologically focused gene set for downstream Stabl selection.

    Args:
        adata: Normalized AnnData object.
        groupby: Column in ``adata.obs`` defining the two groups
            (e.g. ``"condition"`` with values 0/1).
        fdr_alpha: Maximum BH-adjusted p-value to retain a gene.
        min_log2fc: Minimum absolute log2 fold-change.

    Returns:
        AnnData subsetted to DE-significant genes.

    Raises:
        KeyError: If *groupby* is not in ``adata.obs``.
        ValueError: If fewer than 2 groups exist in *groupby*.
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"Column '{groupby}' not found in adata.obs.")

    groups = adata.obs[groupby].unique()
    if len(groups) < 2:
        raise ValueError(
            f"Need ≥2 groups in '{groupby}', found {len(groups)}."
        )

    # Ensure groupby column is categorical (required by scanpy)
    if not hasattr(adata.obs[groupby].dtype, "categories"):
        adata.obs[groupby] = adata.obs[groupby].astype("category")

    # Run t-test (Welch, unequal variance) with BH correction
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method="t-test",
        use_raw=False,
        key_added="_de_prefilter",
    )

    # Collect results for all groups and take the union of significant genes
    sig_genes: set[str] = set()
    for grp in groups:
        df = sc.get.rank_genes_groups_df(
            adata, group=str(grp), key="_de_prefilter"
        )
        mask = (
            (df["pvals_adj"] < fdr_alpha)
            & (df["logfoldchanges"].abs() > min_log2fc)
        )
        sig_genes.update(df.loc[mask, "names"].tolist())

    # Clean up temporary key
    del adata.uns["_de_prefilter"]

    sig_genes_sorted = [g for g in adata.var_names if g in sig_genes]
    adata_de = adata[:, sig_genes_sorted].copy()

    print(
        f"  DE pre-filter ({groupby}, t-test): "
        f"{adata.n_vars} → {adata_de.n_vars} genes "
        f"(FDR < {fdr_alpha}, |log2FC| > {min_log2fc})"
    )
    return adata_de


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


def compute_clusters(
    adata: ad.AnnData,
    n_hvgs: int = DEFAULT_N_HVGS,
) -> ad.AnnData:
    """Add Leiden clusters to AnnData via HVG → PCA → neighbours → Leiden.

    This is a convenience wrapper that selects HVGs, runs PCA +
    Leiden on a copy, and transfers the ``leiden`` column back to
    the original AnnData **in-place**.

    Args:
        adata: Normalized AnnData (full gene set).
        n_hvgs: Number of HVGs for the clustering step.

    Returns:
        The same *adata* with a ``leiden`` column added to ``.obs``.
    """
    adata_hvg = select_hvgs(adata.copy(), n_top=n_hvgs)
    adata_hvg = _compute_pca_and_clusters(adata_hvg)
    adata.obs["leiden"] = adata_hvg.obs["leiden"]
    return adata


def annotate_clusters(
    adata: ad.AnnData,
) -> dict[str, str]:
    """Assign brain-region labels to Leiden clusters via marker-gene scoring.

    Scores each cluster against known brain-region marker panels using
    ``sc.tl.score_genes``.  The region with the highest mean score in
    a cluster wins.  If no panel scores significantly, the cluster is
    labelled ``Unassigned``.

    Args:
        adata: AnnData with a ``leiden`` column and expression data.

    Returns:
        Dict mapping cluster id (str) to region label (str).
    """
    if "leiden" not in adata.obs.columns:
        raise ValueError("Run compute_clusters() before annotate_clusters().")

    src = adata.raw.to_adata() if adata.raw is not None else adata

    # Score every panel
    region_scores: dict[str, str] = {}
    score_cols: list[str] = []
    for region, markers in REGION_MARKER_PANELS.items():
        available = [g for g in markers if g in src.var_names]
        col = f"_score_{region}"
        if len(available) >= 2:
            sc.tl.score_genes(adata, gene_list=available, score_name=col)
        else:
            adata.obs[col] = 0.0
        score_cols.append(col)

    # For each cluster pick the region with the highest mean score
    cluster_ids = sorted(adata.obs["leiden"].unique(), key=int)
    for cid in cluster_ids:
        mask = adata.obs["leiden"] == cid
        best_region = "Unassigned"
        best_val = 0.0
        for region, col in zip(REGION_MARKER_PANELS, score_cols):
            mean_score = float(adata.obs.loc[mask, col].mean())
            if mean_score > best_val:
                best_val = mean_score
                best_region = region
        region_scores[str(cid)] = best_region

    # Clean up temporary columns
    adata.obs.drop(columns=score_cols, inplace=True, errors="ignore")

    summary = pd.Series(region_scores).value_counts()
    print(f"  Cluster annotation: {dict(summary)}")
    return region_scores


# ---------------------------------------------------------------------------
# Downsampling & Batch Correction
# ---------------------------------------------------------------------------


def stratified_downsample(
    adata: ad.AnnData,
    n_per_sample: int = DEFAULT_SPOTS_PER_SAMPLE,
    sample_key: str = "sample_id",
    resolution: float = 0.5,
    random_state: int = 42,
) -> ad.AnnData:
    """Unsupervised Stratified Downsampling to objectively preserve spatial anatomical heterogeneity.

    Runs a rapid PCA + Leiden clustering on the merged AnnData, then
    samples proportionally from each Leiden cluster within each
    biological sample.  This prevents under-representation of
    anatomically distinct regions (e.g. Hippocampus) that pure random
    downsampling would risk eliminating.

    Args:
        adata: Normalized AnnData with a sample identifier column.
        n_per_sample: Target number of spots per sample.
        sample_key: Column in ``.obs`` identifying biological samples.
        resolution: Leiden clustering resolution for stratification.
        random_state: Seed for reproducibility.

    Returns:
        A stratified-downsampled copy of the AnnData.

    Raises:
        KeyError: If *sample_key* is not in ``adata.obs``.
    """
    if sample_key not in adata.obs.columns:
        raise KeyError(f"Column '{sample_key}' not found in adata.obs.")

    rng = np.random.default_rng(random_state)

    # --- Rapid clustering for stratification ---
    print("  Running PCA + Leiden for stratification...")
    import warnings
    adata_tmp = adata.copy()
    sc.pp.highly_variable_genes(adata_tmp, n_top_genes=DEFAULT_N_HVGS)
    adata_tmp = adata_tmp[:, adata_tmp.var["highly_variable"]].copy()
    sc.pp.scale(adata_tmp, max_value=10)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="zero-centering a sparse")
        sc.tl.pca(adata_tmp, svd_solver="arpack")
    sc.pp.neighbors(adata_tmp, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(
        adata_tmp, resolution=resolution,
        flavor="igraph", n_iterations=2, directed=False,
    )
    adata.obs["_strat_leiden"] = adata_tmp.obs["leiden"].values
    n_clusters = adata.obs["_strat_leiden"].nunique()
    print(f"  Stratification clusters: {n_clusters}")

    # --- Proportional sampling per (sample × cluster) ---
    indices: list[int] = []
    for sid in adata.obs[sample_key].unique():
        sample_mask = adata.obs[sample_key] == sid
        sample_idx = np.where(sample_mask)[0]
        n_sample = len(sample_idx)
        n_target = min(n_per_sample, n_sample)

        clusters = adata.obs.loc[sample_mask, "_strat_leiden"]
        cluster_counts = clusters.value_counts()

        chosen: list[int] = []
        for cid, count in cluster_counts.items():
            proportion = count / n_sample
            n_from_cluster = max(1, int(round(proportion * n_target)))
            n_from_cluster = min(n_from_cluster, count)
            cluster_idx = np.where(
                sample_mask & (adata.obs["_strat_leiden"] == cid)
            )[0]
            picked = rng.choice(cluster_idx, size=n_from_cluster, replace=False)
            chosen.extend(picked.tolist())

        # Trim or pad to exact target
        if len(chosen) > n_target:
            chosen = rng.choice(chosen, size=n_target, replace=False).tolist()
        elif len(chosen) < n_target:
            remaining = list(set(sample_idx.tolist()) - set(chosen))
            if remaining:
                extra = rng.choice(
                    remaining,
                    size=min(n_target - len(chosen), len(remaining)),
                    replace=False,
                )
                chosen.extend(extra.tolist())
        indices.extend(chosen)

    indices = sorted(set(indices))
    result = adata[indices].copy()

    # Clean up temporary column
    if "_strat_leiden" in result.obs.columns:
        result.obs.drop(columns=["_strat_leiden"], inplace=True)
    if "_strat_leiden" in adata.obs.columns:
        adata.obs.drop(columns=["_strat_leiden"], inplace=True)

    print(f"  Stratified downsample: {adata.n_obs} → {result.n_obs} spots "
          f"(≤{n_per_sample} per sample, {n_clusters} strata)")
    return result


def batch_correct(
    adata: ad.AnnData,
    batch_key: str = "batch",
) -> ad.AnnData:
    """Apply ComBat batch correction to log-normalized expression.

    Uses ``scanpy.pp.combat`` to regress out technical batch variance
    between tissue slices. Must be called on log-normalized data
    **before** HVG selection.

    Args:
        adata: Log-normalized AnnData with a batch column in ``.obs``.
        batch_key: Column in ``.obs`` identifying batches.

    Returns:
        Batch-corrected AnnData (modified in-place and returned).

    Raises:
        KeyError: If *batch_key* is not in ``adata.obs``.
    """
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column '{batch_key}' not found in adata.obs.")

    n_batches = adata.obs[batch_key].nunique()
    print(f"  Applying ComBat batch correction ({n_batches} batches)...")

    # Remove zero-variance genes that cause ComBat NaN warnings
    import warnings
    if hasattr(adata.X, "toarray"):
        gene_var = np.asarray(adata.X.toarray().var(axis=0))
    else:
        gene_var = np.asarray(adata.X.var(axis=0)).ravel()
    nonzero_mask = gene_var > 0
    n_zero = int((~nonzero_mask).sum())
    if n_zero > 0:
        print(f"  Removed {n_zero} zero-variance genes before ComBat.")
        adata = adata[:, nonzero_mask].copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sc.pp.combat(adata, key=batch_key)
    print("  ComBat correction applied.")
    return adata


# ---------------------------------------------------------------------------
# Label Assignment (Supervised Target for Stabl)
# ---------------------------------------------------------------------------


def assign_condition_labels(
    adata: ad.AnnData,
    method: str = "cluster",
) -> np.ndarray:
    """Assign binary labels for supervised Stabl feature selection.

    Supports two strategies:

    - ``"condition"`` — reads the ground-truth experimental label
      from ``adata.obs['condition']`` (0 = WT, 1 = AD).
    - ``"cluster"`` — scores spots with a neuroinflammation gene
      signature and splits at the median score.

    Args:
        adata: Normalized AnnData.
        method: Labelling strategy — ``"condition"`` for ground-truth
            experimental labels, ``"cluster"`` for proxy scoring.

    Returns:
        Binary numpy array of shape ``(n_obs,)`` with values 0 or 1.

    Raises:
        ValueError: If *method* is not recognised.
        KeyError: If ``method="condition"`` but the column is absent.
    """
    if method == "condition":
        if "condition" not in adata.obs.columns:
            raise KeyError(
                "adata.obs['condition'] not found. Ensure the GEO AD "
                "dataset was loaded with condition labels."
            )
        y = adata.obs["condition"].astype(int).values
        counts = pd.Series(y).value_counts().sort_index()
        print(f"  Ground-truth condition labels: "
              f"WT(0)={counts.get(0, 0)}, AD(1)={counts.get(1, 0)}")
        return y

    if method != "cluster":
        raise ValueError(
            f"Unknown label method '{method}'. Use 'condition' or 'cluster'."
        )

    src = adata.raw.to_adata() if adata.raw is not None else adata
    available = [g for g in NEUROINFLAMMATION_MARKERS if g in src.var_names]

    if len(available) >= 3:
        sc.tl.score_genes(adata, gene_list=available, score_name="neuroinflam_score")
        median_score = float(adata.obs["neuroinflam_score"].median())
        y = np.where(adata.obs["neuroinflam_score"] >= median_score, 1, 0)
        print(f"  Neuroinflammation scoring ({len(available)} markers): "
              f"{y.sum()} reactive / {(1 - y).sum()} homeostatic spots")
    else:
        # Fallback: Leiden-based split when signature genes are absent
        if "leiden" not in adata.obs.columns:
            adata = _compute_pca_and_clusters(adata)
        cluster_counts = adata.obs["leiden"].value_counts()
        median_size = cluster_counts.median()
        large_clusters = cluster_counts[cluster_counts >= median_size].index.tolist()
        y = np.where(adata.obs["leiden"].isin(large_clusters), 1, 0)
        print(f"  Fallback label assignment (cluster-size split): "
              f"{y.sum()} label-1 / {(1 - y).sum()} label-0 spots")
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
        n_jobs=-1,
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

def _cache_key(
    dataset_name: str,
    label_method: str,
    prefilter: str = "hvg",
    n_hvgs: int = DEFAULT_N_HVGS,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    min_log2fc: float = DEFAULT_MIN_LOG2FC,
) -> str:
    """Compute a deterministic cache key string.

    Args:
        dataset_name: Identifier for the dataset (e.g. ``"squidpy"``).
        label_method: Label assignment method.
        prefilter: Pre-filter strategy — ``"de"`` or ``"hvg"``.
        n_hvgs: Number of HVGs (used when ``prefilter="hvg"``).
        fdr_alpha: FDR threshold (used when ``prefilter="de"``).
        min_log2fc: Min |log2FC| (used when ``prefilter="de"``).

    Returns:
        A hex digest string.
    """
    if prefilter == "de":
        raw = f"{dataset_name}_de{fdr_alpha}_fc{min_log2fc}_{label_method}"
    else:
        raw = f"{dataset_name}_{n_hvgs}_{label_method}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def run_stabl_cached(
    adata: ad.AnnData,
    cache_dir: Path | str,
    dataset_name: str = "squidpy",
    n_hvgs: int = DEFAULT_N_HVGS,
    label_method: str = "cluster",
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    prefilter: str = "hvg",
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    min_log2fc: float = DEFAULT_MIN_LOG2FC,
) -> dict[str, Any]:
    """Run Stabl with disk caching of results.

    If a cache file exists for the given parameter combination, loads it
    directly (completing in under one second). Otherwise runs the full
    Stabl computation and persists the results.

    Two pre-filter strategies are supported:

    - ``"de"`` — t-test differential expression with BH FDR correction
      (recommended when ground-truth condition labels are available).
    - ``"hvg"`` — top-N highly variable genes (original behaviour).

    Args:
        adata: Normalized AnnData (not yet subsetted).
        cache_dir: Directory for cache files.
        dataset_name: Identifier for cache key computation.
        n_hvgs: Number of HVGs (used when ``prefilter="hvg"``).
        label_method: Label assignment strategy.
        n_bootstraps: Number of Stabl bootstrap iterations.
        prefilter: Gene pre-filter — ``"de"`` for t-test DE or
            ``"hvg"`` for highly variable genes.
        fdr_alpha: FDR threshold for DE pre-filter (default 0.01).
        min_log2fc: Minimum |log2FC| for DE pre-filter (default 0.5).

    Returns:
        Stabl result dictionary (see :func:`run_stabl_selection`).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(
        dataset_name, label_method,
        prefilter=prefilter, n_hvgs=n_hvgs,
        fdr_alpha=fdr_alpha, min_log2fc=min_log2fc,
    )
    pkl_path = cache_dir / f"stabl_results_{key}.pkl"
    csv_path = cache_dir / f"stabl_features_{key}.csv"

    if pkl_path.exists():
        print(f"  Loading cached Stabl results: {pkl_path}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)  # noqa: S301

    # --- Fresh computation ---
    print("  No cache found — computing from scratch.")

    # 1) Unsupervised Stratified Downsampling
    adata_work = adata.copy()
    if label_method == "condition":
        if "sample_id" in adata_work.obs.columns:
            adata_work = stratified_downsample(adata_work)

    # 2) Gene pre-filtering
    if prefilter == "de" and label_method == "condition":
        adata_sub = select_de_genes(
            adata_work, groupby="condition",
            fdr_alpha=fdr_alpha, min_log2fc=min_log2fc,
        )
    else:
        adata_sub = select_hvgs(adata_work, n_top=n_hvgs)

    # No ComBat or scaling — keep sparse log-normalized matrix intact
    # so Stabl's bootstrapping sees the full biological signal.

    # 3) Label assignment
    if label_method == "condition":
        y = assign_condition_labels(adata_sub, method="condition")
    else:
        adata_sub = _compute_pca_and_clusters(adata_sub)
        y = assign_condition_labels(adata_sub, method=label_method)

    # 4) Prepare dense matrix for Stabl
    X = adata_sub.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    feature_names = list(adata_sub.var_names)

    # 5) Run Stabl
    result = run_stabl_selection(X, y, feature_names, n_bootstraps=n_bootstraps)

    # 6) Persist cache
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


def _has_spatial(adata: ad.AnnData) -> bool:
    """Check whether AnnData has spatial coordinates and library metadata.

    Args:
        adata: AnnData to inspect.

    Returns:
        ``True`` if spatial data exists for at least one library.
    """
    spatial_uns = adata.uns.get("spatial")
    if spatial_uns is None or adata.obsm.get("spatial") is None:
        return False
    return len(spatial_uns) >= 1


def _ensure_umap(adata: ad.AnnData) -> ad.AnnData:
    """Compute PCA, neighbours, and UMAP if not already present.

    Args:
        adata: AnnData (normalized, optionally HVG-subsetted).

    Returns:
        The same AnnData with ``X_umap`` in ``.obsm``.
    """
    if "X_umap" not in adata.obsm:
        print("  Computing UMAP embedding for fallback visualization...")
        if "X_pca" not in adata.obsm:
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, svd_solver="arpack")
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
    return adata


# Anatomical verification markers to always include in plots
_ANATOMICAL_MARKERS: list[str] = ["Prox1"]
# AD inflammatory markers to prioritize in plots
_AD_MARKERS: list[str] = ["Trem2", "Gfap"]


def plot_spatial_markers(
    adata: ad.AnnData,
    markers: list[str],
    save_dir: Path | str,
    n_top: int = 5,
) -> list[Path]:
    """Visualize marker gene expression with spatial H&E overlays or UMAP.

    Always includes **Prox1** (Hippocampus Dentate Gyrus marker) and
    AD inflammatory markers (Trem2, Gfap) alongside the top
    Stabl-selected genes.  If the AnnData contains spatial coordinates
    and H&E images, generates semi-transparent spatial overlay plots
    using ``squidpy.pl.spatial_scatter``.  Otherwise falls back to UMAP.

    Args:
        adata: AnnData with expression data.
        markers: Candidate marker genes sorted by stability score.
        save_dir: Directory to write PNG files.
        n_top: Maximum number of markers to plot.

    Returns:
        List of paths to saved plot images.
    """
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import squidpy as sq

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use raw counts for expression overlay if available
    plot_adata = adata.raw.to_adata() if adata.raw is not None else adata.copy()

    # CRITICAL: Restore spatial metadata stripped by .raw.to_adata()
    if "spatial" in adata.obsm:
        plot_adata.obsm["spatial"] = adata.obsm["spatial"]
    if "spatial" in adata.uns:
        plot_adata.uns["spatial"] = adata.uns["spatial"]

    use_spatial = _has_spatial(plot_adata)

    # Build gene list: top Stabl markers + Prox1 + AD markers (deduplicated)
    gene_list: list[str] = []
    seen: set[str] = set()
    for g in list(markers[:n_top]) + _ANATOMICAL_MARKERS + _AD_MARKERS:
        if g not in seen and g in plot_adata.var_names:
            gene_list.append(g)
            seen.add(g)

    if not gene_list:
        print("  No requested markers found in dataset — skipping plots.")
        return []

    if not use_spatial:
        print("  Spatial coordinates not available — using UMAP fallback.")
        plot_adata = _ensure_umap(plot_adata)

    # Pick one WT and one AD sample for side-by-side spatial plots
    wt_lib: str | None = None
    ad_lib: str | None = None
    if use_spatial and "sample_id" in plot_adata.obs.columns and "condition" in plot_adata.obs.columns:
        for sid in plot_adata.obs["sample_id"].unique():
            cond = plot_adata.obs.loc[plot_adata.obs["sample_id"] == sid, "condition"].iloc[0]
            if int(cond) == 0 and wt_lib is None:
                wt_lib = str(sid)
            elif int(cond) == 1 and ad_lib is None:
                ad_lib = str(sid)

    saved: list[Path] = []
    for gene in gene_list:
        if use_spatial and wt_lib and ad_lib:
            # Side-by-side WT vs AD on H&E tissue overlays
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            for ax, lib_id, label in zip(
                axes, [wt_lib, ad_lib], ["WT", "AD"]
            ):
                sub = plot_adata[plot_adata.obs["sample_id"] == lib_id].copy()
                sub.uns["spatial"] = {lib_id: plot_adata.uns["spatial"][lib_id]}
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    sq.pl.spatial_scatter(
                        sub,
                        color=gene,
                        library_id=[lib_id],
                        img_res_key="lowres",
                        ax=ax,
                        title=f"{label} ({lib_id}) — {gene}",
                        frameon=False,
                        cmap="magma",
                        size=0.8,
                        alpha=0.6,
                        img=True,
                    )
            fig.tight_layout()
            out_path = save_dir / f"spatial_{gene}.png"
        elif use_spatial:
            # Single-library fallback
            lib_id = list(plot_adata.uns["spatial"].keys())[0]
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                sq.pl.spatial_scatter(
                    plot_adata,
                    color=gene,
                    library_id=[lib_id],
                    img_res_key="lowres",
                    ax=ax,
                    title=f"{gene} — Spatial Expression",
                    frameon=False,
                    cmap="magma",
                    size=0.8,
                    alpha=0.6,
                    img=True,
                )
            out_path = save_dir / f"spatial_{gene}.png"
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            sc.pl.umap(
                plot_adata,
                color=gene,
                ax=ax,
                show=False,
                title=f"{gene} — UMAP Expression",
                frameon=False,
            )
            out_path = save_dir / f"umap_{gene}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"  Saved plot: {out_path}")

    # Bonus: UMAP colored by condition label (WT vs AD)
    if not use_spatial and "condition" in plot_adata.obs.columns:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sc.pl.umap(
            plot_adata,
            color="condition",
            ax=ax,
            show=False,
            title="Condition (WT vs AD) — UMAP",
            frameon=False,
        )
        out_path = save_dir / "umap_condition.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"  Saved plot: {out_path}")

    return saved
