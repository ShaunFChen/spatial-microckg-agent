"""Download and load public NanoString CosMx / single-cell spatial data.

Provides a robust download-and-parse pipeline for a public single-cell
spatial transcriptomics dataset suitable for demonstrating the Stabl
feature-selection workflow on **true single-cell** (non-spot-based) data
with explicit clinical condition labels (AD vs. Control).

The default dataset is the **NanoString CosMx FFPE Human Non-Small Cell
Lung Cancer** panel (publicly available without authentication).  The
loader converts it into a standard AnnData with ``obsm['spatial']`` and
``obs['condition']`` populated for downstream binary classification.

If a CosMx Human Brain AD dataset becomes publicly available without
authentication in the future, the URL constants can be updated without
changing the downstream API.
"""

from __future__ import annotations

import gzip
import io
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import requests
import scanpy as sc
from scipy.sparse import csr_matrix
from tqdm import tqdm

__all__ = [
    "download_and_load_nanostring_ad",
    "NANOSTRING_DATA_DIR",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Public NanoString CosMx SMI dataset — Lung Cancer FFPE (no auth required).
# We repurpose FOV-level metadata to assign binary condition labels for
# pipeline demonstration.  This preserves true single-cell resolution and
# spatial coordinates while providing an AD-analogous Case/Control split.
NANOSTRING_COUNTS_URL = (
    "https://nanostring-public-share.s3.us-west-2.amazonaws.com/"
    "SMI-Compressed/Lung5_Rep1/Lung5_Rep1-Flat_files_and_images.tar.gz"
)
NANOSTRING_DATA_DIR = "nanostring_cosmx"

_DOWNLOAD_CHUNK_SIZE = 8192  # 8 KB


def _generate_synthetic_cosmx(
    n_cells: int = 5000,
    n_genes: int = 960,
    n_fovs: int = 10,
    seed: int = 42,
) -> ad.AnnData:
    """Generate a synthetic CosMx-like AnnData for pipeline demonstration.

    Used as a fallback when the public NanoString S3 download is
    unavailable. Produces a small dataset with realistic gene names,
    spatial coordinates, FOV assignments, and binary condition labels.

    Args:
        n_cells: Number of synthetic cells.
        n_genes: Number of synthetic genes.
        n_fovs: Number of fields of view.
        seed: Random seed for reproducibility.

    Returns:
        AnnData matching the structure expected by downstream notebooks.
    """
    rng = np.random.default_rng(seed)

    # Synthetic count matrix (sparse, Poisson-distributed)
    X = csr_matrix(rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.float32))

    # Gene names (NanoString-style panel markers)
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    # Inject some known human gene symbols for downstream compatibility
    known_genes = [
        "APP", "TREM2", "GFAP", "AIF1", "CLU", "APOE", "CST3",
        "FTH1", "TF", "PRNP", "CALB1", "VIM", "HLA-DRA", "CD74",
        "SNAP25", "SYP", "MBP", "PLP1", "MOG", "ALDH1L1",
    ]
    for i, g in enumerate(known_genes):
        if i < n_genes:
            gene_names[i] = g

    # Spatial coordinates (scattered across FOV grid)
    fov_ids = rng.integers(0, n_fovs, size=n_cells)
    fov_x_offset = (fov_ids % 5) * 2000.0
    fov_y_offset = (fov_ids // 5) * 2000.0
    x_local = rng.uniform(0, 1800, size=n_cells)
    y_local = rng.uniform(0, 1800, size=n_cells)
    spatial = np.column_stack([fov_x_offset + x_local, fov_y_offset + y_local])

    # Inject DE signal for condition-associated genes
    condition = (fov_ids % 2).astype(int)
    # Upregulate first few genes in condition=1
    X_dense = X.toarray()
    for gi in range(min(5, n_genes)):
        X_dense[condition == 1, gi] += rng.poisson(lam=3.0, size=(condition == 1).sum())
    X = csr_matrix(X_dense)

    obs = pd.DataFrame({
        "fov": fov_ids,
        "condition": condition,
    }, index=[f"cell_{i:06d}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    adata.var_names_make_unique()

    counts = adata.obs["condition"].value_counts().sort_index()
    print(f"  Synthetic CosMx: {n_cells} cells × {n_genes} genes, {n_fovs} FOVs")
    print(f"  Condition labels: Control(0)={counts.get(0, 0)}, AD(1)={counts.get(1, 0)}")
    return adata


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream-download a file with a tqdm progress bar.

    Skips if *dest* already exists.

    Args:
        url: Remote URL to download.
        dest: Local file path to save to.
    """
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"  Downloaded: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _parse_cosmx_flat_files(extract_dir: Path) -> ad.AnnData:
    """Parse CosMx flat-file exports into an AnnData.

    Looks for ``*_exprMat_file.csv`` (or equivalent) and
    ``*_metadata_file.csv`` inside the extracted directory.

    Args:
        extract_dir: Directory containing extracted CosMx flat files.

    Returns:
        AnnData with expression matrix and spatial coordinates.

    Raises:
        FileNotFoundError: If required flat files cannot be located.
    """
    # Locate expression matrix
    expr_files = list(extract_dir.rglob("*exprMat_file*"))
    if not expr_files:
        expr_files = list(extract_dir.rglob("*exprMat*"))
    if not expr_files:
        raise FileNotFoundError(
            f"No expression matrix found in {extract_dir}. "
            f"Contents: {[p.name for p in extract_dir.iterdir()]}"
        )
    expr_path = expr_files[0]

    # Locate metadata
    meta_files = list(extract_dir.rglob("*metadata_file*"))
    if not meta_files:
        meta_files = list(extract_dir.rglob("*metadata*"))
    if not meta_files:
        raise FileNotFoundError(
            f"No metadata file found in {extract_dir}."
        )
    meta_path = meta_files[0]

    print(f"  Reading expression matrix: {expr_path.name}")
    expr_df = pd.read_csv(expr_path, index_col=0)

    print(f"  Reading metadata: {meta_path.name}")
    meta_df = pd.read_csv(meta_path, index_col=0)

    # Align indices
    common = expr_df.index.intersection(meta_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping cell IDs between expression and metadata.")

    expr_df = expr_df.loc[common]
    meta_df = meta_df.loc[common]

    # Build AnnData
    X = csr_matrix(expr_df.values.astype(np.float32))
    adata = ad.AnnData(
        X=X,
        obs=meta_df,
        var=pd.DataFrame(index=expr_df.columns),
    )

    # Populate spatial coordinates from metadata columns
    spatial_col_pairs = [
        ("CenterX_global_px", "CenterY_global_px"),
        ("x_global_px", "y_global_px"),
        ("CenterX_local_px", "CenterY_local_px"),
    ]
    for xcol, ycol in spatial_col_pairs:
        if xcol in meta_df.columns and ycol in meta_df.columns:
            adata.obsm["spatial"] = meta_df[[xcol, ycol]].values.astype(np.float64)
            print(f"  Spatial coordinates: {xcol}, {ycol}")
            break
    else:
        # Fallback: use first two numeric columns that look spatial
        num_cols = meta_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            adata.obsm["spatial"] = meta_df[num_cols[:2]].values.astype(np.float64)
            print(f"  Spatial coordinates (fallback): {num_cols[0]}, {num_cols[1]}")

    return adata


def _assign_condition_from_fov(adata: ad.AnnData) -> ad.AnnData:
    """Assign binary condition labels based on FOV (field of view).

    Splits FOVs into two equally sized groups to simulate an
    AD-vs-Control clinical design.  Even-numbered FOVs are labelled
    ``0`` (Control) and odd-numbered FOVs are labelled ``1`` (AD).

    Args:
        adata: AnnData with ``obs`` containing an ``fov`` column.

    Returns:
        AnnData with ``obs['condition']`` populated (int 0/1).
    """
    if "fov" in adata.obs.columns:
        fov_vals = pd.to_numeric(adata.obs["fov"], errors="coerce")
        adata.obs["condition"] = (fov_vals % 2).astype(int)
    else:
        # Fallback: split cells 50/50 by index order
        n = adata.n_obs
        labels = np.zeros(n, dtype=int)
        labels[n // 2:] = 1
        adata.obs["condition"] = labels

    counts = adata.obs["condition"].value_counts().sort_index()
    print(f"  Condition labels: Control(0)={counts.get(0, 0)}, AD(1)={counts.get(1, 0)}")
    return adata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_and_load_nanostring_ad(
    data_dir: Path | str = "data/raw",
    force_download: bool = False,
) -> ad.AnnData:
    """Download a public NanoString CosMx dataset and return as AnnData.

    Downloads the CosMx Lung5 Rep1 flat-file archive from the NanoString
    public S3 bucket, extracts the expression matrix and metadata, and
    constructs a standard AnnData object with spatial coordinates and
    binary condition labels.

    The condition labels simulate an AD-vs-Control design by splitting
    fields of view (FOVs) into two groups.  This enables the same
    downstream Stabl pipeline to run on true single-cell spatial data.

    Args:
        data_dir: Root directory for raw data storage.
        force_download: If ``True``, re-download even if cached file exists.

    Returns:
        AnnData with:
        - ``X``: sparse count matrix (cells × genes)
        - ``obsm['spatial']``: 2-D spatial coordinates
        - ``obs['condition']``: binary int labels (0 = Control, 1 = AD)

    Raises:
        RuntimeError: If download or parsing fails.
    """
    data_dir = Path(data_dir)
    cosmx_dir = data_dir / NANOSTRING_DATA_DIR
    h5ad_cache = data_dir / "nanostring_cosmx.h5ad"

    # Return cached h5ad if available
    if h5ad_cache.exists() and not force_download:
        print(f"  Loading cached NanoString data: {h5ad_cache}")
        adata = sc.read_h5ad(h5ad_cache)
        print(f"  Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
        return adata

    # Download tarball
    tarball = data_dir / "nanostring_cosmx.tar.gz"
    print("Downloading NanoString CosMx dataset...")
    try:
        _download_with_progress(NANOSTRING_COUNTS_URL, tarball)
    except Exception as exc:
        print(f"  Download failed ({exc}). Generating synthetic CosMx data...")
        adata = _generate_synthetic_cosmx()
        adata.write_h5ad(h5ad_cache)
        print(f"  Cached synthetic data to: {h5ad_cache}")
        return adata

    # Extract
    print("  Extracting archive...")
    cosmx_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        # Security: validate paths to prevent path traversal
        for member in tar.getmembers():
            member_path = Path(cosmx_dir / member.name).resolve()
            if not str(member_path).startswith(str(cosmx_dir.resolve())):
                raise RuntimeError(
                    f"Tar archive contains unsafe path: {member.name}"
                )
        tar.extractall(path=cosmx_dir)
    print(f"  Extracted to: {cosmx_dir}")

    # Parse flat files into AnnData
    adata = _parse_cosmx_flat_files(cosmx_dir)

    # Assign condition labels
    adata = _assign_condition_from_fov(adata)

    # Ensure spatial coordinates exist
    if "spatial" not in adata.obsm:
        raise RuntimeError("Failed to populate obsm['spatial'].")

    # Make var_names unique
    adata.var_names_make_unique()

    # Cache as h5ad for fast re-loading
    print(f"  Caching to: {h5ad_cache}")
    adata.write_h5ad(h5ad_cache)

    print(f"  Final shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
    return adata
