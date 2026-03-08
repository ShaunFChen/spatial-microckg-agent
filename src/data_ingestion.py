"""Download and cache public spatial transcriptomics datasets.

Supports two sources:
- ``squidpy``: 10x Visium mouse brain (V1_Adult_Mouse_Brain) with H&E image.
- ``geo_tbi``: GEO GSE319409 TBI spatial transcriptomics (3 TBI + 3 Sham,
  Vehicle-only, N=3 biological replicates per condition).
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import requests
import scanpy as sc
from tqdm import tqdm

__all__ = ["get_dataset", "download_squidpy_brain", "download_geo_tbi"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SQUIDPY_SAMPLE_ID = "V1_Adult_Mouse_Brain"
SQUIDPY_FILENAME = "mouse_brain_visium.h5ad"
GEO_FILENAME = "tbi_geo.h5ad"

GEO_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples"

# 6 Vehicle-only samples: 3 Sham + 3 TBI (N=3 biological replicates)
GEO_SAMPLE_REGISTRY: dict[str, dict[str, str | int]] = {
    # Sham + Vehicle
    "GSM9517469": {"prefix": "P3-57298", "condition": 0, "treatment": "Vehicle"},
    "GSM9517475": {"prefix": "P3-58693", "condition": 0, "treatment": "Vehicle"},
    "GSM9517481": {"prefix": "P3-60803", "condition": 0, "treatment": "Vehicle"},
    # TBI + Vehicle
    "GSM9517472": {"prefix": "P3-57301", "condition": 1, "treatment": "Vehicle"},
    "GSM9517477": {"prefix": "P3-58695", "condition": 1, "treatment": "Vehicle"},
    "GSM9517478": {"prefix": "P3-58696", "condition": 1, "treatment": "Vehicle"},
}

# Per-sample file suffixes to download from GEO
_MATRIX_SUFFIXES = ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]
_SPATIAL_SUFFIXES = [
    "tissue_positions.csv.gz",
    "scalefactors_json.json.gz",
    "tissue_lowres_image.png.gz",
]

_DOWNLOAD_CHUNK_SIZE = 8192  # 8 KB


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _geo_ftp_url(gsm_id: str, prefix: str, suffix: str) -> str:
    """Build the GEO supplementary-file download URL.

    Args:
        gsm_id: GEO sample accession (e.g. ``"GSM9517469"``).
        prefix: Library name prefix (e.g. ``"P3-57298"``).
        suffix: File suffix (e.g. ``"barcodes.tsv.gz"``).

    Returns:
        Full HTTPS URL for the supplementary file.
    """
    # GEO groups samples in thousand-ranges: GSM9517469 → GSM9517nnn
    gsm_stub = gsm_id[:7] + "nnn"
    filename = f"{gsm_id}_{prefix}_{suffix}"
    return f"{GEO_BASE_URL}/{gsm_stub}/{gsm_id}/suppl/{filename}"


def _download_file(url: str, dest: Path) -> bool:
    """Download a file with streaming and a tqdm progress bar.

    Skips the download if *dest* already exists on disk.

    Args:
        url: Remote URL to download.
        dest: Local path to save the file.

    Returns:
        ``True`` if the file was downloaded or already exists,
        ``False`` if the download failed.
    """
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  Warning: failed to download {dest.name}: {exc}")
        return False

    total = int(resp.headers.get("content-length", 0))
    with (
        open(dest, "wb") as fh,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"  {dest.name}",
            leave=False,
        ) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
            fh.write(chunk)
            pbar.update(len(chunk))

    return True


def _decompress_gz(gz_path: Path, out_path: Path) -> None:
    """Decompress a ``.gz`` file to *out_path*.

    Args:
        gz_path: Path to the gzip-compressed file.
        out_path: Destination path for the decompressed file.
    """
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def _assemble_single_sample(
    gsm_id: str,
    meta: dict[str, str | int],
    dest_dir: Path,
) -> ad.AnnData | None:
    """Download and assemble one GEO sample into an AnnData object.

    Downloads the 10x matrix files (barcodes, features, matrix) and
    optionally spatial coordinate files. Constructs an AnnData with
    experimental metadata attached to ``.obs``.

    Args:
        gsm_id: GEO sample accession.
        meta: Sample metadata from :data:`GEO_SAMPLE_REGISTRY`.
        dest_dir: Root directory for GEO downloads.

    Returns:
        The assembled AnnData, or ``None`` if matrix download failed.
    """
    prefix = str(meta["prefix"])
    sample_dir = dest_dir / "geo_tbi" / gsm_id
    mtx_dir = sample_dir / "filtered_feature_bc_matrix"
    mtx_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{gsm_id}] Downloading matrix files...")

    # --- Matrix files ---
    # Map GEO suffixes → standard 10x filenames expected by scanpy
    standard_names = {
        "barcodes.tsv.gz": "barcodes.tsv.gz",
        "features.tsv.gz": "features.tsv.gz",
        "matrix.mtx.gz": "matrix.mtx.gz",
    }
    for suffix in _MATRIX_SUFFIXES:
        url = _geo_ftp_url(gsm_id, prefix, suffix)
        local_gz = sample_dir / f"{gsm_id}_{prefix}_{suffix}"
        if not _download_file(url, local_gz):
            print(f"  [{gsm_id}] Matrix download failed — skipping sample.")
            return None
        # Copy/link into the standard-named directory for scanpy
        std_dest = mtx_dir / standard_names[suffix]
        if not std_dest.exists():
            shutil.copy2(local_gz, std_dest)

    # --- Read 10x matrix ---
    adata = sc.read_10x_mtx(str(mtx_dir), var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()
    print(f"  [{gsm_id}] Loaded: {adata.shape[0]} spots × {adata.shape[1]} genes")

    # --- Attach metadata ---
    adata.obs["sample_id"] = gsm_id
    adata.obs["condition"] = int(meta["condition"])
    adata.obs["treatment"] = str(meta["treatment"])
    adata.obs["batch"] = gsm_id

    # --- Spatial files (optional) ---
    print(f"  [{gsm_id}] Attempting spatial file download...")
    spatial_ok = True
    for suffix in _SPATIAL_SUFFIXES:
        url = _geo_ftp_url(gsm_id, prefix, suffix)
        local_gz = sample_dir / f"{gsm_id}_{prefix}_{suffix}"
        if not _download_file(url, local_gz):
            spatial_ok = False
            break

    if spatial_ok:
        try:
            adata = _attach_spatial_data(adata, gsm_id, prefix, sample_dir)
            print(f"  [{gsm_id}] Spatial coordinates attached.")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{gsm_id}] Warning: spatial parsing failed ({exc}). "
                  "Proceeding without spatial data.")
    else:
        print(f"  [{gsm_id}] Spatial files unavailable — proceeding without.")

    return adata


def _attach_spatial_data(
    adata: ad.AnnData,
    gsm_id: str,
    prefix: str,
    sample_dir: Path,
) -> ad.AnnData:
    """Parse CytAssist spatial files and attach to AnnData.

    Reads ``tissue_positions.csv`` (header-format), filters to
    ``in_tissue == 1``, and populates ``adata.obsm['spatial']``
    and ``adata.uns['spatial']``.

    Args:
        adata: Single-sample AnnData (barcodes as index).
        gsm_id: GEO accession for filename construction.
        prefix: Library prefix.
        sample_dir: Directory containing downloaded files.

    Returns:
        AnnData with spatial coordinates and metadata attached.
    """
    import matplotlib.image as mpimg

    # --- Tissue positions (CytAssist format: has header row) ---
    pos_gz = sample_dir / f"{gsm_id}_{prefix}_tissue_positions.csv.gz"
    pos_csv = sample_dir / "tissue_positions.csv"
    if not pos_csv.exists():
        _decompress_gz(pos_gz, pos_csv)

    positions = pd.read_csv(pos_csv, index_col=0)
    # CytAssist columns: barcode(index), in_tissue, array_row, array_col,
    # pxl_row_in_fullres, pxl_col_in_fullres
    positions = positions[positions["in_tissue"] == 1]

    # Align to AnnData barcodes
    common = adata.obs_names.intersection(positions.index)
    if len(common) == 0:
        raise ValueError("No matching barcodes between matrix and positions file.")
    positions = positions.loc[common]
    adata = adata[common, :]  # noqa: PLW2901 (intentional re-bind within helper)

    spatial_coords = positions[
        ["pxl_row_in_fullres", "pxl_col_in_fullres"]
    ].to_numpy(dtype=np.float64)
    adata.obsm["spatial"] = spatial_coords

    # --- Scale factors ---
    sf_gz = sample_dir / f"{gsm_id}_{prefix}_scalefactors_json.json.gz"
    sf_json = sample_dir / "scalefactors_json.json"
    if not sf_json.exists():
        _decompress_gz(sf_gz, sf_json)
    with open(sf_json) as fh:
        scalefactors = json.load(fh)

    # --- Low-res image ---
    img_gz = sample_dir / f"{gsm_id}_{prefix}_tissue_lowres_image.png.gz"
    img_png = sample_dir / "tissue_lowres_image.png"
    if not img_png.exists():
        _decompress_gz(img_gz, img_png)
    img_arr = mpimg.imread(str(img_png))

    # --- Populate uns['spatial'] dict (scanpy convention) ---
    library_id = gsm_id
    adata.uns["spatial"] = {
        library_id: {
            "images": {"lowres": img_arr},
            "scalefactors": scalefactors,
        }
    }
    return adata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_squidpy_brain(dest_dir: Path) -> Path:
    """Download the 10x Visium adult mouse brain dataset via squidpy.

    The dataset includes spatial coordinates and an H&E tissue image,
    making it suitable for spatial marker overlay plots.

    Args:
        dest_dir: Directory to save the ``.h5ad`` file.

    Returns:
        Path to the saved ``.h5ad`` file.
    """
    import squidpy as sq

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / SQUIDPY_FILENAME

    if out_path.exists():
        print(f"  Dataset already cached: {out_path}")
        return out_path

    print(f"  Downloading squidpy Visium dataset ({SQUIDPY_SAMPLE_ID})...")
    adata: ad.AnnData = sq.datasets.visium(SQUIDPY_SAMPLE_ID)
    adata.var_names_make_unique()
    adata.write_h5ad(out_path)
    print(f"  Saved to {out_path} — shape {adata.shape}")
    return out_path


def download_geo_tbi(dest_dir: Path) -> Path:
    """Download TBI and Sham spatial transcriptomics from GEO GSE319409.

    Downloads 6 Visium CytAssist samples (3 TBI + 3 Sham, Vehicle-only)
    from GEO, assembles each into an AnnData object with experimental
    metadata, and concatenates them into a single ``.h5ad`` file.

    Each spot receives the following ``obs`` columns:

    - ``sample_id`` — GEO accession (e.g. ``"GSM9517469"``).
    - ``condition`` — binary label (0 = Sham, 1 = TBI).
    - ``treatment`` — ``"Vehicle"`` for all selected samples.
    - ``batch`` — same as ``sample_id`` (for ComBat batch correction).

    Args:
        dest_dir: Directory to save the ``.h5ad`` file.

    Returns:
        Path to the saved concatenated ``.h5ad`` file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / GEO_FILENAME

    if out_path.exists():
        print(f"  GEO TBI dataset already cached: {out_path}")
        return out_path

    print("  Downloading GEO GSE319409 TBI dataset "
          f"({len(GEO_SAMPLE_REGISTRY)} samples)...")

    adatas: list[ad.AnnData] = []
    for gsm_id, meta in GEO_SAMPLE_REGISTRY.items():
        sample_adata = _assemble_single_sample(gsm_id, meta, dest_dir)
        if sample_adata is not None:
            adatas.append(sample_adata)

    if len(adatas) == 0:
        raise RuntimeError(
            "No GEO samples could be downloaded. Check network connectivity."
        )

    # Concatenate — inner join keeps only genes shared across all samples
    # Merge uns['spatial'] dicts before concat (concat drops uns)
    merged_spatial_uns: dict = {}
    for a in adatas:
        if "spatial" in a.uns:
            merged_spatial_uns.update(a.uns["spatial"])

    adata = ad.concat(adatas, join="inner")
    adata.obs_names_make_unique()

    if merged_spatial_uns:
        adata.uns["spatial"] = merged_spatial_uns

    # Ensure condition is integer
    adata.obs["condition"] = adata.obs["condition"].astype(int)

    cond_counts = adata.obs["condition"].value_counts()
    n_samples = adata.obs["sample_id"].nunique()
    print(f"\n  Assembled GEO TBI dataset: {adata.shape[0]} spots × "
          f"{adata.shape[1]} genes from {n_samples} samples")
    print(f"  Condition distribution: Sham(0)={cond_counts.get(0, 0)}, "
          f"TBI(1)={cond_counts.get(1, 0)}")

    adata.write_h5ad(out_path)
    print(f"  Saved to {out_path}")
    return out_path


def get_dataset(dest_dir: Path | str, source: str = "squidpy") -> Path:
    """Return the path to a spatial transcriptomics ``.h5ad`` dataset.

    Downloads the file on first call; subsequent calls return the cached
    path immediately.

    Args:
        dest_dir: Directory to store downloaded data.
        source: Dataset source — ``"squidpy"`` (default) or ``"geo_tbi"``.

    Returns:
        Absolute path to the ``.h5ad`` file.

    Raises:
        ValueError: If *source* is not a recognised dataset key.
    """
    dest_dir = Path(dest_dir)

    if source == "squidpy":
        return download_squidpy_brain(dest_dir)
    elif source == "geo_tbi":
        return download_geo_tbi(dest_dir)
    else:
        raise ValueError(
            f"Unknown dataset source '{source}'. Choose 'squidpy' or 'geo_tbi'."
        )
