"""Download and cache public spatial transcriptomics datasets.

Supports two sources:
- ``squidpy``: 10x Visium mouse brain (V1_Adult_Mouse_Brain) with H&E image.
- ``geo_ad``: GEO GSE203424 Mouse Alzheimer's Disease spatial transcriptomics
  (3 PSAPP×CO + 3 WT×CO, Corn-Oil vehicle only, N=3 biological replicates
  per condition).
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

__all__ = ["get_dataset", "download_squidpy_brain", "download_geo_ad"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SQUIDPY_SAMPLE_ID = "V1_Adult_Mouse_Brain"
SQUIDPY_FILENAME = "mouse_brain_visium.h5ad"
GEO_AD_FILENAME = "ad_geo.h5ad"

GEO_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples"

# 6 Corn-Oil-only samples from GSE203424:
#   3 WT×CO (healthy, label 0) + 3 PSAPP×CO (AD model, label 1)
# Tamoxifen groups (WT×TAM, PSAPP×TAM) are strictly excluded.
GEO_AD_SAMPLE_REGISTRY: dict[str, dict[str, str | int]] = {
    # WT × Corn Oil (healthy controls)
    "GSM6171782": {"prefix": "WT_CO1", "condition": 0, "genotype": "WT"},
    "GSM6171786": {"prefix": "WT_CO2", "condition": 0, "genotype": "WT"},
    "GSM6171790": {"prefix": "WT_CO3", "condition": 0, "genotype": "WT"},
    # PSAPP × Corn Oil (Alzheimer's disease model)
    "GSM6171784": {"prefix": "PSAPP_CO1", "condition": 1, "genotype": "PSAPP"},
    "GSM6171788": {"prefix": "PSAPP_CO2", "condition": 1, "genotype": "PSAPP"},
    "GSM6171792": {"prefix": "PSAPP_CO3", "condition": 1, "genotype": "PSAPP"},
}

# Per-sample supplementary file suffixes on GEO for GSE203424
_AD_H5_SUFFIX = "filtered_feature_bc_matrix.h5"
_AD_SPATIAL_SUFFIXES = [
    "tissue_positions_list.csv.gz",
    "scalefactors_json.json.gz",
    "tissue_lowres_image.png.gz",
]

_DOWNLOAD_CHUNK_SIZE = 8192  # 8 KB


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _geo_suppl_url(gsm_id: str, prefix: str, suffix: str) -> str:
    """Build the GEO supplementary-file download URL.

    Args:
        gsm_id: GEO sample accession (e.g. ``"GSM6171782"``).
        prefix: Sample name prefix (e.g. ``"WT_CO1"``).
        suffix: File suffix (e.g. ``"filtered_feature_bc_matrix.h5"``).

    Returns:
        Full HTTPS URL for the supplementary file.
    """
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
    """Download and assemble one GEO GSE203424 sample into an AnnData.

    Downloads the 10x ``filtered_feature_bc_matrix.h5`` and optionally
    spatial coordinate files. Constructs an AnnData with experimental
    metadata attached to ``.obs``.

    Args:
        gsm_id: GEO sample accession.
        meta: Sample metadata from :data:`GEO_AD_SAMPLE_REGISTRY`.
        dest_dir: Root directory for GEO downloads.

    Returns:
        The assembled AnnData, or ``None`` if the download failed.
    """
    prefix = str(meta["prefix"])
    sample_dir = dest_dir / "geo_ad" / gsm_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    # --- Download filtered_feature_bc_matrix.h5 ---
    h5_url = _geo_suppl_url(gsm_id, prefix, _AD_H5_SUFFIX)
    h5_local = sample_dir / f"{gsm_id}_{prefix}_{_AD_H5_SUFFIX}"
    print(f"\n  [{gsm_id}] Downloading {_AD_H5_SUFFIX}...")
    if not _download_file(h5_url, h5_local):
        print(f"  [{gsm_id}] H5 download failed — skipping sample.")
        return None

    # --- Read 10x H5 ---
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variable names are not unique")
        adata = sc.read_10x_h5(str(h5_local))
    adata.var_names_make_unique()
    print(f"  [{gsm_id}] Loaded: {adata.shape[0]} spots × {adata.shape[1]} genes")

    # --- Attach metadata ---
    adata.obs["sample_id"] = gsm_id
    adata.obs["condition"] = int(meta["condition"])
    adata.obs["genotype"] = str(meta["genotype"])
    adata.obs["batch"] = gsm_id

    # --- Spatial files (optional) ---
    print(f"  [{gsm_id}] Attempting spatial file download...")
    spatial_ok = True
    for suffix in _AD_SPATIAL_SUFFIXES:
        url = _geo_suppl_url(gsm_id, prefix, suffix)
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
    """Parse spatial files and attach to AnnData.

    Auto-detects the tissue-positions format:

    - **SpaceRanger ≤ 1.3** — ``tissue_positions_list.csv`` with no
      header; columns are ``barcode, in_tissue, array_row, array_col,
      pxl_row_in_fullres, pxl_col_in_fullres``.
    - **CytAssist / SpaceRanger ≥ 2.0** — ``tissue_positions.csv``
      with a header row.

    Args:
        adata: Single-sample AnnData (barcodes as index).
        gsm_id: GEO accession for filename construction.
        prefix: Library prefix.
        sample_dir: Directory containing downloaded files.

    Returns:
        AnnData with spatial coordinates and metadata attached.
    """
    import matplotlib.image as mpimg

    # --- Tissue positions ---
    # Try SpaceRanger 1.x name first, then CytAssist name
    _POSITION_COLS = [
        "barcode", "in_tissue", "array_row", "array_col",
        "pxl_row_in_fullres", "pxl_col_in_fullres",
    ]

    pos_csv = sample_dir / "tissue_positions_list.csv"
    pos_gz_list = sample_dir / f"{gsm_id}_{prefix}_tissue_positions_list.csv.gz"
    pos_gz_cytassist = sample_dir / f"{gsm_id}_{prefix}_tissue_positions.csv.gz"

    if not pos_csv.exists():
        if pos_gz_list.exists():
            _decompress_gz(pos_gz_list, pos_csv)
        elif pos_gz_cytassist.exists():
            pos_csv = sample_dir / "tissue_positions.csv"
            _decompress_gz(pos_gz_cytassist, pos_csv)

    # Auto-detect header presence
    with open(pos_csv) as fh:
        first_line = fh.readline().strip()
    if "in_tissue" in first_line or "barcode" in first_line:
        positions = pd.read_csv(pos_csv, index_col=0)
    else:
        positions = pd.read_csv(
            pos_csv, header=None, names=_POSITION_COLS, index_col=0,
        )

    positions = positions[positions["in_tissue"] == 1]

    # Align to AnnData barcodes
    common = adata.obs_names.intersection(positions.index)
    if len(common) == 0:
        raise ValueError("No matching barcodes between matrix and positions file.")
    positions = positions.loc[common]
    adata = adata[common, :].copy()  # noqa: PLW2901

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


def download_geo_ad(dest_dir: Path) -> Path:
    """Download Mouse AD spatial transcriptomics from GEO GSE203424.

    Downloads 6 Visium samples (3 PSAPP×CO + 3 WT×CO, Corn-Oil vehicle
    only) from GEO, assembles each into an AnnData object with
    experimental metadata, and concatenates them into a single
    ``.h5ad`` file.  Tamoxifen intervention groups are excluded.

    Each spot receives the following ``obs`` columns:

    - ``sample_id`` — GEO accession (e.g. ``"GSM6171782"``).
    - ``condition`` — binary label (0 = WT, 1 = AD/PSAPP).
    - ``genotype`` — ``"WT"`` or ``"PSAPP"``.
    - ``batch`` — same as ``sample_id`` (for ComBat batch correction).

    Args:
        dest_dir: Directory to save the ``.h5ad`` file.

    Returns:
        Path to the saved concatenated ``.h5ad`` file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / GEO_AD_FILENAME

    if out_path.exists():
        print(f"  GEO AD dataset already cached: {out_path}")
        return out_path

    print("  Downloading GEO GSE203424 Mouse AD dataset "
          f"({len(GEO_AD_SAMPLE_REGISTRY)} samples)...")

    adatas: list[ad.AnnData] = []
    for gsm_id, meta in GEO_AD_SAMPLE_REGISTRY.items():
        sample_adata = _assemble_single_sample(gsm_id, meta, dest_dir)
        if sample_adata is not None:
            adatas.append(sample_adata)

    if len(adatas) == 0:
        raise RuntimeError(
            "No GEO samples could be downloaded. Check network connectivity."
        )

    # Concatenate — inner join keeps only genes shared across all samples
    merged_spatial_uns: dict = {}
    for a in adatas:
        if "spatial" in a.uns:
            merged_spatial_uns.update(a.uns["spatial"])

    # Make obs_names unique per sample before concat to avoid warnings
    for a in adatas:
        a.obs_names = a.obs["sample_id"].astype(str) + "_" + a.obs_names.astype(str)
    adata = ad.concat(adatas, join="inner")

    if merged_spatial_uns:
        adata.uns["spatial"] = merged_spatial_uns

    # Ensure condition is integer
    adata.obs["condition"] = adata.obs["condition"].astype(int)

    cond_counts = adata.obs["condition"].value_counts()
    n_samples = adata.obs["sample_id"].nunique()
    print(f"\n  Assembled GEO AD dataset: {adata.shape[0]} spots × "
          f"{adata.shape[1]} genes from {n_samples} samples")
    print(f"  Condition distribution: WT(0)={cond_counts.get(0, 0)}, "
          f"AD(1)={cond_counts.get(1, 0)}")

    adata.write_h5ad(out_path)
    print(f"  Saved to {out_path}")
    return out_path


def get_dataset(dest_dir: Path | str, source: str = "squidpy") -> Path:
    """Return the path to a spatial transcriptomics ``.h5ad`` dataset.

    Downloads the file on first call; subsequent calls return the cached
    path immediately.

    Args:
        dest_dir: Directory to store downloaded data.
        source: Dataset source — ``"squidpy"`` (default) or ``"geo_ad"``.

    Returns:
        Absolute path to the ``.h5ad`` file.

    Raises:
        ValueError: If *source* is not a recognised dataset key.
    """
    dest_dir = Path(dest_dir)

    if source == "squidpy":
        return download_squidpy_brain(dest_dir)
    elif source == "geo_ad":
        return download_geo_ad(dest_dir)
    else:
        raise ValueError(
            f"Unknown dataset source '{source}'. Choose 'squidpy' or 'geo_ad'."
        )
