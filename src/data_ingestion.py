"""Download and cache public spatial transcriptomics datasets.

Supports two sources:
- ``squidpy``: 10x Visium mouse brain (V1_Adult_Mouse_Brain) with H&E image.
- ``geo_tbi``: GEO GSE319409 TBI spatial transcriptomics (Iteration 2).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad

__all__ = ["get_dataset", "download_squidpy_brain"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SQUIDPY_SAMPLE_ID = "V1_Adult_Mouse_Brain"
SQUIDPY_FILENAME = "mouse_brain_visium.h5ad"
GEO_FILENAME = "tbi_visium.h5ad"


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
    """Download a TBI spatial transcriptomics sample from GEO GSE319409.

    .. note::
        Iteration 2 feature — not yet implemented. Falls back to the
        squidpy dataset if called.

    Args:
        dest_dir: Directory to save the ``.h5ad`` file.

    Returns:
        Path to the saved ``.h5ad`` file.

    Raises:
        NotImplementedError: Always, until GEO integration is complete.
    """
    raise NotImplementedError(
        "GEO TBI ingestion (GSE319409) is planned for Iteration 2. "
        "Use source='squidpy' for the current PoC."
    )


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
