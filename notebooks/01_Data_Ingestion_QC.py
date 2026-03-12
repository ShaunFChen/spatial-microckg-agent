# %% [markdown]
# # 01 — Data Ingestion & Quality Control
#
# **Pipeline Step 1 of 5**
#
# This notebook downloads **6 Corn-Oil-only spatial transcriptomics samples** (one coronal brain section per mouse: 3 WT, 3 PSAPP/AD) from GEO [GSE203424](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE203424) and applies a standard QC workflow before saving the preprocessed data for downstream analysis.
#
# **Study design (Corn-Oil-only subset):**
# | Group | N | GEO Accessions |
# |-------|---|----------------|
# | WT × CO (healthy controls) | 3 | GSM6171782, GSM6171786, GSM6171790 |
# | PSAPP × CO (Alzheimer's model) | 3 | GSM6171784, GSM6171788, GSM6171792 |
#
# The platform is **10x Visium** (mouse brain sections, Space Ranger v1.3.1). Each sample contains ~3,000–5,000 capture spots profiling ~32,000 genes with spatial coordinates and tissue images.
#
# ### Outputs
# | File | Description |
# |---|---|
# | `data/processed/ad_preprocessed.h5ad` | QC-filtered, normalized AnnData ready for feature selection |

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import get_dataset
from src.spatial_pipeline import load_adata, run_qc, normalize

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print("Imports ready.")

# %% [markdown]
# ## 1.1 Download Dataset
#
# We download the 6 Corn-Oil-only samples from GEO GSE203424. For each sample the pipeline fetches the 10x H5 expression matrix, spatial coordinates (`tissue_positions_list.csv`), scale factors, and tissue images from the GEO FTP server.
#
# The data is cached locally as a single concatenated `.h5ad` file so re-runs skip the download step. Per-sample metadata (`sample_id`, `condition`, `batch`) is embedded in `adata.obs`.

# %%
h5ad_path = get_dataset(DATA_RAW, source="geo_ad")
adata = load_adata(h5ad_path)

print(f"\nRaw dataset: {adata.shape[0]} spots × {adata.shape[1]} genes")
print(f"Samples: {adata.obs['sample_id'].nunique()}")
print(f"Conditions: {dict(adata.obs['condition'].value_counts())}")

# %%
from IPython.display import display

print("Sample metadata snapshot (one row per sample):")
display(
    adata.obs[["sample_id", "condition", "genotype"]]
    .drop_duplicates()
    .sort_values("sample_id")
    .reset_index(drop=True)
)

print("\nSpot counts per sample:")
print(
    adata.obs.groupby(["sample_id", "genotype"])
    .size()
    .rename("n_spots")
    .to_string()
)

# %% [markdown]
# ## 1.2 Quality Control
#
# Standard QC filters applied:
#
# 1. **Minimum genes per spot (≥ 200):** Remove empty or damaged spots.
# 2. **Minimum cells per gene (≥ 3):** Exclude rarely detected genes.
# 3. **Maximum mitochondrial gene percentage (< 30%):** Remove stressed/dying spots. Brain tissue naturally has higher mitochondrial fractions due to the high metabolic demand of neurons, so we use a relaxed threshold of 30%.

# %%
adata = run_qc(adata)
print(f"\nPost-QC: {adata.shape[0]} spots × {adata.shape[1]} genes")

# %% [markdown]
# ## 1.3 Normalization
#
# After QC filtering, we apply two normalization steps:
#
# 1. **Library-size normalization (CPM-like):** Each spot's total UMI count is scaled to a common target of 10,000 counts. This corrects for variation in sequencing depth between spots, making expression values comparable.
# 2. **Log-transformation (`log1p`):** A `log(x + 1)` transform is applied to stabilize variance and compress the dynamic range. Highly expressed genes (e.g., ribosomal, mitochondrial) would otherwise dominate variance-based analyses.
#
# Raw counts are preserved in `adata.raw` so that they remain available for differential expression and spatial plotting (which expect un-normalized or separately normalized values).

# %%
adata = normalize(adata)
print(f"\nNormalized: {adata.shape[0]} spots × {adata.shape[1]} genes")
print(f"Raw layer preserved: {adata.raw is not None}")

# %% [markdown]
# ## 1.4 Save Preprocessed Data
#
# The preprocessed AnnData is serialized to `.h5ad` format. Downstream notebooks (Stabl, visualization, KG) load this checkpoint directly.

# %%
out_path = DATA_PROCESSED / "ad_preprocessed.h5ad"
adata.write_h5ad(out_path)
print(f"Saved preprocessed AnnData to {out_path}")
print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")
