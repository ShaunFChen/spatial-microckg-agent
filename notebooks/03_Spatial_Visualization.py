# %% [markdown]
# # 03 — Spatial Visualization
#
# **Pipeline Step 3 of 5**
#
# This notebook visualizes Stabl-selected biomarker gene expression via **spatial overlay** (when spatial coordinates are available) or **UMAP embedding** (fallback). A condition UMAP is also generated to confirm separation.
#
# ### Visualization targets
#
# | Category | Genes | Rationale |
# |----------|-------|-----------|
# | **Stabl top 5** | Prnp, Fth1, Calb1, Trf, Cst3 | Top stability-scored AD drivers from Notebook 02 |
# | **Anatomical baseline** | Prox1 | Hippocampus Dentate Gyrus marker for spatial verification |
# | **Classical AD baselines** | Trem2, Gfap | Standard neuroinflammation markers for comparison |
#
# ### Inputs
# | File | Description |
# |---|---|
# | `data/processed/ad_preprocessed.h5ad` | QC-filtered, normalized AnnData from Step 01 |
# | `cache/stabl_results_<hash>.pkl` | Stabl results from Step 02 |
#
# ### Outputs
# | File | Description |
# |---|---|
# | `assets/umap_<Gene>.png` / `assets/spatial_<Gene>.png` | Expression plots for top 5 markers + Prox1 + Trem2 + Gfap |
# | `assets/umap_condition.png` | UMAP colored by WT/AD condition |

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_pipeline import (
    load_adata,
    run_stabl_cached,
    plot_spatial_markers,
    remap_condition_labels,
    set_plot_defaults,
)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"
ASSETS_DIR = PROJECT_ROOT / "assets"

print("Imports ready.")

# Apply publication-quality global plot defaults
set_plot_defaults(fontsize=12, dpi=300)

# %% [markdown]
# ## 3.1 Load Data and Stabl Results
#
# Reload the preprocessed AD AnnData and cached Stabl results. Sort markers by descending stability score.

# %%
import pandas as pd

adata = load_adata(DATA_PROCESSED / "ad_preprocessed.h5ad")

stabl_result = run_stabl_cached(
    adata,
    cache_dir=CACHE_DIR,
    dataset_name="geo_ad",
    label_method="condition",
    n_bootstraps=50,
    prefilter="de",
)

df_features = pd.DataFrame({
    "Gene": stabl_result["selected_genes"],
    "Stability Score": [
        stabl_result["stability_scores"][g]
        for g in stabl_result["selected_genes"]
    ],
}).sort_values("Stability Score", ascending=False)
top_markers = df_features["Gene"].tolist()

print(f"Loaded {adata.shape[0]} spots, {stabl_result['n_selected']} Stabl markers")
print(f"Top 5: {top_markers[:5]}")

# %% [markdown]
# ## 3.2 Spatial Autocorrelation (Moran's I)
#
# Moran's I quantifies whether a gene's expression is spatially clustered (I > 0), random (I ≈ 0), or dispersed (I < 0). A statistically significant positive Moran's I (permutation p < 0.05) confirms that a biomarker marks a real tissue compartment rather than technical noise — a prerequisite for spatial drug targeting.
#
# We test all Stabl-selected genes against the spatial neighbourhood graph (k = 6 nearest spots). Results are displayed as a ranked table and subsequently passed to the spatial overlay plots as title annotations.

# %%
from src.spatial_analytics import compute_spatial_neighbors, compute_spatial_autocorr
from IPython.display import display
import pandas as pd

# Build spatial neighbourhood graph (k=6 nearest spots)
adata = compute_spatial_neighbors(adata, n_neighs=6)

# Compute Moran's I for Stabl-selected genes (100 permutations)
gene_list = list(stabl_result["selected_genes"])
moran_df = compute_spatial_autocorr(
    adata,
    genes=gene_list,
    mode="moran",
    n_perms=100,
)

# Display ranked table (significant genes only)
sig_moran = (
    moran_df[moran_df["pval_norm"] < 0.05]
    .sort_values("I", ascending=False)
    .copy()
)
sig_moran.index.name = "gene"
print(f"Spatially autocorrelated Stabl genes: {len(sig_moran)} / {len(moran_df)}")
display(
    sig_moran[["I", "pval_norm"]]
    .rename(columns={"I": "Moran's I", "pval_norm": "p-value (norm)"})
    .style.format({"Moran's I": "{:.4f}", "p-value (norm)": "{:.3e}"})
    .background_gradient(subset=["Moran's I"], cmap="YlOrRd")
)

# Build dict for plot_spatial_markers title annotations
# {gene: (moran_I, pval)} — only significant genes; others get no annotation
morans_scores = {
    gene: (float(row["I"]), float(row["pval_norm"]))
    for gene, row in moran_df.iterrows()
    if row["pval_norm"] < 0.05
}
print(f"morans_scores dict: {len(morans_scores)} entries")

# %% [markdown]
# ## 3.3 Generate Marker Plots
#
# For each of the top 5 Stabl markers (Prnp, Fth1, Calb1, Trf, Cst3), generates a spatial overlay or UMAP plot (fallback when spatial coordinates lack full image metadata). Prox1 (Hippocampal Dentate Gyrus) and classical AD markers Trem2/Gfap are appended automatically for anatomical verification and baseline comparison.

# %%
saved_plots = plot_spatial_markers(
    adata,
    markers=top_markers,
    save_dir=ASSETS_DIR,
    n_top=5,
    morans_scores=morans_scores,
)

print(f"\n{len(saved_plots)} spatial plots saved to {ASSETS_DIR}")

# %% [markdown]
# ## 3.4 Display Plots

# %%
from IPython.display import Image, display

for plot_path in saved_plots:
    display(Image(filename=str(plot_path), width=500))

# %% [markdown]
# ## 3.5 Spatial Anatomical Map (Leiden Clusters)
#
# UMAP embedding colored by Leiden cluster to verify that anatomical structure is preserved in the data. Each cluster broadly corresponds to a distinct brain region or cell-type niche.

# %%
import scanpy as sc
import matplotlib.pyplot as plt

# Compute Leiden clusters if not already present
if "leiden" not in adata.obs.columns:
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", subset=False)
    sc.pp.pca(adata, use_highly_variable=True)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=1.0)

# Ensure UMAP exists
if "X_umap" not in adata.obsm:
    sc.tl.umap(adata)

fig, ax = plt.subplots(figsize=(8, 8))
sc.pl.umap(adata, color="leiden", ax=ax, show=False,
           title="Anatomical Map (Leiden Clusters)", frameon=False)
plt.tight_layout()
out_path = ASSETS_DIR / "umap_leiden_clusters.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
display(Image(filename=str(out_path), width=500))
print(f"Clusters: {adata.obs['leiden'].nunique()}")

# %% [markdown]
# ## 3.6 Quantitative Comparison (Violin Plots)
#
# Side-by-side violin plots comparing expression of top Stabl markers between WT and AD conditions. These show the distributional shift that Stabl identified — even though the global UMAP is intermixed, individual marker distributions differ significantly.

# %%
remap_condition_labels(adata)

sc.pl.violin(adata, keys=["Prnp", "Fth1", "Calb1"], groupby="condition_label", rotation=0, show=False)
out_path = ASSETS_DIR / "violin_top3_condition.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
display(Image(filename=str(out_path), width=500))
print("Violin plots for: [Prnp, Fth1, Calb1]")

# %% [markdown]
# ## 3.7 Dot Plot by Condition
#
# Dot plot for the top 5 Stabl markers across conditions showing both mean expression level (color intensity) and fraction of spots expressing the marker (dot size).

# %%
sc.pl.dotplot(adata, var_names=["Prnp", "Fth1", "Calb1", "Trf", "Prox1"], groupby="condition_label", standard_scale="var", show=False)
out_path = ASSETS_DIR / "dotplot_top5_condition.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
display(Image(filename=str(out_path), width=500))
print("Dot plot for: [Prnp, Fth1, Calb1, Trf, Prox1]")

# %% [markdown]
# ## 3.8 Analytical Note
#
# > **Why is the global UMAP (WT vs AD) intermixed?**
# >
# > This is biologically expected in spatial transcriptomics. Anatomical and structural variance — cortex vs hippocampus vs white matter — dominates the principal components and therefore the global manifold. The disease signature is a nuanced secondary vector that operates *within* each anatomical niche, not across the entire tissue.
# >
# > Stabl properly extracts this secondary vector by evaluating each gene's discriminative power (WT vs AD) through L1-regularized logistic regression with bootstrapped stability selection. The violin plots above confirm that the selected markers (Prnp, Fth1, Calb1, etc.) show genuine distributional shifts between conditions, even though the global UMAP does not visually separate them.
# >
# > This is a fundamental distinction between **tissue-level spatial** data (anatomy-dominated) and **dissociated single-cell** data (cell-type-dominated), where condition-based UMAP separation is more common.

# %% [markdown]
# ## 3.9 High-Contrast Prox1 Spatial Overlay
#
# Publication-quality side-by-side comparison of **Prox1** expression 
# (Hippocampus Dentate Gyrus marker) between WT and AD sections. 
# The H&E histology is dimmed to ~15% opacity, the **Magma** colormap 
# maximizes signal contrast, and `vmax` is set to the 99th percentile 
# to compress the dynamic range so that the DG clusters appear 
# intensely bright.

# %%
from importlib import reload
import src.spatial_pipeline as sp
reload(sp)
from src.spatial_pipeline import plot_spatial_highcontrast
from IPython.display import Image, display

hc_path = plot_spatial_highcontrast(
    adata,
    gene="Prox1",
    save_path=ASSETS_DIR / "spatial_Prox1_highcontrast.png",
    wt_sample="GSM6171782",
    ad_sample="GSM6171784",
    cmap="magma",
    img_alpha=0.15,
    spot_scale=2.0,
    vmax_percentile=99.0,
)
display(Image(filename=str(hc_path), width=700))
