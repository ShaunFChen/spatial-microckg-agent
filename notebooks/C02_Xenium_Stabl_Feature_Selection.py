# %% [markdown]
# # C02 — Xenium Stabl Feature Selection (Spatial Niches)
#
# **Parallel Track C: Human True Single-Cell Spatial Niche (10x Xenium)**
#
# This notebook applies the Stabl stability-selection pipeline to the
# niche-labelled Xenium data from C01. Intermediate (NaN) cells are dropped
# so that the binary classifier operates on a clean Disease Niche (1) vs.
# Healthy Background (0) comparison — enabling discovery of
# plaque-induced neuroinflammatory targets purely from spatial context.
#
# ### Pipeline
# 1. Load niche-labelled Xenium checkpoint from C01
# 2. Drop intermediate (NaN) cells for clean binary classification
# 3. Run standard QC + normalization on the niche-filtered subset
# 4. Run Wilcoxon DE baseline: Disease Niche vs. Healthy Background
# 5. Run Stabl stability selection on DE-filtered candidates
# 6. Generate Volcano plot with Stabl-selected biomarkers highlighted
#
# ### Inputs
# | File | Description |
# |---|---|
# | `data/processed/xenium_niche.h5ad` | Niche-labelled Xenium AnnData from C01 |
#
# ### Outputs
# | File | Description |
# |---|---|
# | `cache/stabl_results_<hash>.pkl` | Full Stabl result dictionary for Xenium niches |
# | `cache/stabl_features_<hash>.csv` | Selected genes with stability scores |
# | `assets/xenium_volcano_de.png` | Volcano plot: genome-wide DE |
# | `assets/xenium_volcano_stabl.png` | Volcano plot: DE + Stabl overlay |

# %%
import warnings
warnings.filterwarnings("ignore", module="tqdm")

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_pipeline import (
    load_adata,
    run_qc,
    normalize,
    run_stabl_cached,
    set_plot_defaults,
)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

print("Imports ready.")

# %% [markdown]
# ## C2.1 Load Niche-Labelled Xenium Data
#
# Load the checkpoint from C01 and drop intermediate (NaN) cells to produce
# a clean binary classification target.

# %%
import numpy as np

adata = load_adata(DATA_PROCESSED / "xenium_niche.h5ad")
print(f"\nLoaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

# Drop NaN condition cells
n_before = adata.n_obs
mask_labelled = ~adata.obs["condition"].isna()
adata = adata[mask_labelled].copy()
n_after = adata.n_obs
print(f"Dropped intermediate cells: {n_before:,} → {n_after:,}")

# Ensure condition is integer
adata.obs["condition"] = adata.obs["condition"].astype(int)
print(f"Conditions: {dict(adata.obs['condition'].value_counts())}")

# %% [markdown]
# ## C2.2 QC & Normalization
#
# Apply standard QC filtering and normalization to the niche-filtered subset
# before DE and Stabl analysis.

# %%
adata = run_qc(adata)
print(f"\nPost-QC: {adata.shape[0]} cells × {adata.shape[1]} genes")

adata = normalize(adata)
print(f"Normalized: {adata.shape[0]} cells × {adata.shape[1]} genes")
print(f"Raw layer preserved: {adata.raw is not None}")

# %% [markdown]
# ## C2.3 Genome-Wide Differential Expression & Volcano Plot
#
# Wilcoxon rank-sum test: Disease Niche (1) vs. Healthy Background (0).
# Since these labels are data-driven (proximity to plaque hotspots), any
# DE signature reflects the local microenvironment rather than batch effects
# or pre-existing clinical annotations.

# %%
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

set_plot_defaults(fontsize=12, dpi=300)

# Ensure condition is categorical string for scanpy
adata.obs["condition"] = adata.obs["condition"].astype(str).astype("category")

# Wilcoxon DE: Disease Niche (1) vs. Healthy Background (0)
sc.tl.rank_genes_groups(
    adata,
    groupby="condition",
    groups=["1"],
    reference="0",
    method="wilcoxon",
    key_added="wilcoxon_xenium",
)

de_df = sc.get.rank_genes_groups_df(adata, group="1", key="wilcoxon_xenium")
de_df = de_df.rename(
    columns={"names": "gene", "logfoldchanges": "log2FC", "pvals_adj": "pval_adj"}
)
de_df["pval_adj"] = de_df["pval_adj"].clip(lower=1e-300)
de_df["neg_log10_pval"] = -np.log10(de_df["pval_adj"])

# Clean up
del adata.uns["wilcoxon_xenium"]
import gc; gc.collect()

# Classify genes
FC_THR = 0.5
FDR_THR = 0.05
de_df["sig"] = "ns"
de_df.loc[(de_df["pval_adj"] < FDR_THR) & (de_df["log2FC"] > FC_THR), "sig"] = "up"
de_df.loc[(de_df["pval_adj"] < FDR_THR) & (de_df["log2FC"] < -FC_THR), "sig"] = "down"

n_up = (de_df["sig"] == "up").sum()
n_down = (de_df["sig"] == "down").sum()
n_ns = (de_df["sig"] == "ns").sum()
print(f"DE results: {n_up:,} up  |  {n_down:,} down  |  {n_ns:,} ns  (total {len(de_df):,})")

# ---------- Volcano Plot 1: DE ----------
COLOR_MAP = {"ns": "#CCCCCC", "up": "#1F77B4", "down": "#D62728"}
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("white")

for cat, zorder in [("ns", 1), ("down", 3), ("up", 3)]:
    sub = de_df[de_df["sig"] == cat]
    sz = 8 if cat == "ns" else 30
    alpha = 0.25 if cat == "ns" else 0.7
    ax.scatter(
        sub["log2FC"], sub["neg_log10_pval"],
        s=sz, alpha=alpha, c=COLOR_MAP[cat], linewidths=0, zorder=zorder,
    )

# Reference lines
pval_line = -np.log10(FDR_THR)
ax.axhline(pval_line, color="#888", ls="--", lw=0.8, zorder=0)
ax.axvline(FC_THR, color="#888", ls=":", lw=0.7, zorder=0)
ax.axvline(-FC_THR, color="#888", ls=":", lw=0.7, zorder=0)

# Significance counts
ax.text(0.02, 0.03, f"Down: {n_down:,}", transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="#D62728", va="bottom")
ax.text(0.98, 0.03, f"Up: {n_up:,}", transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="#1F77B4", va="bottom", ha="right")

ax.set_xlabel("log2 Fold Change (Disease Niche vs. Background)", fontsize=12)
ax.set_ylabel("−log10 (Adjusted p-value)", fontsize=12)
ax.set_title(
    "10x Xenium — Spatial Niche DE (Wilcoxon)\n"
    "Disease Niche (near plaque) vs. Healthy Background (far from plaque)",
    fontsize=12, pad=10,
)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
vol1_path = ASSETS_DIR / "xenium_volcano_de.png"
fig.savefig(vol1_path, dpi=300, bbox_inches="tight")
plt.close(fig)
display(Image(filename=str(vol1_path), width=650))
print(f"Volcano plot saved → {vol1_path}")

# %% [markdown]
# ## C2.4 Stabl Feature Selection (on DE-Filtered Candidates)
#
# Stabl receives the niche-based DE-filtered candidate pool as input. The
# ``condition`` labels are the algorithmically defined niche assignments
# (Disease = 1, Healthy Background = 0). Results are cached by parameter hash.

# %%
stabl_result = run_stabl_cached(
    adata,
    cache_dir=CACHE_DIR,
    dataset_name="xenium_niche",
    label_method="condition",
    n_bootstraps=50,
    prefilter="de",
)

print(f"\nStabl results:")
print(f"  Features selected: {stabl_result['n_selected']}")
print(f"  FDP+ threshold: {stabl_result['threshold']:.4f}")
print(f"  Minimum FDP+: {stabl_result['fdr']:.4f}")

# %% [markdown]
# ## C2.5 Review Selected Features
#
# Stabl-certified biomarker genes that distinguish the Disease Niche
# (plaque-proximal) from the Healthy Background (plaque-distal). These
# represent plaque-induced neuroinflammatory targets discovered purely
# from spatial context.

# %%
df_features = pd.DataFrame({
    "Gene": stabl_result["selected_genes"],
    "Stability Score": [
        stabl_result["stability_scores"][g]
        for g in stabl_result["selected_genes"]
    ],
}).sort_values("Stability Score", ascending=False).reset_index(drop=True)

print(f"\nStabl-certified niche biomarker genes ({len(df_features)} total):")
display(df_features.head(20))

print(f"\nScore statistics:")
print(f"  Mean: {df_features['Stability Score'].mean():.4f}")
print(f"  Median: {df_features['Stability Score'].median():.4f}")
print(f"  Min: {df_features['Stability Score'].min():.4f}")
print(f"  Max: {df_features['Stability Score'].max():.4f}")
print(f"  Genes with score = 1.0: {(df_features['Stability Score'] == 1.0).sum()}")

# %% [markdown]
# ## C2.6 Volcano Plot — Stabl Overlay
#
# Same genome-wide DE background as §C2.3, now with **Stabl-selected
# niche biomarkers** highlighted in gold. These genes were selected for
# reproducibility across bootstrap iterations — they represent the most
# stable molecular discriminators of plaque-proximal vs. plaque-distal
# microenvironments.

# %%
# Mark Stabl genes in the genome-wide DE table
de_df["is_stabl"] = de_df["gene"].isin(set(stabl_result["selected_genes"]))
n_stabl = de_df["is_stabl"].sum()

# ---------- Volcano Plot 2: DE + Stabl overlay ----------
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("white")

# Background: DE-coloured genes
for cat, zorder in [("ns", 1), ("down", 3), ("up", 3)]:
    sub = de_df[(de_df["sig"] == cat) & ~de_df["is_stabl"]]
    sz = 8 if cat == "ns" else 30
    alpha = 0.25 if cat == "ns" else 0.5
    ax.scatter(
        sub["log2FC"], sub["neg_log10_pval"],
        s=sz, alpha=alpha, c=COLOR_MAP[cat], linewidths=0, zorder=zorder,
    )

# Foreground: Stabl-selected genes
fg = de_df[de_df["is_stabl"]]
ax.scatter(
    fg["log2FC"], fg["neg_log10_pval"], s=80, alpha=0.95, c="#FF8C00",
    linewidths=0.8, edgecolors="black", zorder=10,
    label=f"Stabl-selected (n={n_stabl})",
)

# Annotate Stabl gene names
for _, row in fg.iterrows():
    ax.annotate(
        row["gene"],
        xy=(row["log2FC"], row["neg_log10_pval"]),
        xytext=(6, 4), textcoords="offset points",
        fontsize=9, color="#333", fontweight="bold",
    )

# Reference lines
ax.axhline(pval_line, color="#888", ls="--", lw=0.8, zorder=0)
ax.axvline(FC_THR, color="#888", ls=":", lw=0.7, zorder=0)
ax.axvline(-FC_THR, color="#888", ls=":", lw=0.7, zorder=0)

# Counts
ax.text(0.02, 0.03, f"Down: {n_down:,}", transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="#D62728", va="bottom")
ax.text(0.98, 0.03, f"Up: {n_up:,}", transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="#1F77B4", va="bottom", ha="right")
ax.text(0.50, 0.03, f"Stabl: {n_stabl}", transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="#FF8C00", va="bottom", ha="center")

ax.set_xlabel("log2 Fold Change (Disease Niche vs. Background)", fontsize=12)
ax.set_ylabel("−log10 (Adjusted p-value)", fontsize=12)
ax.set_title(
    "10x Xenium Niches — DE + Stabl-Selected Biomarkers\n"
    "Plaque-Induced Neuroinflammatory Targets",
    fontsize=12, pad=10,
)
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
vol2_path = ASSETS_DIR / "xenium_volcano_stabl.png"
fig.savefig(vol2_path, dpi=300, bbox_inches="tight")
plt.close(fig)
display(Image(filename=str(vol2_path), width=650))
print(f"Volcano plot saved → {vol2_path}")
