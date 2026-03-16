# %% [markdown]
# # B03 — NanoString Micro-CKG & LLM Agent
#
# **Parallel Track B: Human True Single-Cell Spatial (NanoString CosMx)**
#
# This notebook demonstrates **end-to-end pipeline modularity** by routing
# the NanoString CosMx Stabl-selected human genes through the Knowledge
# Graph and LLM Agent stages — the same infrastructure used for Phase 1
# Visium mouse data. Because the NanoString panel targets **human genes
# directly**, no mouse→human ortholog mapping is needed.
#
# ### Pipeline
# 1. Load Stabl-selected genes from Track B02 (or run Stabl inline)
# 2. Fetch GO/KEGG/Reactome pathways via Enrichr for human targets
# 3. Fetch STRING PPI for the human gene set
# 4. Build a human-specific Micro-CKG with PathwayGene/PPI edges
# 5. Query the LLM agent for pathway interpretation
#
# ### Inputs
# | File | Description |
# |---|---|
# | `data/processed/nanostring_preprocessed.h5ad` | QC-filtered NanoString AnnData from B01 |
# | `cache/stabl_results_<hash>.pkl` | Stabl results from B02 (if cached) |
#
# ### Outputs
# | File | Description |
# |---|---|
# | `cache/nanostring_micro_ckg.graphml` | Human-specific Micro-CKG |
# | `assets/nanostring_kg_subgraph.png` | KG visualization |

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import warnings
warnings.filterwarnings("ignore", module="tqdm")

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

print("Imports ready.")

# %% [markdown]
# ## B3.1 Load NanoString Data & Obtain Stabl Results
#
# Load the preprocessed NanoString AnnData from B01 and retrieve the
# Stabl-selected human genes. If B01/B02 have not been run yet (i.e.
# the data does not exist on disk), we fall back to running the Stabl
# pipeline inline on the existing AD preprocessed data (Phase 1) and
# map to human orthologs — proving that the KG/LLM code is fully
# modular and data-source agnostic.

# %%
import numpy as np
import pandas as pd
import scanpy as sc

from src.spatial_pipeline import load_adata, run_stabl_cached

nanostring_path = DATA_PROCESSED / "nanostring_preprocessed.h5ad"
USE_NANOSTRING = nanostring_path.exists()

if USE_NANOSTRING:
    print("Loading NanoString preprocessed data...")
    adata = load_adata(nanostring_path)

    # Run Stabl if not cached
    stabl_result = run_stabl_cached(
        adata,
        cache_dir=CACHE_DIR,
        dataset_name="nanostring_cosmx",
        label_method="condition",
        n_bootstraps=50,
        prefilter="de",
    )
    # NanoString genes are already human — no ortholog mapping needed
    human_genes = list(stabl_result["selected_genes"])
    print(f"\nNanoString Stabl genes (human): {human_genes}")
else:
    print("NanoString data not found — using Phase 1 AD data + ortholog mapping...")
    adata = load_adata(DATA_PROCESSED / "ad_preprocessed.h5ad")

    stabl_result = run_stabl_cached(
        adata,
        cache_dir=CACHE_DIR,
        dataset_name="geo_ad",
        label_method="condition",
        n_bootstraps=50,
        prefilter="de",
    )

    # Map mouse genes → human orthologs
    from src.external_knowledge import map_orthologs
    ortho_df = map_orthologs(stabl_result["selected_genes"])
    human_genes = ortho_df["human_symbol"].dropna().tolist()
    print(f"\nMouse Stabl genes: {stabl_result['selected_genes']}")
    print(f"Human orthologs: {human_genes}")

print(f"\n{len(human_genes)} human genes for KG construction.")

# %% [markdown]
# ## B3.2 Fetch Pathways, Diseases, Drugs & PPIs
#
# Query Enrichr for pathway enrichment, OpenTargets for disease
# associations, ChEMBL for drug targets, and STRING for protein–protein
# interactions. These form the biological context layer of the Micro-CKG.
# Results are cached locally to avoid redundant API calls.

# %%
from src.external_knowledge import (
    run_go_enrichment, get_string_ppi, get_disease_associations, get_drug_targets,
)

# Pathway enrichment
enrich_df = run_go_enrichment(human_genes)
if enrich_df is not None and not enrich_df.empty:
    sig_enrich = enrich_df[enrich_df["Adjusted P-value"] < 0.05]
    print(f"Significant pathways: {len(sig_enrich)} (of {len(enrich_df)} total)")
    from IPython.display import display
    display(sig_enrich.head(10)[["Term", "Adjusted P-value", "Gene_set_library", "Genes"]])
else:
    print("No significant enrichment results.")
    sig_enrich = pd.DataFrame()

# Gene–disease associations (OpenTargets — needs Ensembl IDs)
# For human data the ortho_df may not be available; resolve IDs automatically
disease_df = get_disease_associations(human_genes)
if disease_df is not None and not disease_df.empty:
    print(f"\nDisease associations: {disease_df['gene'].nunique()} genes, {len(disease_df)} entries")
    display(disease_df.head(10))
else:
    print("\nNo disease associations found.")

# Drug-target associations (ChEMBL)
drug_df = get_drug_targets(human_genes)
if drug_df is not None and not drug_df.empty:
    print(f"\nDrug targets: {drug_df['gene'].nunique()} genes, {len(drug_df)} drugs")
    display(drug_df.head(10))
else:
    print("\nNo drug-target associations found.")

# STRING PPI
ppi_df = get_string_ppi(human_genes)
if ppi_df is not None and not ppi_df.empty:
    print(f"\nSTRING PPI: {len(ppi_df)} interactions")
    display(ppi_df.head(10))
else:
    print("\nNo PPI data returned from STRING.")

# %% [markdown]
# ## B3.3 Build Human-Specific Micro-CKG via BioCypher
#
# We route the NanoString (or fallback AD) data through the **same
# BioCypher-backed `build_micro_ckg()`** used in Phase 1 — proving
# full pipeline modularity. BioCypher validates every entity against the
# Biolink ontology, deduplicates nodes, and produces a schema-validated
# NetworkX DiGraph.
#
# For human data, we use an **identity ortholog map** (human→human) so
# that the same gene→pathway/disease/PPI edge generators work unchanged.

# %%
from src.spatial_pipeline import compute_clusters, annotate_clusters
from src.biocypher_adapter import build_micro_ckg, build_micro_ckg_agent, save_graph

# Leiden clustering (needed for DE-filtered edges)
adata = compute_clusters(adata, n_hvgs=2000)
cluster_annotation = annotate_clusters(adata)
print(f"Leiden clusters: {adata.obs['leiden'].nunique()}")

# For human data: identity ortholog map (human→human)
identity_ortho_map = {g: g for g in human_genes}

# Build via BioCypher ETL (Option A)
schema_path = PROJECT_ROOT / "config" / "schema_config.yaml"

G = build_micro_ckg(
    stabl_result=stabl_result,
    adata=adata,
    schema_path=schema_path,
    cluster_annotation=cluster_annotation,
    min_genes=20,
    ortho_map=identity_ortho_map,
    enrich_df=enrich_df,
    disease_df=disease_df,
    drug_df=drug_df,
    ppi_df=ppi_df,
)

print(f"\nHuman Micro-CKG (BioCypher ETL): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
from collections import Counter
node_type_counts = Counter(d.get("label", "unknown") for _, d in G.nodes(data=True))
for ntype, cnt in sorted(node_type_counts.items(), key=lambda x: -x[1]):
    print(f"  {ntype}: {cnt}")

graph_path = save_graph(G, CACHE_DIR / "nanostring_micro_ckg.graphml")
print(f"Graph saved: {graph_path}")

# %% [markdown]
# ## B3.4 BioCypher Agent Graph (Option B)
#
# Build the same graph using the **Agent Graph API** for type-aware
# queries and JSON serialisation — ideal for LLM prompt injection.

# %%
agent_graph = build_micro_ckg_agent(
    stabl_result=stabl_result,
    adata=adata,
    schema_path=schema_path,
    cluster_annotation=cluster_annotation,
    min_genes=20,
    ortho_map=identity_ortho_map,
    enrich_df=enrich_df,
    disease_df=disease_df,
    drug_df=drug_df,
    ppi_df=ppi_df,
)

# Demonstrate type-aware queries
all_nodes = agent_graph.get_nodes()
gene_nodes = [n for n in all_nodes if n.type == "gene"]
pw_nodes = [n for n in all_nodes if n.type == "biological_process"]
disease_nodes = [n for n in all_nodes if n.type == "disease"]
drug_nodes = [n for n in all_nodes if n.type == "drug"]
print(f"\nAgent Graph: {len(gene_nodes)} genes, {len(pw_nodes)} pathways, "
      f"{len(disease_nodes)} diseases, {len(drug_nodes)} drugs")

if disease_nodes:
    print("\nTop disease associations:")
    for node in disease_nodes[:8]:
        print(f"  {node.properties.get('name', node.id)}")

if drug_nodes:
    print("\nDrug nodes:")
    for node in drug_nodes[:8]:
        print(f"  {node.properties.get('name', node.id)}")

# Find paths between a gene and a pathway
if gene_nodes and pw_nodes:
    src = gene_nodes[0].id
    tgt = pw_nodes[0].id
    paths = agent_graph.find_paths(src, tgt, max_length=3)
    if paths:
        print(f"\nPath ({src} → {tgt}):")
        for edge in paths[0]:
            print(f"  {edge}")

json_str = agent_graph.to_json()
print(f"\nJSON context: {len(json_str):,} chars")

# %% [markdown]
# ## B3.5 Micro-CKG Visualization
#
# Network visualization of the human-specific Micro-CKG. Gene nodes are
# coloured blue, pathway nodes purple, with edge thickness proportional
# to combined enrichment score.

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

_COLORS = {
    "gene": "#4A90D9",
    "biological_process": "#9B59B6",
}

node_colors = []
node_sizes = []
for n in G.nodes():
    lbl = G.nodes[n].get("label", "")
    node_colors.append(_COLORS.get(lbl, "#999"))
    node_sizes.append(500 if lbl == "gene" else 400)

# Shorten labels for display
node_labels = {}
for n in G.nodes():
    raw = str(n).split(":", 1)[-1]
    if len(raw) > 30:
        raw = raw[:28] + ".."
    node_labels[n] = raw

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("white")

if G.number_of_nodes() > 0:
    pos = nx.spring_layout(G, k=2.0, seed=42)

    # Separate solid and dashed edges
    ppi_edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get("label") == "gene_interacts_with_gene"]
    other_edges = [(u, v) for u, v, d in G.edges(data=True)
                   if d.get("label") != "gene_interacts_with_gene"]

    if other_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=other_edges, ax=ax,
            alpha=0.6, arrows=True, arrowsize=10,
            edge_color="#9B59B6", width=1.5,
        )
    if ppi_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=ppi_edges, ax=ax,
            alpha=0.4, arrows=False,
            edge_color="#888", width=1.2, style="dashed",
        )

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, node_labels, ax=ax,
                            font_size=8, font_weight="bold")

legend_handles = [
    mpatches.Patch(color="#4A90D9", label="Human Gene (Stabl)"),
    mpatches.Patch(color="#9B59B6", label="Pathway (GO/KEGG/Reactome)"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.85)
ax.set_title(
    f"NanoString Human Micro-CKG · {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
    fontsize=14, fontweight="bold",
)
ax.axis("off")
plt.tight_layout()

kg_path = ASSETS_DIR / "nanostring_kg_subgraph.png"
fig.savefig(kg_path, dpi=300, bbox_inches="tight")
plt.close(fig)

from IPython.display import Image, display
display(Image(filename=str(kg_path), width=800))
print(f"KG plot saved → {kg_path}")

# %% [markdown]
# ## B3.6 LLM Agent Query
#
# Instantiate the evidence-traced QA agent with the human Micro-CKG and
# ask a pathway-focused translational question. The agent answers
# strictly from the graph context with mandatory evidence citations.

# %%
from src.llm_agent import create_qa_agent, query_graph

try:
    agent = create_qa_agent(G, provider="ollama", model="deepseek-r1:14b")
    print("QA agent initialised (Ollama — deepseek-r1:14b).")
    _OLLAMA_AVAILABLE = True
except Exception as _ollama_err:
    print(f"⚠ Ollama not available: {_ollama_err}")
    print("  LLM query cells below will be skipped.")
    _OLLAMA_AVAILABLE = False
    agent = None

# %%
if _OLLAMA_AVAILABLE:
    answer = query_graph(
        agent,
        "Based on the graph, what biological pathways are these NanoString-derived "
        "human genes involved in? For each gene-pathway link, cite the exact "
        "evidence path and enrichment p-value from the graph.",
    )
    print(answer)
else:
    print("Skipped — Ollama not available.")

# %% [markdown]
# ## B3.7 Interpretation
#
# The NanoString CosMx → Stabl → KG → LLM pipeline demonstrates
# **complete platform modularity**:
#
# - The **same** `run_stabl_cached()` function ran on NanoString
#   single-cell data (Track B) that previously ran on Visium spot data
#   (Phase 1).
# - The **same** `run_go_enrichment()` and `get_string_ppi()` functions
#   enriched human gene hits regardless of the upstream spatial platform.
# - The **same** `build_micro_ckg()` and `build_micro_ckg_agent()`
#   functions — powered by BioCypher — produced schema-validated
#   knowledge graphs queryable by the LLM agent.
#
# This proves the pipeline is data-source agnostic — any spatial
# technology producing an AnnData with expression + coordinates + binary
# condition labels can be routed through the full Stabl → KG → LLM
# pipeline without code modification.
