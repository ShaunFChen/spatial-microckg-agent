# %% [markdown]
# # 04 — Knowledge Graph Construction & Drug Target Discovery
#
# **Pipeline Step 4 of 5**
#
# Constructs a **Micro-Clinical Knowledge Graph (Micro-CKG)** from the AD spatial transcriptomics data and enriches it with external biological knowledge.
#
# ### Pipeline
# 1. **Leiden clustering** of spots → proxy cell-type assignments
# 2. **Wilcoxon DE testing** across clusters → DE-filtered edges (p_adj < 0.05, |log2FC| > 0.5)
# 3. **Build BioCypher graph** with Gene, CellType, and Region nodes
# 4. **Spatial validation** via Moran's I autocorrelation
# 5. **Translational discovery** — mouse→human orthologs, GO enrichment, ChEMBL drug targets
#
# ### Inputs
# | File | Description |
# |---|---|
# | `data/processed/ad_preprocessed.h5ad` | QC-filtered, normalized AnnData from Step 01 |
# | `cache/stabl_results_<hash>.pkl` | Stabl results from Step 02 |
# | `config/schema_config.yaml` | BioCypher schema mapping |
#
# ### Outputs
# | File | Description |
# |---|---|
# | `cache/micro_ckg.graphml` | Serialized Micro-CKG in GraphML format |

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_pipeline import (
    load_adata, run_stabl_cached, compute_clusters,
    annotate_clusters, assign_condition_labels,
)
from src.biocypher_adapter import build_micro_ckg, save_graph, visualize_graph
from src.spatial_analytics import (
    compute_spatial_neighbors, compute_spatial_autocorr,
    run_nhood_enrichment,
)
from src.external_knowledge import (
    map_orthologs, run_go_enrichment, get_drug_targets,
)
from src.graph_analytics import (
    detect_communities, compute_centrality, find_bridge_genes, summarise_graph,
)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"

print("Imports ready.")

# %% [markdown]
# ## 4.1 Load Data and Stabl Results

# %%
adata = load_adata(DATA_PROCESSED / "ad_preprocessed.h5ad")

stabl_result = run_stabl_cached(
    adata,
    cache_dir=CACHE_DIR,
    dataset_name="geo_ad",
    label_method="condition",
    n_bootstraps=50,
    prefilter="de",
)

print(f"\n{stabl_result['n_selected']} Stabl-selected features loaded.")

# %% [markdown]
# ## 4.2 Compute Leiden Clusters
#
# Before building the graph, we need cell-type assignments. We apply Leiden community detection (a graph-based clustering algorithm) to the spot-level expression profiles. The procedure is: select highly variable genes, compute PCA (40 components), build a k-nearest-neighbor graph (k=10), and partition the graph using the Leiden algorithm at resolution 0.8.
#
# Each resulting cluster represents a group of spots with similar expression profiles. These clusters serve as proxy cell-type labels (e.g., neuronal subtypes, glial populations) and are used to create CellType nodes in the knowledge graph. The cluster-to-region mapping assigns anatomical labels (Cortex, Hippocampus, Thalamus, etc.) based on cluster rank order, providing spatial context for each cell-type node.

# %%
adata = compute_clusters(adata, n_hvgs=2000)
print(f"\nLeiden clusters: {adata.obs['leiden'].nunique()}")
print(adata.obs["leiden"].value_counts().sort_index())

# Annotate clusters with brain-region marker gene signatures
cluster_annotation = annotate_clusters(adata)
print("\nCluster annotations:")
for cid, region in sorted(cluster_annotation.items(), key=lambda x: int(x[0])):
    print(f"  Cluster {cid} → {region}")

# Assign condition labels based on ground-truth metadata
condition_labels = assign_condition_labels(adata)
n_ad = int(condition_labels.sum())
n_wt = len(condition_labels) - n_ad
print(f"\nCondition labels: {n_ad} AD / {n_wt} WT spots")

# %% [markdown]
# ## 4.3 Build Micro-CKG (DE-Filtered, Gene-Expanded)
#
# The knowledge graph uses **Wilcoxon rank-sum differential expression testing** to create statistically significant edges instead of simple expression thresholds.
#
# **Gene expansion:** Stabl's strict FDP+ threshold often selects very few genes (e.g., 2). To ensure the graph is informative for drug discovery, `build_micro_ckg` automatically expands the gene list to at least `min_genes` (default 20) by including the next-highest-ranked genes by stability score. The original Stabl-selected genes are marked `is_selected=True`; expanded genes are marked `is_selected=False`.
#
# **Filtering criteria:**
# - Gene → CellType edges require adjusted p-value < 0.05 AND |log2FC| > 0.5
# - Gene → Region edges are aggregated from DE-significant cluster-level associations
#
# **Edge attributes include:**
# - `log2fc` — log2 fold change from DE test
# - `pval_adj` — Benjamini-Hochberg adjusted p-value
# - `stability_score` — Stabl bootstrap stability score
# - `mean_expression` — mean expression in the cluster

# %%
schema_path = PROJECT_ROOT / "config" / "schema_config.yaml"

graph = build_micro_ckg(
    stabl_result=stabl_result,
    adata=adata,
    schema_path=schema_path,
    cluster_annotation=cluster_annotation,
    min_genes=20,
)

print(f"\nMicro-CKG:")
print(f"  Nodes: {graph.number_of_nodes()}")
print(f"  Edges: {graph.number_of_edges()}")

# Quick topology summary
summary = summarise_graph(graph)
print(f"  Density: {summary['density']:.4f}")
print(f"  Components: {summary['n_components']}")

# %% [markdown]
# ## 4.4 Graph Analytics & Drug Discovery Dashboard
#
# Multi-panel visualization showing:
# - **Top hub genes** — the most connected biomarkers and their cell-type/region associations
# - **Gene × Region heatmap** — which biomarkers are spatially specific to which brain regions
# - **Centrality ranking** — PageRank identifies the most influential genes in the network
# - **Edge composition** — breakdown of relationship types in the knowledge graph

# %%
# Community detection & centrality
community_map = detect_communities(graph)
centrality_df = compute_centrality(graph)
bridge_df = find_bridge_genes(graph, centrality_df)

print("Top 10 hub genes by PageRank:")
gene_centrality = centrality_df[centrality_df["label"] == "gene"].head(10)
print(gene_centrality[["degree", "betweenness", "pagerank"]].to_string())

print("\nBridge genes (connect multiple graph communities):")
print(bridge_df.head(10)[["gene", "bridge_score", "n_communities_bridged", "betweenness"]].to_string(index=False))

# Multi-panel drug discovery dashboard
visualize_graph(graph, community_map=community_map, centrality_df=centrality_df)

# %% [markdown]
# ## 4.5 Save Graph
#
# The Micro-CKG is serialized to GraphML format, a standard XML-based graph format supported by NetworkX, Cytoscape, Neo4j, and other graph analysis tools. This file serves as the input to the LLM agent in Step 05 and can also be loaded into graph visualization software for interactive exploration.

# %%
graph_path = save_graph(graph, CACHE_DIR / "micro_ckg.graphml")
print(f"\nGraph persisted: {graph_path}")
print(f"File size: {graph_path.stat().st_size / 1e3:.1f} KB")

# %% [markdown]
# ## 4.6 Spatial Validation (Moran's I)
#
# **Why this matters for drug development:** A biomarker is only therapeutically relevant if its spatial expression pattern is non-random. Moran's I > 0 with p < 0.05 confirms that a gene shows **spatially clustered expression** — it marks a real tissue compartment, not noise. This spatial specificity is critical for targeted drug delivery.

# %%
# Build spatial neighbourhood graph
adata = compute_spatial_neighbors(adata, n_neighs=6)

# Spatial autocorrelation — test Stabl-selected genes
moran_df = compute_spatial_autocorr(
    adata,
    genes=stabl_result["selected_genes"],
    mode="moran",
    n_perms=100,
)

# Show top spatially autocorrelated genes
sig_moran = moran_df[moran_df["pval_norm"] < 0.05].sort_values("I", ascending=False)
print(f"\nTop spatially autocorrelated Stabl genes (Moran's I):")
print(f"({len(sig_moran)}/{len(stabl_result['selected_genes'])} genes spatially significant)\n")
print(sig_moran.head(15)[["I", "pval_norm"]].to_string())

# %% [markdown]
# ## 4.7 Translational Drug Target Discovery
#
# This is the key drug-development step. We take the spatially-validated mouse biomarker genes and:
# 1. **Map to human orthologs** — mouse gene symbols → human equivalents via HomoloGene
# 2. **Pathway enrichment** — what GO biological processes / KEGG pathways do these genes converge on?
# 3. **Drug target query** — which human orthologs have approved or clinical-stage drugs in ChEMBL?
#
# This creates a direct pipeline from spatial transcriptomics → druggable targets.

# %%
# Mouse → Human ortholog mapping
ortho_df = map_orthologs(stabl_result["selected_genes"])
human_genes = ortho_df["human_symbol"].dropna().tolist()
print(f"Mapped {len(human_genes)} / {len(stabl_result['selected_genes'])} genes to human orthologs")
display(ortho_df.head(10))

# GO / KEGG pathway enrichment
enrich_df = run_go_enrichment(human_genes)
if enrich_df is not None and not enrich_df.empty:
    print(f"\nTop enriched pathways ({len(enrich_df)} total):")
    display(enrich_df.head(15))
else:
    print("\nNo significant enrichment results returned.")

# ChEMBL drug target lookup
drug_df = get_drug_targets(human_genes)
if drug_df is not None and not drug_df.empty:
    print(f"\nDruggable targets found: {drug_df['gene'].nunique()} genes, {len(drug_df)} drug associations")
    display(drug_df.head(15))
else:
    print("\nNo drug targets found in ChEMBL for these genes.")

# %% [markdown]
# ## 4.8 Enrich Micro-CKG with Drug, Ortholog, and Human Gene Nodes
#
# Drug nodes sourced from ChEMBL (Section 4.7) are integrated into the Micro-CKG as first-class nodes, forming an explicit **3-hop translational evidence path**:
#
# ```
# [Drug] --drug_gene_association--> [HumanGene] --is_ortholog_of--> [MouseGene]
# ```
#
# **Added elements:**
# - **`human_gene` nodes** — one node per unique human ortholog symbol; serve as the explicit intermediate between drug compound and mouse biomarker
# - **`is_ortholog_of` edges** — directed `human_gene → gene` edges linking each human ortholog to its mouse counterpart (sourced from HomoloGene via Section 4.7)
# - **Drug nodes** — one node per unique compound; attributes: `name`, `mechanism_of_action`, `max_phase`
# - **`drug_gene_association` edges** — drug → human gene node (drugs are approved/tested against human targets, not the mouse symbol directly)
# - **`human_ortholog` attribute** — back-populated on existing mouse gene nodes for backward compatibility with the LLM agent
#
# The enriched graph is re-saved to replace the base CKG.

# %%
from src.biocypher_adapter import (
    generate_drug_nodes,
    generate_drug_gene_edges,
    generate_human_ortholog_nodes,
    generate_human_ortholog_edges,
    save_graph,
)

# Build mouse → human ortholog lookup
ortho_map: dict[str, str] = {}
if ortho_df is not None and not ortho_df.empty:
    for _, row in ortho_df.iterrows():
        mouse_sym = row.get("mouse_symbol")
        human_sym = row.get("human_symbol")
        if mouse_sym and human_sym:
            ortho_map[str(mouse_sym)] = str(human_sym)

# Back-populate human_ortholog attribute on existing gene nodes (LLM compat)
for node_id, data in graph.nodes(data=True):
    if data.get("label") == "gene":
        sym = data.get("symbol", "")
        human = ortho_map.get(sym)
        if human:
            graph.nodes[node_id]["human_ortholog"] = human

# 1. Add HumanGene intermediate nodes
ortholog_nodes_added = 0
for node_id, node_label, props in generate_human_ortholog_nodes(ortho_map):
    graph.add_node(node_id, label=node_label, **props)
    ortholog_nodes_added += 1

# 2. Add is_ortholog_of edges (human_gene → mouse gene)
ortholog_edges_added = 0
for edge_id, src, tgt, label, props in generate_human_ortholog_edges(ortho_map):
    if tgt in graph:
        graph.add_edge(src, tgt, key=edge_id, label=label, **props)
        ortholog_edges_added += 1

print(f"HumanGene nodes added : {ortholog_nodes_added}")
print(f"is_ortholog_of edges  : {ortholog_edges_added}")

# 3. Add drug compound nodes
drug_nodes_added = 0
if drug_df is not None and not drug_df.empty:
    for node_id, node_label, props in generate_drug_nodes(drug_df):
        graph.add_node(node_id, label=node_label, **props)
        drug_nodes_added += 1

    # 4. Add drug → human_gene edges (human_gene nodes now exist in graph)
    drug_edges_added = 0
    for edge_id, src, tgt, label, props in generate_drug_gene_edges(drug_df, ortho_map):
        if tgt in graph:
            graph.add_edge(src, tgt, key=edge_id, label=label, **props)
            drug_edges_added += 1

    print(f"Drug nodes added      : {drug_nodes_added}")
    print(f"Drug→HumanGene edges  : {drug_edges_added}")
else:
    print("No drug_df available — skipping drug node enrichment.")

total_nodes = graph.number_of_nodes()
total_edges = graph.number_of_edges()
print(f"\nEnriched Micro-CKG: {total_nodes} nodes, {total_edges} edges")

# Re-save the enriched graph
graph_path = save_graph(graph, CACHE_DIR / "micro_ckg.graphml")
print(f"Enriched graph saved: {graph_path}")

# %% [markdown]
# ## 4.9 Drug Target Sub-Graph (3-Hop Translational Path)
#
# Focused sub-graph visualising the explicit **Drug → Human Ortholog → Mouse Gene** 3-hop translational evidence path.
#
# | Node color | Node type | Role |
# |---|---|---|
# | **Red** | Drug (ChEMBL) | FDA-approved or clinical-stage compound |
# | **Orange** | Human gene ortholog | Human protein target of the drug |
# | **Blue** | Mouse gene (Stabl) | Spatially validated AD biomarker |
#
# Edge thickness encodes `max_phase` for drug→human edges (thicker = more advanced clinical stage). The subgraph is preferentially centered on the **Fth1** and **Prnp** neighborhoods; if no direct drug paths exist for these genes, all available 3-hop paths are shown.

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# Node sets by type
drug_nodes = {n for n, d in graph.nodes(data=True) if d.get("label") == "drug"}
human_gene_nodes = {n for n, d in graph.nodes(data=True) if d.get("label") == "human_gene"}
mouse_gene_nodes = {n for n, d in graph.nodes(data=True) if d.get("label") == "gene"}

if not drug_nodes:
    print("No drug nodes in graph — drug enrichment may have been skipped (no ChEMBL hits).")
else:
    # Preferentially focus on Fth1 / Prnp neighborhoods
    focus_mouse = {"gene:Fth1", "gene:Prnp"}
    focus_human = {
        n for n in human_gene_nodes
        if any(
            graph.has_edge(n, mg) for mg in focus_mouse
        )
    }
    focus_drugs = {
        u for u, v, d in graph.edges(data=True)
        if d.get("label") == "drug_gene_association" and v in focus_human
    }

    # Fall back to all paths if focus neighborhood is empty
    if not focus_drugs:
        print("No drug paths found for Fth1/Prnp — showing all 3-hop paths.")
        focus_human = set(human_gene_nodes)
        focus_drugs = set(drug_nodes)
        focus_mouse = {
            v for u, v, d in graph.edges(data=True)
            if d.get("label") == "is_ortholog_of" and u in focus_human
        }

    focused_nodes = focus_drugs | focus_human | focus_mouse
    sub = graph.subgraph(focused_nodes).copy()

    # Assign colors and sizes
    node_colors = []
    node_sizes = []
    for n in sub.nodes():
        lbl = sub.nodes[n].get("label", "")
        if lbl == "drug":
            node_colors.append("#E74C3C")   # red
            node_sizes.append(700)
        elif lbl == "human_gene":
            node_colors.append("#F39C12")   # orange
            node_sizes.append(500)
        else:
            node_colors.append("#4A90D9")   # blue
            node_sizes.append(400)

    # Edge widths: drug→human edges scaled by max_phase; ortholog edges thin dashed
    edge_widths = []
    edge_styles = []
    for u, v, d in sub.edges(data=True):
        if d.get("label") == "drug_gene_association":
            phase = int(d.get("max_phase", 1) or 1)
            edge_widths.append(0.5 + phase * 0.6)
            edge_styles.append("solid")
        else:
            edge_widths.append(1.0)
            edge_styles.append("dashed")

    # Node labels (strip prefix, shorten long drug names)
    node_labels = {}
    for n in sub.nodes():
        raw = str(n).split(":", 1)[-1]
        if sub.nodes[n].get("label") == "drug" and len(raw) > 18:
            raw = raw[:16] + ".."
        node_labels[n] = raw

    fig, ax = plt.subplots(figsize=(13, 9))
    pos = nx.spring_layout(sub, k=1.5, seed=42)

    # Draw solid and dashed edges separately
    solid_edges = [(u, v) for u, v, d in sub.edges(data=True)
                   if d.get("label") == "drug_gene_association"]
    dashed_edges = [(u, v) for u, v, d in sub.edges(data=True)
                    if d.get("label") == "is_ortholog_of"]
    solid_widths = [0.5 + int(sub.edges[u, v].get("max_phase", 1) or 1) * 0.6
                    for u, v in solid_edges]

    nx.draw_networkx_edges(sub, pos, edgelist=solid_edges, ax=ax,
                           alpha=0.7, arrows=True, arrowsize=14,
                           edge_color="#C0392B", width=solid_widths)
    nx.draw_networkx_edges(sub, pos, edgelist=dashed_edges, ax=ax,
                           alpha=0.6, arrows=True, arrowsize=12,
                           edge_color="#888", width=1.2, style="dashed")
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_labels(sub, pos, node_labels, ax=ax,
                            font_size=8, font_weight="bold")

    legend_handles = [
        mpatches.Patch(color="#E74C3C", label="Drug (ChEMBL)"),
        mpatches.Patch(color="#F39C12", label="Human Ortholog"),
        mpatches.Patch(color="#4A90D9", label="Mouse Gene (Stabl)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.85)

    n_drugs = len(focus_drugs)
    n_human = len(focus_human)
    n_mouse = len(focus_mouse & set(sub.nodes()))
    ax.set_title(
        f"3-Hop Translational Sub-Graph  ·  {n_drugs} drugs → {n_human} human orthologs → {n_mouse} mouse genes",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()

    out_path = CACHE_DIR / "drug_subgraph.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    from IPython.display import Image, display
    display(Image(filename=str(out_path), width=750))
    print(f"Sub-graph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
    print(f"  Red (drugs): {n_drugs}  |  Orange (human): {n_human}  |  Blue (mouse): {n_mouse}")

# %% [markdown]
# ## 4.10 Presentation-Grade Subgraph (Prnp, Fth1, Calb1)
#
# The full Micro-CKG has hundreds of edges — optimal for LLM queries but visually cluttered. This cell draws a **clean, filtered subgraph** focused on three narrative-target biomarkers. Only edges with |log2FC| > 0.25 or spatial correlation > 0.5 are retained, producing a publication-quality figure.

# %%
import matplotlib.pyplot as plt
import networkx as nx

key_genes = ["gene:Prnp", "gene:Fth1", "gene:Calb1"]
presentation_nodes = set(key_genes)
presentation_edges = []

for u, v, d in graph.edges(data=True):
    if u in key_genes and (abs(d.get("log2fc", 0)) > 0.25 or d.get("spatial_correlation", 0) > 0.5):
        presentation_nodes.add(v)
        presentation_edges.append((u, v))

sub = graph.subgraph(presentation_nodes).copy()
sub.remove_edges_from(
    [e for e in sub.edges() if e not in presentation_edges]
)

fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(sub, k=0.8, seed=42)
colors = [
    "#4A90D9" if sub.nodes[n].get("label") == "gene"
    else "#27AE60" if sub.nodes[n].get("label") == "cell_type"
    else "#E67E22"
    for n in sub.nodes()
]
labels = {n: str(n).split(":")[1] if ":" in str(n) else str(n) for n in sub.nodes()}

nx.draw(sub, pos, ax=ax, with_labels=True, labels=labels, node_color=colors,
        node_size=1200, font_size=10, font_weight="bold",
        edge_color="#555", width=2.0)
ax.set_title("Targeted Biomarker Knowledge Graph (Prnp, Fth1, Calb1)",
             fontsize=16, fontweight="bold")
out_path = CACHE_DIR / "presentation_subgraph.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

from IPython.display import Image, display
display(Image(filename=str(out_path), width=700))
print(f"Subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
