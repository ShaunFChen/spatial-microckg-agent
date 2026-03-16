# %% [markdown]
# # 05 — LLM Agent (Evidence-Traced Graph QA)
#
# **Pipeline Step 5 of 5**
#
# This notebook demonstrates the LLM-powered QA agent that queries the Micro-CKG with strict evidence traceability. Every answer cites exact `(Source)--[Edge_Type, Score=X.XX]-->(Target)` graph evidence.
#
# It also showcases the **BioCypher Agent Graph API** — a pure-Python
# graph with built-in type-aware queries, path finding, and JSON
# serialisation that can be injected directly into LLM prompts.
#
# ### Hardened Guardrails
# 1. **No External Knowledge** — answers from graph context only
# 2. **Missing Data Fallback** — explicit "No evidence found" response
# 3. **Mandatory Citation** — `[Evidence: (Source) --(Edge)--> (Target)]`
# 4. **Objective Tone** — no speculation
#
# ### Prerequisites
# - Ollama running locally with `deepseek-r1:14b` pulled
#
# ### Inputs
# | File | Description |
# |---|---|
# | `cache/micro_ckg.graphml` | Serialized Micro-CKG from Step 04 |

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.biocypher_adapter import load_graph
from src.llm_agent import create_qa_agent, query_graph

CACHE_DIR = PROJECT_ROOT / "cache"

print("Imports ready.")

# %% [markdown]
# ## 5.1 Load Micro-CKG
#
# We load the persisted GraphML file produced by notebook 04. The graph is deserialized back into a NetworkX DiGraph with all node and edge attributes intact. The summary below confirms the graph structure matches expectations (number of gene nodes, cell-type nodes, region nodes, and total edges).

# %%
graph = load_graph(CACHE_DIR / "micro_ckg.graphml")
print(f"\nMicro-CKG: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# %% [markdown]
# ## 5.2 Create QA Agent (Ollama — deepseek-r1:14b)
#
# The QA agent uses **deepseek-r1:14b** running locally via Ollama with a pruned graph context. Weak edges (|log2FC| < 0.25 and spatial correlation < 0.4) are removed to reduce prompt token size. No API key needed — runs entirely on-device.

# %%
import time

# Aggressively prune the graph to reduce LLM prompt token size,
# but always preserve drug translational edges (they have no log2fc/spatial attrs)
_KEEP_EDGE_LABELS = {"drug_gene_association", "gene_associated_with_disease"}

optimized_graph = graph.copy()
edges_to_remove = [
    (u, v) for u, v, d in optimized_graph.edges(data=True)
    if d.get("label") not in _KEEP_EDGE_LABELS
    and abs(d.get("log2fc", 0)) < 0.25
    and d.get("spatial_correlation", 0) < 0.4
]
optimized_graph.remove_edges_from(edges_to_remove)

print(f"Original edges: {graph.number_of_edges()}")
print(f"Pruned edges for LLM: {optimized_graph.number_of_edges()}")

try:
    agent = create_qa_agent(optimized_graph, provider="ollama", model="deepseek-r1:14b")
    print("QA agent initialised (Ollama — deepseek-r1:14b).")
    _OLLAMA_AVAILABLE = True
except Exception as _ollama_err:
    print(f"⚠ Ollama not available: {_ollama_err}")
    print("  LLM query cells below will be skipped.")
    _OLLAMA_AVAILABLE = False
    agent = None

# %% [markdown]
# ## 5.3 AD-Relevant Translational Queries
#
# Four questions that probe the disease mechanisms surfaced by
# Stabl feature selection (Notebook 02) and encoded in the Micro-CKG
# (Notebook 04).  Each targets a different data-supported AD theme:
#
# 1. **Aβ–PrP axis** — drugs and diseases linked to the amyloid-β receptor Prnp
# 2. **Monoaminergic neurodegeneration** — Th as the highest-degree hub gene; drug landscape
# 3. **Direct AD-associated signalling** — Cdc42ep1’s Alzheimer’s disease link and neighbouring pathways
# 4. **Cross-theme integration** — which single gene bridges the most diseases, drugs, and regions?

# %%
if _OLLAMA_AVAILABLE:
    print("[Query 1] A\u03b2\u2013PrP axis: drugs targeting amyloid-\u03b2 synaptic toxicity...")
    print(query_graph(
        agent,
        "Prnp (PrP^C) is the primary neuronal receptor for amyloid-\u03b2 oligomers in the "
        "PSAPP Alzheimer's model. Using only edges present in the Micro-CKG, trace the "
        "complete evidence path from the mouse gene Prnp to its human ortholog PRNP, then "
        "to any connected drug nodes (ChEMBL). For each drug, report the mechanism of action, "
        "clinical phase, and the disease associations linked to PRNP. Cite every edge.",
    ))
else:
    print("Skipped — Ollama not available.")

# %%
if _OLLAMA_AVAILABLE:
    print("\n[Query 2] Monoaminergic neurodegeneration (Th)...")
    print(query_graph(
        agent,
        "Th (Tyrosine Hydroxylase) is the highest-degree gene in the Micro-CKG with "
        "40 edges, including 15 drug associations. TH loss in the locus coeruleus is "
        "one of the earliest neuropathological events in Alzheimer's disease. Using "
        "only edges present in the graph: (a) list all drug nodes connected to gene:Th "
        "with their mechanism of action and clinical phase, (b) list all disease "
        "associations, and (c) identify which brain regions and cell types Th is "
        "differentially expressed in. Cite every edge with scores.",
    ))
else:
    print("Skipped — Ollama not available.")

# %%
if _OLLAMA_AVAILABLE:
    print("\n[Query 3] Direct AD-associated gene (Cdc42ep1)...")
    print(query_graph(
        agent,
        "Cdc42ep1 carries a direct Alzheimer's disease association in OpenTargets, "
        "alongside Parkinson's disease, multiple sclerosis, and other neurodegenerative "
        "conditions. Using the Micro-CKG, identify: (a) all disease nodes connected to "
        "gene:Cdc42ep1 and their association scores, (b) which cell types and brain "
        "regions Cdc42ep1 is differentially expressed in, and (c) any pathway or drug "
        "nodes in its neighbourhood. Explain how this gene's graph context supports "
        "its role as a neurodegeneration hub. Cite each edge.",
    ))
else:
    print("Skipped — Ollama not available.")

# %%
if _OLLAMA_AVAILABLE:
    print("\n[Query 4] Cross-theme hub gene integration...")
    print(query_graph(
        agent,
        "Considering the three AD-relevant themes in this Micro-CKG (A\u03b2 synaptic "
        "toxicity via Prnp/Ngfr, monoaminergic neurodegeneration via Th, and direct "
        "AD-associated signalling via Cdc42ep1), which single gene node has the highest "
        "PageRank centrality and connects to the most disease, drug, and anatomical-region "
        "nodes? Explain why this gene is the strongest candidate for a multi-target "
        "therapeutic strategy in Alzheimer's disease. Cite all supporting edges.",
    ))
else:
    print("Skipped — Ollama not available.")

# %% [markdown]
# ## 5.4 BioCypher Agent Graph — Structured Exploration
#
# The **Agent Graph** (`biocypher.Graph`) provides type-aware indexing,
# path finding, subgraph extraction, and lightweight JSON serialisation
# without the overhead of a full DBMS. This section loads the Micro-CKG
# via the Agent Graph API and demonstrates its built-in query methods.

# %%
from src.biocypher_adapter import load_graph as _unused, build_micro_ckg_agent
from src.spatial_pipeline import load_adata, run_stabl_cached, compute_clusters, annotate_clusters

# Rebuild Agent Graph from the same data used in notebook 04
_adata = load_adata(PROJECT_ROOT / "data" / "processed" / "ad_preprocessed.h5ad")
_stabl = run_stabl_cached(
    _adata, cache_dir=CACHE_DIR, dataset_name="geo_ad",
    label_method="condition", n_bootstraps=50, prefilter="de",
)
_adata = compute_clusters(_adata, n_hvgs=2000)
_cluster_ann = annotate_clusters(_adata)

# Fetch same enrichment data
from src.external_knowledge import map_orthologs, run_go_enrichment, get_disease_associations, get_string_ppi, get_drug_targets

_ortho_df = map_orthologs(_stabl["selected_genes"])
_human_genes = _ortho_df["human_symbol"].dropna().tolist()
_ortho_map = {
    str(r["mouse_symbol"]): str(r["human_symbol"])
    for _, r in _ortho_df.iterrows()
    if r.get("mouse_symbol") and r.get("human_symbol")
}
_ensembl_map = {
    str(r["human_symbol"]): str(r["ensembl_gene"])
    for _, r in _ortho_df.iterrows()
    if r.get("human_symbol") and r.get("ensembl_gene")
}
_enrich_df = run_go_enrichment(_human_genes)
_disease_df = get_disease_associations(_human_genes, ensembl_map=_ensembl_map)
_ppi_df = get_string_ppi(_human_genes)
_drug_df = get_drug_targets(_human_genes)

agent_graph = build_micro_ckg_agent(
    stabl_result=_stabl, adata=_adata,
    cluster_annotation=_cluster_ann, min_genes=20,
    ortho_map=_ortho_map, enrich_df=_enrich_df,
    disease_df=_disease_df, ppi_df=_ppi_df,
    drug_df=_drug_df,
)

# %% [markdown]
# ### Type-Aware Queries & Path Finding
#
# The Agent Graph lets you query by node type and find paths between
# any two nodes — useful for building focused LLM prompts.

# %%
# Query by type
all_nodes = agent_graph.get_nodes()
gene_nodes = [n for n in all_nodes if n.type == "gene"]
pathway_nodes = [n for n in all_nodes if n.type == "biological_process"]
disease_nodes = [n for n in all_nodes if n.type == "disease"]
print(f"Agent Graph contents:")
print(f"  Genes: {len(gene_nodes)}")
print(f"  Pathways: {len(pathway_nodes)}")
print(f"  Diseases: {len(disease_nodes)}")

# Find paths
if gene_nodes and pathway_nodes:
    src = gene_nodes[0].id
    tgt = pathway_nodes[0].id
    paths = agent_graph.find_paths(src, tgt, max_length=3)
    print(f"\nPaths from {src} → {tgt}:")
    if paths:
        for edge in paths[0]:
            print(f"  {edge}")
    else:
        print("  No path found (max_length=3)")

# JSON context for LLM injection
json_ctx = agent_graph.to_json()
print(f"\nJSON context size: {len(json_ctx):,} chars (ready for LLM prompt injection)")
