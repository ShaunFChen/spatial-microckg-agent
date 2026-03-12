# %% [markdown]
# # 05 — LLM Agent (Evidence-Traced Graph QA)
#
# **Pipeline Step 5 of 5**
#
# This notebook demonstrates the LLM-powered QA agent that queries the Micro-CKG with strict evidence traceability. Every answer cites exact `(Source)--[Edge_Type, Score=X.XX]-->(Target)` graph evidence.
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
_KEEP_EDGE_LABELS = {"drug_gene_association"}

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

agent = create_qa_agent(optimized_graph, provider="ollama")
print("QA agent initialised (Ollama — deepseek-r1:14b).")

# %% [markdown]
# ## 5.3 Translational Interview Queries
#
# Four questions designed to demonstrate the end-to-end value of the Micro-CKG for a Takeda Pharmaceutical interview — each exercises a different translational capability of the pipeline:
#
# 1. **Translational drug discovery** — which FDA/clinical compounds in ChEMBL target the human orthologs of Stabl-selected AD biomarkers?
# 2. **Spatial pathway analysis** — which spatially autocorrelated genes are linked to the hippocampus and what do the cell-type associations imply for AD neurodegeneration?
# 3. **Cluster-specific drug targets** — within Leiden Cluster 3, which Stabl DE genes have human orthologs with PhaseIII+ drug matches?
# 4. **PageRank hub gene druggability** — which gene is the highest-connectivity hub connecting cell types and regions, and does its human ortholog have an FDA-approved drug?

# %%
print("[Query 1] Translational drug discovery...")
print(query_graph(
    agent,
    "Which FDA-approved or clinical-stage drugs in the Micro-CKG target the human orthologs "
    "of Stabl-selected AD biomarkers? For each drug-target pair, cite the full translational "
    "evidence path: mouse gene → human ortholog → drug compound, and include mechanism of "
    "action and clinical phase.",
))

# %%
print("\n[Query 2] Spatial pathway + anatomy...")
print(query_graph(
    agent,
    "Which spatially autocorrelated Stabl genes are associated with the hippocampus in the "
    "Micro-CKG? Identify the connecting cell-type nodes and describe what their expression "
    "profiles suggest about AD-driven neurodegeneration in this anatomical region.",
))

# %%
print("\n[Query 3] Cluster-specific drug targets...")
# Note: Leiden Cluster 3 is assigned deterministically (seeded), but cluster
# labels may shift if the preprocessing or clustering parameters change.
print(query_graph(
    agent,
    "Within Leiden Cluster 3, which differentially expressed Stabl genes have human orthologs "
    "with matching ChEMBL drug entries in the Micro-CKG? Trace the evidence path from log2FC "
    "and p_adj through the ortholog mapping to the drug candidate, and flag any drugs at "
    "max_phase >= 3.",
))

# %%
print("\n[Query 4] PageRank hub gene druggability...")
print(query_graph(
    agent,
    "Based on the topology of the Micro-CKG, which gene node has the highest connectivity to "
    "both cell types and anatomical regions? Is there an FDA-approved drug in the graph that "
    "targets its human ortholog? Provide the complete evidence chain, including PageRank score "
    "if available, ortholog mapping, and drug mechanism of action.",
))
