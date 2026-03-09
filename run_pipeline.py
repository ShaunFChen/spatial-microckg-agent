"""End-to-end execution of the Spatial-MicroCKG Mouse AD pipeline.

Orchestrates:
1. GEO GSE203424 data download (3 PSAPP×CO + 3 WT×CO, Corn-Oil only)
2. QC, normalization, unsupervised stratified downsampling, ComBat batch correction
3. Stabl stability-based feature selection (DE pre-filter → top markers)
4. Spatial / UMAP marker visualization (including Prox1 verification)
5. BioCypher Micro-CKG construction
6. Local Ollama LLM agent query with evidence traceability
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "cache"
ASSETS_DIR = PROJECT_ROOT / "assets"

for d in (DATA_DIR, CACHE_DIR, ASSETS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run the full Spatial-MicroCKG Mouse AD pipeline."""
    # ------------------------------------------------------------------
    # 1. Data Ingestion — GEO GSE203424
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Data Ingestion — GEO GSE203424 (3 PSAPP×CO + 3 WT×CO)")
    print("=" * 70)

    from src.data_ingestion import get_dataset

    h5ad_path = get_dataset(DATA_DIR, source="geo_ad")

    from src.spatial_pipeline import load_adata

    adata = load_adata(h5ad_path)
    print(f"\n  Dataset: {adata.shape[0]} spots × {adata.shape[1]} genes")
    print(f"  Samples: {adata.obs['sample_id'].nunique()}")
    print(f"  Conditions: {dict(adata.obs['condition'].value_counts())}")

    # ------------------------------------------------------------------
    # 2. QC & Normalization
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Quality Control & Normalization")
    print("=" * 70)

    from src.spatial_pipeline import run_qc, normalize

    adata = run_qc(adata)
    adata = normalize(adata)

    # ------------------------------------------------------------------
    # 3. Stabl Feature Selection (stratified downsample → ComBat → DE)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Stabl Feature Selection (stratified downsample → ComBat → DE pre-filter)")
    print("=" * 70)

    from src.spatial_pipeline import run_stabl_cached

    stabl_result = run_stabl_cached(
        adata,
        cache_dir=CACHE_DIR,
        dataset_name="geo_ad",
        label_method="condition",
        n_bootstraps=50,
        prefilter="hvg",
        n_hvgs=2000,
    )

    print(f"\n  Stabl results:")
    print(f"    Features selected: {stabl_result['n_selected']}")
    print(f"    FDP+ threshold:    {stabl_result['threshold']:.4f}")
    print(f"    Minimum FDP+:      {stabl_result['fdr']:.4f}")

    import pandas as pd

    df_features = pd.DataFrame({
        "Gene": stabl_result["selected_genes"],
        "Stability_Score": [
            stabl_result["stability_scores"][g]
            for g in stabl_result["selected_genes"]
        ],
    }).sort_values("Stability_Score", ascending=False).reset_index(drop=True)

    print(f"\n  Top 20 Stabl-selected genes:")
    print(df_features.head(20).to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Spatial / UMAP Marker Visualization
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Marker Visualization (Spatial or UMAP fallback)")
    print("=" * 70)

    from src.spatial_pipeline import plot_spatial_markers

    top_markers = df_features["Gene"].tolist()
    saved_plots = plot_spatial_markers(
        adata,
        markers=top_markers,
        save_dir=ASSETS_DIR,
        n_top=5,
    )
    print(f"\n  {len(saved_plots)} plots saved to {ASSETS_DIR}")

    # ------------------------------------------------------------------
    # 5. BioCypher Micro-CKG Construction
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: BioCypher Micro-CKG Construction")
    print("=" * 70)

    from src.spatial_pipeline import compute_clusters, annotate_clusters
    from src.biocypher_adapter import build_micro_ckg, save_graph

    adata = compute_clusters(adata, n_hvgs=2000)
    cluster_annotation = annotate_clusters(adata)

    schema_path = PROJECT_ROOT / "config" / "schema_config.yaml"
    graph = build_micro_ckg(
        stabl_result=stabl_result,
        adata=adata,
        schema_path=schema_path,
        cluster_annotation=cluster_annotation,
    )

    graph_path = save_graph(graph, CACHE_DIR / "micro_ckg.graphml")
    print(f"\n  Graph saved: {graph_path}")
    print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # ------------------------------------------------------------------
    # 6. Local Ollama LLM Agent — Evidence-Traced Query
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Ollama LLM Agent — Evidence-Traced Query")
    print("=" * 70)

    from src.llm_agent import create_qa_agent, query_graph

    agent = create_qa_agent(graph, provider="ollama")
    print("  Agent initialized (Ollama — deepseek-r1:14b).\n")

    test_question = (
        "What are the most significant genes associated with the "
        "Alzheimer's disease condition in this Micro-CKG?"
    )
    print(f"  Question: {test_question}\n")

    try:
        answer = query_graph(agent, test_question)
        print("  " + "-" * 60)
        print("  ANSWER:")
        print("  " + "-" * 60)
        print(answer)
    except Exception as exc:
        print(f"  Query failed: {type(exc).__name__}: {exc}")
        print("  Ensure Ollama is running locally (ollama serve) with "
              "deepseek-r1:14b pulled.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
