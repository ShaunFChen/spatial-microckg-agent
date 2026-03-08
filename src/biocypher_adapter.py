"""Map Stabl-selected features and spatial clusters to a BioCypher Micro-CKG.

Generates Gene, CellType, and AnatomicalEntity nodes with quantifiable
edges driven by Stabl stability scores, differential expression testing,
and spatial expression data.  Uses BioCypher with the Biolink model
ontology to produce an in-memory NetworkX directed graph.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

__all__ = [
    "generate_gene_nodes",
    "generate_cell_type_nodes",
    "generate_anatomical_nodes",
    "generate_gene_cell_type_edges",
    "generate_gene_region_edges",
    "generate_cell_type_region_edges",
    "build_micro_ckg",
    "save_graph",
    "load_graph",
    "visualize_graph",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DE_PVAL_THRESHOLD = 0.05
_DE_LOG2FC_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Node Generators
# ---------------------------------------------------------------------------


def generate_gene_nodes(
    stabl_result: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield BioCypher node tuples for Stabl-selected genes.

    Args:
        stabl_result: Dictionary returned by ``run_stabl_selection``.

    Yields:
        Tuples of ``(node_id, node_label, properties)`` for each selected gene.
    """
    for gene in stabl_result["selected_genes"]:
        score = stabl_result["stability_scores"][gene]
        yield (
            f"gene:{gene}",
            "gene",
            {
                "symbol": gene,
                "stability_score": round(score, 4),
                "is_selected": True,
            },
        )


def generate_cell_type_nodes(
    adata: ad.AnnData,
    cluster_annotation: dict[str, str] | None = None,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield BioCypher node tuples for spatial clusters (cell types).

    Args:
        adata: AnnData with a ``leiden`` column in ``.obs``.
        cluster_annotation: Dict mapping cluster id to region label,
            from :func:`~src.spatial_pipeline.annotate_clusters`.

    Yields:
        Tuples of ``(node_id, node_label, properties)``.
    """
    if "leiden" not in adata.obs.columns:
        return
    if cluster_annotation is None:
        cluster_annotation = {}

    cluster_counts = adata.obs["leiden"].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        region = cluster_annotation.get(str(cluster_id), "Unassigned")
        name = f"Cluster_{cluster_id}_{region}"
        yield (
            f"celltype:{name}",
            "cell_type",
            {
                "name": name,
                "cluster_id": int(cluster_id),
                "cell_count": int(count),
            },
        )


def generate_anatomical_nodes(
    adata: ad.AnnData,
    cluster_annotation: dict[str, str] | None = None,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield BioCypher node tuples for anatomical brain regions.

    Args:
        adata: AnnData with a ``leiden`` column in ``.obs``.
        cluster_annotation: Dict mapping cluster id to region label.

    Yields:
        Tuples of ``(node_id, node_label, properties)``.
    """
    if "leiden" not in adata.obs.columns:
        return
    if cluster_annotation is None:
        cluster_annotation = {}

    used_regions: set[str] = set()
    for region in cluster_annotation.values():
        if region not in used_regions:
            used_regions.add(region)
            yield (
                f"region:{region}",
                "anatomical_entity",
                {
                    "name": region,
                    "region_type": "brain_region",
                },
            )


# ---------------------------------------------------------------------------
# Edge Generators
# ---------------------------------------------------------------------------


def _run_de_analysis(
    adata: ad.AnnData,
    genes: list[str],
) -> pd.DataFrame:
    """Run Wilcoxon rank-sum DE test for *genes* across Leiden clusters.

    Args:
        adata: AnnData with ``leiden`` clusters and raw expression.
        genes: Gene symbols to test.

    Returns:
        DataFrame with columns ``gene``, ``cluster``, ``log2fc``,
        ``pval_adj``, ``mean_expr``.
    """
    src = adata.raw.to_adata() if adata.raw is not None else adata
    available = [g for g in genes if g in src.var_names]
    if not available:
        return pd.DataFrame(columns=["gene", "cluster", "log2fc", "pval_adj", "mean_expr"])

    # Subset to selected genes and run DE
    sub = src[:, available].copy()
    sub.obs["leiden"] = adata.obs["leiden"].values
    sc.tl.rank_genes_groups(sub, groupby="leiden", method="wilcoxon", use_raw=False)

    rows: list[dict[str, Any]] = []
    result = sub.uns["rank_genes_groups"]
    clusters = list(result["names"].dtype.names)
    for cid in clusters:
        names = result["names"][cid]
        logfcs = result["logfoldchanges"][cid]
        pvals = result["pvals_adj"][cid]
        for name, lfc, pval in zip(names, logfcs, pvals):
            if name in available:
                mask = adata.obs["leiden"] == cid
                expr_vals = src[mask, name].X
                if hasattr(expr_vals, "toarray"):
                    expr_vals = expr_vals.toarray()
                mean_e = float(np.mean(expr_vals))
                rows.append({
                    "gene": name,
                    "cluster": cid,
                    "log2fc": float(lfc),
                    "pval_adj": float(pval),
                    "mean_expr": mean_e,
                })
    return pd.DataFrame(rows)


def generate_gene_cell_type_edges(
    stabl_result: dict[str, Any],
    adata: ad.AnnData,
    cluster_annotation: dict[str, str] | None = None,
    de_df: pd.DataFrame | None = None,
) -> Iterator[tuple[str, str, str, str, dict[str, Any]]]:
    """Yield gene -> cell-type edges filtered by differential expression.

    Only creates edges where a gene is significantly differentially
    expressed in a cluster (adj. p < 0.05, log2FC > 0.5).

    Args:
        stabl_result: Stabl result dictionary.
        adata: AnnData with ``leiden`` clusters and raw expression.
        cluster_annotation: Dict mapping cluster id to region label.
        de_df: Pre-computed DE results from :func:`_run_de_analysis`.

    Yields:
        Tuples of ``(edge_id, source, target, label, properties)``.
    """
    if cluster_annotation is None:
        cluster_annotation = {}
    if de_df is None:
        de_df = _run_de_analysis(adata, stabl_result["selected_genes"])

    cluster_names: dict[str, str] = {}
    for cid in sorted(adata.obs["leiden"].unique(), key=int):
        region = cluster_annotation.get(str(cid), "Unassigned")
        cluster_names[str(cid)] = f"Cluster_{cid}_{region}"

    for gene in stabl_result["selected_genes"]:
        score = stabl_result["stability_scores"][gene]
        gene_de = de_df[de_df["gene"] == gene]
        for _, row in gene_de.iterrows():
            if row["pval_adj"] >= _DE_PVAL_THRESHOLD:
                continue
            if abs(row["log2fc"]) < _DE_LOG2FC_THRESHOLD:
                continue
            cid = str(row["cluster"])
            ct_name = cluster_names.get(cid, f"Cluster_{cid}")
            yield (
                f"edge:gene_ct:{gene}_{ct_name}",
                f"gene:{gene}",
                f"celltype:{ct_name}",
                "gene_cell_type_association",
                {
                    "stability_score": round(score, 4),
                    "mean_expression": round(row["mean_expr"], 4),
                    "log2fc": round(row["log2fc"], 4),
                    "pval_adj": round(row["pval_adj"], 6),
                },
            )


def generate_gene_region_edges(
    stabl_result: dict[str, Any],
    adata: ad.AnnData,
    cluster_annotation: dict[str, str] | None = None,
    de_df: pd.DataFrame | None = None,
) -> Iterator[tuple[str, str, str, str, dict[str, Any]]]:
    """Yield gene -> anatomical-region edges aggregated from DE results.

    A gene links to a region if it is significantly DE in at least one
    cluster annotated to that region.  The ``spatial_correlation``
    property is the maximum mean expression across the region's clusters.

    Args:
        stabl_result: Stabl result dictionary.
        adata: AnnData with ``leiden`` clusters.
        cluster_annotation: Dict mapping cluster id to region label.
        de_df: Pre-computed DE results.

    Yields:
        Tuples of ``(edge_id, source, target, label, properties)``.
    """
    if cluster_annotation is None:
        cluster_annotation = {}
    if de_df is None:
        de_df = _run_de_analysis(adata, stabl_result["selected_genes"])

    for gene in stabl_result["selected_genes"]:
        gene_de = de_df[
            (de_df["gene"] == gene)
            & (de_df["pval_adj"] < _DE_PVAL_THRESHOLD)
            & (de_df["log2fc"].abs() >= _DE_LOG2FC_THRESHOLD)
        ]
        region_vals: dict[str, float] = {}
        for _, row in gene_de.iterrows():
            region = cluster_annotation.get(str(row["cluster"]), "Unassigned")
            region_vals[region] = max(region_vals.get(region, 0), row["mean_expr"])

        for region, max_expr in region_vals.items():
            yield (
                f"edge:gene_region:{gene}_{region}",
                f"gene:{gene}",
                f"region:{region}",
                "gene_anatomical_entity_association",
                {
                    "spatial_correlation": round(max_expr, 4),
                },
            )


def generate_cell_type_region_edges(
    adata: ad.AnnData,
    cluster_annotation: dict[str, str] | None = None,
) -> Iterator[tuple[str, str, str, str, dict[str, Any]]]:
    """Yield cell-type -> anatomical-region association edges.

    Args:
        adata: AnnData with ``leiden`` clusters.
        cluster_annotation: Dict mapping cluster id to region label.

    Yields:
        Tuples of ``(edge_id, source, target, label, properties)``.
    """
    if cluster_annotation is None:
        cluster_annotation = {}
    cluster_counts = adata.obs["leiden"].value_counts().sort_index()
    total_spots = adata.n_obs

    for cid, count in cluster_counts.items():
        region = cluster_annotation.get(str(cid), "Unassigned")
        ct_name = f"Cluster_{cid}_{region}"
        enrichment = count / total_spots
        yield (
            f"edge:ct_region:{ct_name}_{region}",
            f"celltype:{ct_name}",
            f"region:{region}",
            "cell_type_anatomical_entity_association",
            {
                "enrichment_score": round(float(enrichment), 4),
            },
        )


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------


def build_micro_ckg(
    stabl_result: dict[str, Any],
    adata: ad.AnnData,
    schema_path: Path | str | None = None,
    cluster_annotation: dict[str, str] | None = None,
) -> nx.DiGraph:
    """Construct a BioCypher-backed Micro-CKG as a NetworkX DiGraph.

    Performs Wilcoxon DE testing across Leiden clusters and uses the
    results to create statistically-filtered edges (adj. p < 0.05,
    |log2FC| > 0.5).

    Args:
        stabl_result: Dictionary from :func:`run_stabl_cached`.
        adata: AnnData with ``leiden`` clusters and raw expression.
        schema_path: Path to the BioCypher ``schema_config.yaml``.
        cluster_annotation: Dict mapping cluster id to region label
            (from :func:`annotate_clusters`).

    Returns:
        A NetworkX directed graph with Gene, CellType, and
        AnatomicalEntity nodes plus DE-filtered association edges.
    """
    print("  Running DE testing (Wilcoxon rank-sum)...")
    de_df = _run_de_analysis(adata, stabl_result["selected_genes"])
    sig = de_df[
        (de_df["pval_adj"] < _DE_PVAL_THRESHOLD)
        & (de_df["log2fc"].abs() >= _DE_LOG2FC_THRESHOLD)
    ]
    print(f"  DE results: {len(de_df)} tests, {len(sig)} significant")

    print("  Building Micro-CKG...")
    G = nx.DiGraph()

    # Add nodes
    for node_id, node_label, props in generate_gene_nodes(stabl_result):
        G.add_node(node_id, label=node_label, **props)

    for node_id, node_label, props in generate_cell_type_nodes(adata, cluster_annotation):
        G.add_node(node_id, label=node_label, **props)

    for node_id, node_label, props in generate_anatomical_nodes(adata, cluster_annotation):
        G.add_node(node_id, label=node_label, **props)

    # Add DE-filtered edges
    for edge_id, src, tgt, label, props in generate_gene_cell_type_edges(
        stabl_result, adata, cluster_annotation, de_df
    ):
        G.add_edge(src, tgt, key=edge_id, label=label, **props)

    for edge_id, src, tgt, label, props in generate_gene_region_edges(
        stabl_result, adata, cluster_annotation, de_df
    ):
        G.add_edge(src, tgt, key=edge_id, label=label, **props)

    for edge_id, src, tgt, label, props in generate_cell_type_region_edges(
        adata, cluster_annotation
    ):
        G.add_edge(src, tgt, key=edge_id, label=label, **props)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    gene_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "gene")
    ct_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "cell_type")
    region_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "anatomical_entity")

    print(f"  Micro-CKG: {n_nodes} nodes ({gene_nodes} genes, {ct_nodes} cell types, {region_nodes} regions)")
    print(f"  Micro-CKG: {n_edges} edges (DE-filtered)")
    return G


def save_graph(graph: nx.DiGraph, path: Path | str) -> Path:
    """Persist a Micro-CKG to disk as a GraphML file.

    Args:
        graph: The Micro-CKG as a NetworkX DiGraph.
        path: Output file path (should end in ``.graphml``).

    Returns:
        The resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, str(path))
    print(f"  Graph saved to {path}")
    return path


def load_graph(path: Path | str) -> nx.DiGraph:
    """Load a Micro-CKG from a GraphML file.

    Args:
        path: Path to the ``.graphml`` file.

    Returns:
        The loaded NetworkX DiGraph.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    graph = nx.read_graphml(str(path))
    print(f"  Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_NODE_COLORS: dict[str, str] = {
    "gene": "#4A90D9",
    "cell_type": "#27AE60",
    "anatomical_entity": "#E67E22",
    "biological_process": "#9B59B6",
    "drug": "#E74C3C",
    "disease": "#F39C12",
    "protein": "#1ABC9C",
}

_NODE_SIZES: dict[str, int] = {
    "gene": 120,
    "cell_type": 300,
    "anatomical_entity": 400,
    "biological_process": 200,
    "drug": 250,
    "disease": 250,
    "protein": 150,
}


def visualize_graph(
    graph: nx.DiGraph,
    figsize: tuple[int, int] = (20, 16),
    seed: int = 42,
    *,
    community_map: dict[str, int] | None = None,
    centrality_df: Any | None = None,
    top_n_genes: int = 15,
) -> None:
    """Render a multi-panel drug-development-focused KG dashboard.

    Four panels:
    1. **Hub-gene subgraph** — only the top-N genes by degree, with
       their immediate cell-type and region neighbours.
    2. **Gene–Region heatmap** — spatial correlation matrix.
    3. **Centrality ranking** — bar chart of top genes by PageRank.
    4. **Edge-type composition** — stacked bar of edge labels.

    Args:
        graph: The Micro-CKG as a NetworkX DiGraph.
        figsize: Overall figure dimensions ``(width, height)``.
        seed: Random seed for layout reproducibility.
        community_map: Optional dict mapping node id to community index.
        centrality_df: Pre-computed centrality DataFrame.
        top_n_genes: Number of top hub genes to show in the subgraph.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import to_hex

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Micro-CKG Drug Discovery Dashboard",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # ── Panel 1: Hub-gene subgraph ────────────────────────────────
    ax1 = axes[0, 0]
    gene_nodes = [n for n, d in graph.nodes(data=True) if d.get("label") == "gene"]
    gene_degrees = {n: graph.degree(n) for n in gene_nodes}
    top_genes = sorted(gene_degrees, key=gene_degrees.get, reverse=True)[:top_n_genes]

    # Collect their neighbours
    sub_nodes = set(top_genes)
    for g in top_genes:
        sub_nodes |= set(graph.successors(g)) | set(graph.predecessors(g))
    subgraph = graph.subgraph(sub_nodes)

    # Colour by type
    sub_colors = []
    sub_sizes = []
    for n in subgraph.nodes():
        lbl = subgraph.nodes[n].get("label", "")
        if community_map and n in community_map and lbl == "gene":
            n_comm = max(community_map.values()) + 1
            cmap_cm = plt.cm.get_cmap("Set2", max(n_comm, 2))
            sub_colors.append(to_hex(cmap_cm(community_map[n])))
        else:
            sub_colors.append(_NODE_COLORS.get(lbl, "#999999"))
        sub_sizes.append(_NODE_SIZES.get(lbl, 100))

    sub_labels = {}
    for n in subgraph.nodes():
        parts = str(n).split(":", 1)
        raw = parts[1] if len(parts) > 1 else str(n)
        # Shorten cluster names: "Cluster_5_White_Matter" → "C5 WM"
        if raw.startswith("Cluster_"):
            segs = raw.split("_", 2)
            region_short = segs[2][:4] if len(segs) > 2 else ""
            raw = f"C{segs[1]} {region_short}"
        sub_labels[n] = raw

    # Edge widths
    sub_widths = []
    for u, v, d in subgraph.edges(data=True):
        lfc = abs(d.get("log2fc", 0))
        sub_widths.append(0.5 + min(lfc, 5) * 0.6)

    pos = nx.kamada_kawai_layout(subgraph)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.25, arrows=True, arrowsize=8,
                           edge_color="#888", width=sub_widths, ax=ax1)
    nx.draw_networkx_nodes(subgraph, pos, node_color=sub_colors,
                           node_size=sub_sizes, alpha=0.9, ax=ax1)
    nx.draw_networkx_labels(subgraph, pos, sub_labels, font_size=7,
                            font_weight="bold", ax=ax1)

    legend_handles = [
        mpatches.Patch(color=_NODE_COLORS["gene"], label="Gene"),
        mpatches.Patch(color=_NODE_COLORS["cell_type"], label="Cell Type"),
        mpatches.Patch(color=_NODE_COLORS["anatomical_entity"], label="Brain Region"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.7)
    ax1.set_title(f"Top {top_n_genes} Hub Genes & Neighbours", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # ── Panel 2: Gene × Region heatmap ────────────────────────────
    ax2 = axes[0, 1]
    region_edges: list[dict[str, Any]] = []
    for u, v, d in graph.edges(data=True):
        if d.get("label") == "gene_anatomical_entity_association":
            gene_name = str(u).replace("gene:", "")
            region_name = str(v).replace("region:", "")
            region_edges.append({
                "gene": gene_name,
                "region": region_name,
                "spatial_corr": d.get("spatial_correlation", 0),
            })
    if region_edges:
        re_df = pd.DataFrame(region_edges)
        pivot = re_df.pivot_table(index="gene", columns="region",
                                   values="spatial_corr", fill_value=0)
        # Sort by max correlation
        pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
        if len(pivot) > 20:
            pivot = pivot.head(20)
        im = ax2.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax2.set_xticks(range(len(pivot.columns)))
        ax2.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax2.set_yticks(range(len(pivot.index)))
        ax2.set_yticklabels(pivot.index, fontsize=7)
        fig.colorbar(im, ax=ax2, shrink=0.6, label="Spatial correlation")
    ax2.set_title("Gene × Region Spatial Correlation", fontsize=11, fontweight="bold")

    # ── Panel 3: Centrality ranking ───────────────────────────────
    ax3 = axes[1, 0]
    if centrality_df is not None and len(centrality_df) > 0:
        gene_cent = centrality_df[centrality_df["label"] == "gene"].copy()
        gene_cent = gene_cent.head(15)
        gene_cent["name"] = gene_cent.index.str.replace("gene:", "", regex=False)
        bars = ax3.barh(gene_cent["name"], gene_cent["pagerank"],
                        color="#4A90D9", edgecolor="white", linewidth=0.5)
        ax3.set_xlabel("PageRank", fontsize=9)
        ax3.invert_yaxis()
        # Add betweenness as text
        for bar, (_, row) in zip(bars, gene_cent.iterrows()):
            ax3.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
                     f"BC={row['betweenness']:.3f}", va="center", fontsize=6, color="#666")
    ax3.set_title("Gene Hub Ranking (PageRank + Betweenness)", fontsize=11, fontweight="bold")

    # ── Panel 4: Edge composition ─────────────────────────────────
    ax4 = axes[1, 1]
    edge_labels_list = [d.get("label", "unknown") for _, _, d in graph.edges(data=True)]
    edge_counts = pd.Series(edge_labels_list).value_counts()
    nice_labels = [lbl.replace("_", " ").title() for lbl in edge_counts.index]
    colors = ["#4A90D9", "#27AE60", "#E67E22", "#9B59B6", "#E74C3C"][:len(edge_counts)]
    wedges, texts, autotexts = ax4.pie(
        edge_counts.values, labels=nice_labels, autopct="%1.0f%%",
        colors=colors, textprops={"fontsize": 8},
    )
    ax4.set_title(
        f"Edge Composition ({graph.number_of_edges()} total)",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
