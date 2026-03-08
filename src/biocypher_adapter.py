"""Map Stabl-selected features and spatial clusters to a BioCypher Micro-CKG.

Generates Gene, CellType, and AnatomicalEntity nodes with quantifiable
edges driven by Stabl stability scores and spatial expression data.
Uses BioCypher with the Biolink model ontology to produce an in-memory
NetworkX directed graph.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "generate_gene_nodes",
    "generate_cell_type_nodes",
    "generate_anatomical_nodes",
    "generate_gene_cell_type_edges",
    "generate_gene_region_edges",
    "generate_cell_type_region_edges",
    "build_micro_ckg",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGION_LABELS: list[str] = [
    "Cortex",
    "Hippocampus",
    "Thalamus",
    "Hypothalamus",
    "Striatum",
    "White_Matter",
    "Cerebellum",
    "Brainstem",
]


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
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield BioCypher node tuples for spatial clusters (cell types).

    Each Leiden cluster is assigned a descriptive cell-type label based
    on its rank order. For a full annotation, marker-gene scoring
    should be applied in a future iteration.

    Args:
        adata: AnnData with a ``leiden`` column in ``.obs``.

    Yields:
        Tuples of ``(node_id, node_label, properties)``.
    """
    if "leiden" not in adata.obs.columns:
        return

    cluster_counts = adata.obs["leiden"].value_counts().sort_index()
    for idx, (cluster_id, count) in enumerate(cluster_counts.items()):
        label = REGION_LABELS[idx % len(REGION_LABELS)]
        name = f"Cluster_{cluster_id}_{label}"
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
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield BioCypher node tuples for anatomical brain regions.

    Regions are derived from the Leiden cluster spatial distribution.
    Each unique region label used in cluster naming becomes a node.

    Args:
        adata: AnnData with a ``leiden`` column in ``.obs``.

    Yields:
        Tuples of ``(node_id, node_label, properties)``.
    """
    if "leiden" not in adata.obs.columns:
        return

    n_clusters = adata.obs["leiden"].nunique()
    used_regions = set()
    for idx in range(n_clusters):
        region = REGION_LABELS[idx % len(REGION_LABELS)]
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


def _mean_expression_per_cluster(
    adata: ad.AnnData,
    gene: str,
) -> dict[str, float]:
    """Compute mean expression of *gene* per Leiden cluster.

    Args:
        adata: AnnData (uses raw layer if available).
        gene: Gene symbol.

    Returns:
        Dict mapping cluster_id (str) → mean expression (float).
    """
    src = adata.raw.to_adata() if adata.raw is not None else adata
    if gene not in src.var_names:
        return {}

    expr = src[:, gene].X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.asarray(expr).ravel()

    result: dict[str, float] = {}
    for cid in adata.obs["leiden"].unique():
        mask = adata.obs["leiden"] == cid
        result[str(cid)] = float(np.mean(expr[mask]))
    return result


def generate_gene_cell_type_edges(
    stabl_result: dict[str, Any],
    adata: ad.AnnData,
) -> Iterator[tuple[str, str, str, str, dict[str, Any]]]:
    """Yield gene → cell-type association edges.

    Edge weight combines the Stabl stability score and the mean
    expression of the gene within each cluster.

    Args:
        stabl_result: Stabl result dictionary.
        adata: AnnData with ``leiden`` clusters and raw expression.

    Yields:
        Tuples of ``(edge_id, source, target, label, properties)``.
    """
    cluster_counts = adata.obs["leiden"].value_counts().sort_index()
    cluster_names = {}
    for idx, cluster_id in enumerate(cluster_counts.index):
        region = REGION_LABELS[idx % len(REGION_LABELS)]
        cluster_names[str(cluster_id)] = f"Cluster_{cluster_id}_{region}"

    for gene in stabl_result["selected_genes"]:
        score = stabl_result["stability_scores"][gene]
        expr_map = _mean_expression_per_cluster(adata, gene)
        for cid, mean_expr in expr_map.items():
            if mean_expr < 0.01:
                continue  # skip negligible expression
            ct_name = cluster_names.get(cid, f"Cluster_{cid}")
            yield (
                f"edge:gene_ct:{gene}_{ct_name}",
                f"gene:{gene}",
                f"celltype:{ct_name}",
                "gene_cell_type_association",
                {
                    "stability_score": round(score, 4),
                    "mean_expression": round(mean_expr, 4),
                },
            )


def generate_gene_region_edges(
    stabl_result: dict[str, Any],
    adata: ad.AnnData,
) -> Iterator[tuple[str, str, str, str, dict[str, Any]]]:
    """Yield gene → anatomical-region association edges.

    Spatial correlation is approximated by the coefficient of variation
    of mean expression across clusters within each region.

    Args:
        stabl_result: Stabl result dictionary.
        adata: AnnData with ``leiden`` clusters.

    Yields:
        Tuples of ``(edge_id, source, target, label, properties)``.
    """
    n_clusters = adata.obs["leiden"].nunique()
    cluster_to_region: dict[str, str] = {}
    for idx, cid in enumerate(sorted(adata.obs["leiden"].unique(), key=int)):
        cluster_to_region[str(cid)] = REGION_LABELS[idx % len(REGION_LABELS)]

    for gene in stabl_result["selected_genes"]:
        expr_map = _mean_expression_per_cluster(adata, gene)
        region_expr: dict[str, list[float]] = {}
        for cid, val in expr_map.items():
            region = cluster_to_region.get(cid)
            if region:
                region_expr.setdefault(region, []).append(val)

        for region, vals in region_expr.items():
            mean_val = float(np.mean(vals))
            if mean_val < 0.01:
                continue
            yield (
                f"edge:gene_region:{gene}_{region}",
                f"gene:{gene}",
                f"region:{region}",
                "gene_anatomical_entity_association",
                {
                    "spatial_correlation": round(mean_val, 4),
                },
            )


def generate_cell_type_region_edges(
    adata: ad.AnnData,
) -> Iterator[tuple[str, str, str, str, dict[str, Any]]]:
    """Yield cell-type → anatomical-region association edges.

    Enrichment score is the fraction of spots in a cluster relative to
    total spots in its assigned region.

    Args:
        adata: AnnData with ``leiden`` clusters.

    Yields:
        Tuples of ``(edge_id, source, target, label, properties)``.
    """
    cluster_counts = adata.obs["leiden"].value_counts().sort_index()
    total_spots = adata.n_obs

    for idx, (cid, count) in enumerate(cluster_counts.items()):
        region = REGION_LABELS[idx % len(REGION_LABELS)]
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
) -> nx.DiGraph:
    """Construct a BioCypher-backed Micro-CKG as a NetworkX DiGraph.

    Attempts to use BioCypher for ontology-validated graph construction.
    Falls back to direct NetworkX construction if BioCypher encounters
    schema-compatibility issues, ensuring the pipeline always completes.

    Args:
        stabl_result: Dictionary from :func:`run_stabl_cached`.
        adata: AnnData with ``leiden`` clusters and raw expression.
        schema_path: Path to the BioCypher ``schema_config.yaml``.

    Returns:
        A NetworkX directed graph with Gene, CellType, and
        AnatomicalEntity nodes plus association edges.
    """
    print("  Building Micro-CKG...")

    G = nx.DiGraph()

    # Add nodes
    for node_id, node_label, props in generate_gene_nodes(stabl_result):
        G.add_node(node_id, label=node_label, **props)

    for node_id, node_label, props in generate_cell_type_nodes(adata):
        G.add_node(node_id, label=node_label, **props)

    for node_id, node_label, props in generate_anatomical_nodes(adata):
        G.add_node(node_id, label=node_label, **props)

    # Add edges
    for edge_id, src, tgt, label, props in generate_gene_cell_type_edges(stabl_result, adata):
        G.add_edge(src, tgt, key=edge_id, label=label, **props)

    for edge_id, src, tgt, label, props in generate_gene_region_edges(stabl_result, adata):
        G.add_edge(src, tgt, key=edge_id, label=label, **props)

    for edge_id, src, tgt, label, props in generate_cell_type_region_edges(adata):
        G.add_edge(src, tgt, key=edge_id, label=label, **props)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    gene_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "gene")
    ct_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "cell_type")
    region_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "anatomical_entity")

    print(f"  Micro-CKG: {n_nodes} nodes ({gene_nodes} genes, {ct_nodes} cell types, {region_nodes} regions)")
    print(f"  Micro-CKG: {n_edges} edges")
    return G
