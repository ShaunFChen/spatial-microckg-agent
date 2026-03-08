"""Multi-scale graph analytics for the Micro-CKG.

Provides community detection, centrality analysis, and graph queries
that operate directly on the NetworkX knowledge graph.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "detect_communities",
    "compute_centrality",
    "find_bridge_genes",
    "summarise_graph",
]


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------


def detect_communities(
    graph: nx.DiGraph,
) -> dict[str, int]:
    """Partition graph nodes into communities via Louvain on the undirected projection.

    Args:
        graph: Micro-CKG directed graph.

    Returns:
        Dict mapping each node id to its community index (0-based).
    """
    ug = graph.to_undirected()
    communities = nx.community.louvain_communities(ug, seed=42)
    mapping: dict[str, int] = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            mapping[node] = idx

    print(f"  Communities: {len(communities)} detected via Louvain")
    return mapping


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------


def compute_centrality(
    graph: nx.DiGraph,
) -> pd.DataFrame:
    """Compute multiple centrality metrics for every node.

    Metrics: degree, betweenness, PageRank, eigenvector (on the
    undirected projection).

    Args:
        graph: Micro-CKG directed graph.

    Returns:
        DataFrame indexed by node id with centrality columns.
    """
    ug = graph.to_undirected()

    degree = dict(ug.degree())
    betweenness = nx.betweenness_centrality(ug)
    pagerank = nx.pagerank(graph, alpha=0.85)

    try:
        eigenvector = nx.eigenvector_centrality(ug, max_iter=500)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in ug.nodes()}

    df = pd.DataFrame({
        "degree": degree,
        "betweenness": betweenness,
        "pagerank": pagerank,
        "eigenvector": eigenvector,
    })

    # Add node metadata
    labels = nx.get_node_attributes(graph, "label")
    df["label"] = df.index.map(labels)
    df = df.sort_values("pagerank", ascending=False)

    print(f"  Centrality computed for {len(df)} nodes")
    return df


# ---------------------------------------------------------------------------
# Bridge gene analysis
# ---------------------------------------------------------------------------


def find_bridge_genes(
    graph: nx.DiGraph,
    centrality_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Identify genes that bridge multiple communities or node types.

    A bridge gene connects cell types in different anatomical regions
    and has high betweenness centrality.

    Args:
        graph: Micro-CKG directed graph.
        centrality_df: Pre-computed centrality (from
            :func:`compute_centrality`).  Computed on-the-fly if ``None``.

    Returns:
        DataFrame of gene nodes sorted by bridge score (descending).
    """
    if centrality_df is None:
        centrality_df = compute_centrality(graph)

    communities = detect_communities(graph)

    gene_rows: list[dict[str, Any]] = []
    for node, data in graph.nodes(data=True):
        if data.get("label") != "gene":
            continue

        neighbours = set(graph.successors(node)) | set(graph.predecessors(node))
        neighbour_labels = {graph.nodes[n].get("label") for n in neighbours if n in graph.nodes}
        neighbour_communities = {communities.get(n) for n in neighbours if n in communities}
        neighbour_communities.discard(None)

        row = {
            "gene": node,
            "degree": centrality_df.loc[node, "degree"] if node in centrality_df.index else 0,
            "betweenness": centrality_df.loc[node, "betweenness"] if node in centrality_df.index else 0,
            "pagerank": centrality_df.loc[node, "pagerank"] if node in centrality_df.index else 0,
            "n_neighbour_types": len(neighbour_labels),
            "n_communities_bridged": len(neighbour_communities),
        }
        # Bridge score: weighted combination of betweenness and community bridging
        row["bridge_score"] = row["betweenness"] * np.log1p(row["n_communities_bridged"])
        gene_rows.append(row)

    df = pd.DataFrame(gene_rows).sort_values("bridge_score", ascending=False)
    n_bridges = (df["n_communities_bridged"] > 1).sum()
    print(f"  Bridge genes: {n_bridges}/{len(df)} genes bridge >1 community")
    return df


# ---------------------------------------------------------------------------
# Graph summary
# ---------------------------------------------------------------------------


def summarise_graph(
    graph: nx.DiGraph,
) -> dict[str, Any]:
    """Produce a structured summary of graph topology.

    Args:
        graph: Micro-CKG directed graph.

    Returns:
        Dictionary with node/edge counts by type, density, components,
        and diameter of the largest component.
    """
    labels = nx.get_node_attributes(graph, "label")
    edge_labels = nx.get_edge_attributes(graph, "label")

    node_counts: dict[str, int] = {}
    for lbl in labels.values():
        node_counts[lbl] = node_counts.get(lbl, 0) + 1

    edge_counts: dict[str, int] = {}
    for lbl in edge_labels.values():
        edge_counts[lbl] = edge_counts.get(lbl, 0) + 1

    ug = graph.to_undirected()
    components = list(nx.connected_components(ug))
    largest_cc = max(components, key=len) if components else set()

    diameter = -1
    if len(largest_cc) > 1:
        sub = ug.subgraph(largest_cc)
        try:
            diameter = nx.diameter(sub)
        except nx.NetworkXError:
            diameter = -1

    summary = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "density": nx.density(graph),
        "n_components": len(components),
        "largest_component_size": len(largest_cc),
        "diameter": diameter,
    }

    print(f"  Graph summary: {summary['n_nodes']} nodes, {summary['n_edges']} edges, "
          f"density={summary['density']:.4f}, {summary['n_components']} components")
    return summary
