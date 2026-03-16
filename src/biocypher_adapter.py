"""Map Stabl-selected features and spatial clusters to a BioCypher Micro-CKG.

Generates Gene, CellType, and AnatomicalEntity nodes with quantifiable
edges driven by Stabl stability scores, differential expression testing,
and spatial expression data.  Uses BioCypher with the Biolink model
ontology to produce an in-memory NetworkX directed graph.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from biocypher import BioCypher
from biocypher import Graph as BCGraph

__all__ = [
    "generate_gene_nodes",
    "generate_cell_type_nodes",
    "generate_anatomical_nodes",
    "generate_pathway_nodes",
    "generate_disease_nodes",
    "generate_drug_nodes",
    "generate_gene_cell_type_edges",
    "generate_gene_region_edges",
    "generate_cell_type_region_edges",
    "generate_gene_pathway_edges",
    "generate_gene_disease_edges",
    "generate_ppi_edges",
    "generate_drug_gene_edges",
    "build_micro_ckg",
    "build_micro_ckg_agent",
    "save_graph",
    "load_graph",
    "visualize_graph",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DE_PVAL_THRESHOLD = 0.10
_DE_LOG2FC_THRESHOLD = 0.1
_DEFAULT_MIN_GENES = 20
_PATHWAY_PVAL_THRESHOLD = 0.05
_PATHWAY_MAX_NODES = 50

# Default paths for BioCypher configuration files.
_DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "config" / "schema_config.yaml"
_DEFAULT_BIOCYPHER_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "biocypher_config.yaml"
)

# Compiled regex for sanitising arbitrary text into safe node-ID segments.
_UNSAFE_ID_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_gene_list(
    stabl_result: dict[str, Any],
    min_genes: int = _DEFAULT_MIN_GENES,
) -> dict[str, Any]:
    """Expand Stabl results to include at least *min_genes* genes.

    When Stabl selects fewer than *min_genes*, the top-ranked genes by
    stability score are added (marked ``is_selected=False``).

    Args:
        stabl_result: Dictionary from :func:`run_stabl_cached`.
        min_genes: Minimum number of genes to include.

    Returns:
        A shallow copy of *stabl_result* with expanded
        ``selected_genes`` and ``stability_scores``.
    """
    selected = list(stabl_result["selected_genes"])
    scores = dict(stabl_result["stability_scores"])

    if len(selected) >= min_genes:
        return stabl_result

    all_scores = stabl_result.get("all_scores")
    all_names = stabl_result.get("all_feature_names")
    if all_scores is None or all_names is None:
        return stabl_result

    # Rank all features by stability score, descending
    ranked = sorted(
        zip(all_names, all_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    selected_set = set(selected)
    for name, score in ranked:
        if len(selected) >= min_genes:
            break
        if name not in selected_set:
            selected.append(name)
            selected_set.add(name)
            scores[name] = float(score)

    expanded = dict(stabl_result)
    expanded["selected_genes"] = selected
    expanded["stability_scores"] = scores
    expanded["n_selected_original"] = stabl_result["n_selected"]
    expanded["n_selected"] = len(selected)
    return expanded


# ---------------------------------------------------------------------------
# Node Generators
# ---------------------------------------------------------------------------


def generate_gene_nodes(
    stabl_result: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield BioCypher node tuples for Stabl-selected genes.

    Genes from the original FDP+ selection are marked
    ``is_selected=True``; genes added via the *min_genes* expansion
    are marked ``is_selected=False``.

    If ``stabl_result`` contains an ``"ortho_map"`` key (a dict mapping
    mouse gene symbol → human ortholog symbol), the ``human_ortholog``
    property is included in each gene node's attributes when available.

    Args:
        stabl_result: Dictionary returned by ``run_stabl_selection``
            (or expanded by :func:`_expand_gene_list`).  May optionally
            contain an ``"ortho_map"`` key.

    Yields:
        Tuples of ``(node_id, node_label, properties)`` for each gene.
    """
    n_orig = stabl_result.get("n_selected_original", len(stabl_result["selected_genes"]))
    ortho_map = stabl_result.get("ortho_map", {})
    for idx, gene in enumerate(stabl_result["selected_genes"]):
        score = stabl_result["stability_scores"][gene]
        props: dict[str, Any] = {
            "symbol": gene,
            "stability_score": round(score, 4),
            "is_selected": idx < n_orig,
        }
        human_sym = ortho_map.get(gene)
        if human_sym:
            props["human_ortholog"] = human_sym
        yield (
            f"gene:{gene}",
            "gene",
            props,
        )


def generate_drug_nodes(
    drug_df: pd.DataFrame,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield BioCypher node tuples for approved drug compounds.

    Expects a DataFrame produced by
    :func:`~src.external_knowledge.get_drug_targets` with at least the
    columns ``drug_name``, ``mechanism_of_action``, and ``max_phase``.

    Args:
        drug_df: DataFrame of drug-target entries from ChEMBL.

    Yields:
        Tuples of ``(node_id, node_label, properties)`` for each
        unique drug compound.
    """
    if drug_df.empty:
        return
    seen: set[str] = set()
    for _, row in drug_df.iterrows():
        name = str(row.get("drug_name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        node_id = f"drug:{name.lower().replace(' ', '_')}"
        yield (
            node_id,
            "drug",
            {
                "name": name,
                "mechanism_of_action": str(row.get("mechanism_of_action", "")),
                "max_phase": int(float(row.get("max_phase", 0) or 0)),
            },
        )


def generate_drug_gene_edges(
    drug_df: pd.DataFrame,
    ortho_map: dict[str, str] | None = None,
) -> Iterator[tuple[str, str, str, str, dict[str, Any]]]:
    """Yield drug → gene target edges from ChEMBL drug-target data.

    Each edge links a drug compound node to a mouse gene node via the
    human-ortholog mapping.  Only genes that appear in the graph are
    targeted.

    Args:
        drug_df: DataFrame from :func:`~src.external_knowledge.get_drug_targets`
            with columns ``gene``, ``drug_name``, ``mechanism_of_action``,
            ``max_phase``.
        ortho_map: Dict mapping mouse gene symbol → human ortholog symbol.
            When supplied, the edge label includes the human ortholog.

    Yields:
        Tuples of ``(edge_id, source, target, label, properties)``.
    """
    if drug_df.empty:
        return
    if ortho_map is None:
        ortho_map = {}

    # Build reverse map: human symbol → mouse gene symbol
    human_to_mouse: dict[str, str] = {v: k for k, v in ortho_map.items() if v}

    for _, row in drug_df.iterrows():
        human_gene = str(row.get("gene", "")).strip()
        drug_name = str(row.get("drug_name", "")).strip()
        if not human_gene or not drug_name:
            continue
        # Resolve to mouse gene via reverse ortho mapping; fall back to human symbol
        mouse_gene = human_to_mouse.get(human_gene, human_gene)
        drug_node = f"drug:{drug_name.lower().replace(' ', '_')}"
        gene_node = f"gene:{mouse_gene}"
        edge_id = f"edge:drug_gene:{drug_name}_{mouse_gene}"
        yield (
            edge_id,
            drug_node,
            gene_node,
            "drug_gene_association",
            {
                "mechanism_of_action": str(row.get("mechanism_of_action", "")),
                "max_phase": int(float(row.get("max_phase", 0) or 0)),
                "human_target": human_gene,
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


def generate_pathway_nodes(
    enrich_df: pd.DataFrame,
    *,
    pval_threshold: float = _PATHWAY_PVAL_THRESHOLD,
    max_pathways: int = _PATHWAY_MAX_NODES,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield BioCypher node tuples for enriched biological pathways.

    Selects the top significant pathways from GO/KEGG/Reactome enrichment
    results, sorted by adjusted p-value.

    Args:
        enrich_df: DataFrame from
            :func:`~src.external_knowledge.run_go_enrichment` with columns
            ``Term``, ``Adjusted P-value``, ``Gene_set_library``,
            ``Genes``, and ``Combined Score``.
        pval_threshold: Maximum adjusted p-value for inclusion.
        max_pathways: Upper bound on pathway nodes emitted.

    Yields:
        Tuples of ``(node_id, node_label, properties)`` for each unique
        enriched pathway term.
    """
    if enrich_df.empty:
        return
    sig = enrich_df[enrich_df["Adjusted P-value"] < pval_threshold].copy()
    sig = sig.sort_values("Adjusted P-value").head(max_pathways)
    seen: set[str] = set()
    for _, row in sig.iterrows():
        term = str(row.get("Term", "")).strip()
        if not term or term in seen:
            continue
        seen.add(term)
        node_id = f"pathway:{_UNSAFE_ID_RE.sub('_', term.lower()).strip('_')[:80]}"
        yield (
            node_id,
            "biological_process",
            {
                "name": term,
                "source_library": str(row.get("Gene_set_library", "")),
                "pval_adj": float(row.get("Adjusted P-value", 1.0)),
                "combined_score": float(row.get("Combined Score", 0.0)),
            },
        )


def generate_disease_nodes(
    disease_df: pd.DataFrame,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield BioCypher node tuples for gene-disease associations.

    Args:
        disease_df: DataFrame from
            :func:`~src.external_knowledge.get_disease_associations` with
            columns ``gene``, ``disease_name``, ``disease_id``.

    Yields:
        Tuples of ``(node_id, node_label, properties)`` for each unique
        disease.
    """
    if disease_df.empty:
        return
    seen: set[str] = set()
    for _, row in disease_df.iterrows():
        name = str(row.get("disease_name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        disease_id = str(row.get("disease_id", "")).strip()
        node_id = (
            f"disease:{disease_id}"
            if disease_id
            else f"disease:{_UNSAFE_ID_RE.sub('_', name.lower()).strip('_')[:80]}"
        )
        yield (
            node_id,
            "disease",
            {
                "name": name,
                "disease_id": disease_id,
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


def generate_gene_pathway_edges(
    stabl_result: dict[str, Any],
    enrich_df: pd.DataFrame,
    ortho_map: dict[str, str] | None = None,
    *,
    pval_threshold: float = _PATHWAY_PVAL_THRESHOLD,
) -> Iterator[tuple[str, str, str, str, dict[str, Any]]]:
    """Yield gene → pathway participation edges.

    A mouse gene is linked to a pathway when its human ortholog appears
    in the pathway's enriched gene list from Enrichr.

    Args:
        stabl_result: Stabl result dict containing ``selected_genes``.
        enrich_df: DataFrame from
            :func:`~src.external_knowledge.run_go_enrichment` with columns
            ``Term``, ``Adjusted P-value``, ``Gene_set_library``,
            ``Genes``, and ``Combined Score``.
        ortho_map: Dict mapping mouse gene symbol → human ortholog symbol.
        pval_threshold: Maximum adjusted p-value for pathway inclusion.

    Yields:
        Tuples of ``(edge_id, source, target, label, properties)``.
    """
    if enrich_df.empty:
        return
    if ortho_map is None:
        ortho_map = {}
    # Reverse map: human symbol UPPERCASE → mouse gene symbol
    human_to_mouse: dict[str, str] = {v.upper(): k for k, v in ortho_map.items() if v}
    selected_set = set(stabl_result["selected_genes"])
    sig = enrich_df[enrich_df["Adjusted P-value"] < pval_threshold].copy()
    for _, row in sig.iterrows():
        term = str(row.get("Term", "")).strip()
        if not term:
            continue
        pathway_id = f"pathway:{_UNSAFE_ID_RE.sub('_', term.lower()).strip('_')[:80]}"
        genes_str = str(row.get("Genes", ""))
        gene_list = [
            g.strip().upper()
            for g in genes_str.replace(";", ",").split(",")
            if g.strip()
        ]
        for human_upper in gene_list:
            mouse_gene = human_to_mouse.get(human_upper)
            if not mouse_gene or mouse_gene not in selected_set:
                continue
            edge_id = (
                f"edge:gene_pathway:{mouse_gene}_"
                f"{_UNSAFE_ID_RE.sub('_', term.lower())[:40]}"
            )
            yield (
                edge_id,
                f"gene:{mouse_gene}",
                pathway_id,
                "gene_participates_in_pathway",
                {
                    "pval_adj": float(row.get("Adjusted P-value", 1.0)),
                    "combined_score": float(row.get("Combined Score", 0.0)),
                    "source_library": str(row.get("Gene_set_library", "")),
                },
            )


def generate_gene_disease_edges(
    stabl_result: dict[str, Any],
    disease_df: pd.DataFrame,
    ortho_map: dict[str, str] | None = None,
) -> Iterator[tuple[str, str, str, str, dict[str, Any]]]:
    """Yield gene → disease association edges.

    A mouse gene is linked to a disease when its human ortholog has a
    documented association in the mygene disease database.

    Args:
        stabl_result: Stabl result dict containing ``selected_genes``.
        disease_df: DataFrame from
            :func:`~src.external_knowledge.get_disease_associations` with
            columns ``gene``, ``disease_name``, ``disease_id``.
        ortho_map: Dict mapping mouse gene symbol → human ortholog symbol.

    Yields:
        Tuples of ``(edge_id, source, target, label, properties)``.
    """
    if disease_df.empty:
        return
    if ortho_map is None:
        ortho_map = {}
    human_to_mouse: dict[str, str] = {v.upper(): k for k, v in ortho_map.items() if v}
    selected_set = set(stabl_result["selected_genes"])
    for _, row in disease_df.iterrows():
        human_gene = str(row.get("gene", "")).strip().upper()
        disease_name = str(row.get("disease_name", "")).strip()
        if not human_gene or not disease_name:
            continue
        mouse_gene = human_to_mouse.get(human_gene)
        if not mouse_gene or mouse_gene not in selected_set:
            continue
        disease_id = str(row.get("disease_id", "")).strip()
        disease_node = (
            f"disease:{disease_id}"
            if disease_id
            else f"disease:{_UNSAFE_ID_RE.sub('_', disease_name.lower()).strip('_')[:80]}"
        )
        edge_id = (
            f"edge:gene_disease:{mouse_gene}_"
            f"{_UNSAFE_ID_RE.sub('_', disease_name.lower())[:40]}"
        )
        yield (
            edge_id,
            f"gene:{mouse_gene}",
            disease_node,
            "gene_associated_with_disease",
            {
                "disease_id": disease_id,
                "score": float(row.get("score", 0.0)),
            },
        )


def generate_ppi_edges(
    ppi_df: pd.DataFrame,
    ortho_map: dict[str, str] | None = None,
) -> Iterator[tuple[str, str, str, str, dict[str, Any]]]:
    """Yield gene ↔ gene protein-protein interaction edges from STRING.

    Both endpoints must resolve to mouse gene nodes via *ortho_map*.
    Duplicate undirected pairs and self-loops are skipped.

    Args:
        ppi_df: DataFrame from
            :func:`~src.external_knowledge.get_string_ppi` with columns
            ``gene1``, ``gene2``, ``score``.
        ortho_map: Dict mapping mouse gene symbol → human ortholog symbol.

    Yields:
        Tuples of ``(edge_id, source, target, label, properties)``.
    """
    if ppi_df.empty:
        return
    if ortho_map is None:
        ortho_map = {}
    human_to_mouse: dict[str, str] = {v.upper(): k for k, v in ortho_map.items() if v}
    seen_pairs: set[frozenset[str]] = set()
    for _, row in ppi_df.iterrows():
        g1_upper = str(row.get("gene1", "")).strip().upper()
        g2_upper = str(row.get("gene2", "")).strip().upper()
        mouse1 = human_to_mouse.get(g1_upper)
        mouse2 = human_to_mouse.get(g2_upper)
        if not mouse1 or not mouse2 or mouse1 == mouse2:
            continue
        pair: frozenset[str] = frozenset({mouse1, mouse2})
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        edge_id = f"edge:ppi:{min(mouse1, mouse2)}_{max(mouse1, mouse2)}"
        yield (
            edge_id,
            f"gene:{mouse1}",
            f"gene:{mouse2}",
            "gene_interacts_with_gene",
            {
                "string_score": float(row.get("score", 0)),
                "human_gene1": g1_upper,
                "human_gene2": g2_upper,
            },
        )


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------


def _collect_nodes(
    expanded: dict[str, Any],
    adata: ad.AnnData,
    cluster_annotation: dict[str, str] | None,
    drug_df: pd.DataFrame | None,
    enrich_df: pd.DataFrame | None,
    disease_df: pd.DataFrame | None,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Collect all node tuples from the individual generators.

    Args:
        expanded: Expanded Stabl result dict.
        adata: AnnData with ``leiden`` clusters.
        cluster_annotation: Cluster → region mapping.
        drug_df: Optional drug-target DataFrame.
        enrich_df: Optional enrichment DataFrame.
        disease_df: Optional disease-association DataFrame.

    Returns:
        List of ``(node_id, input_label, properties)`` tuples.
    """
    nodes: list[tuple[str, str, dict[str, Any]]] = []
    nodes.extend(generate_gene_nodes(expanded))
    nodes.extend(generate_cell_type_nodes(adata, cluster_annotation))
    nodes.extend(generate_anatomical_nodes(adata, cluster_annotation))
    if drug_df is not None and not drug_df.empty:
        nodes.extend(generate_drug_nodes(drug_df))
    if enrich_df is not None and not enrich_df.empty:
        nodes.extend(generate_pathway_nodes(enrich_df))
    if disease_df is not None and not disease_df.empty:
        nodes.extend(generate_disease_nodes(disease_df))
    return nodes


def _collect_edges(
    expanded: dict[str, Any],
    adata: ad.AnnData,
    cluster_annotation: dict[str, str] | None,
    de_df: pd.DataFrame,
    drug_df: pd.DataFrame | None,
    ortho_map: dict[str, str] | None,
    enrich_df: pd.DataFrame | None,
    disease_df: pd.DataFrame | None,
    ppi_df: pd.DataFrame | None,
    node_ids: set[str],
) -> list[tuple[str | None, str, str, str, dict[str, Any]]]:
    """Collect all edge tuples from the individual generators.

    Only edges whose source and target exist in *node_ids* are kept.

    Args:
        expanded: Expanded Stabl result dict.
        adata: AnnData with ``leiden`` clusters.
        cluster_annotation: Cluster → region mapping.
        de_df: Pre-computed DE results.
        drug_df: Optional drug-target DataFrame.
        ortho_map: Optional mouse→human gene mapping.
        enrich_df: Optional enrichment DataFrame.
        disease_df: Optional disease-association DataFrame.
        ppi_df: Optional STRING PPI DataFrame.
        node_ids: Set of node IDs present in the graph.

    Returns:
        List of ``(edge_id, source, target, input_label, properties)``
        tuples.
    """
    edges: list[tuple[str | None, str, str, str, dict[str, Any]]] = []

    for eid, src, tgt, lbl, props in generate_gene_cell_type_edges(
        expanded, adata, cluster_annotation, de_df
    ):
        edges.append((eid, src, tgt, lbl, props))

    for eid, src, tgt, lbl, props in generate_gene_region_edges(
        expanded, adata, cluster_annotation, de_df
    ):
        edges.append((eid, src, tgt, lbl, props))

    for eid, src, tgt, lbl, props in generate_cell_type_region_edges(
        adata, cluster_annotation
    ):
        edges.append((eid, src, tgt, lbl, props))

    if drug_df is not None and not drug_df.empty:
        for eid, src, tgt, lbl, props in generate_drug_gene_edges(drug_df, ortho_map):
            if tgt in node_ids:
                edges.append((eid, src, tgt, lbl, props))

    if enrich_df is not None and not enrich_df.empty:
        for eid, src, tgt, lbl, props in generate_gene_pathway_edges(
            expanded, enrich_df, ortho_map
        ):
            if src in node_ids:
                edges.append((eid, src, tgt, lbl, props))

    if disease_df is not None and not disease_df.empty:
        for eid, src, tgt, lbl, props in generate_gene_disease_edges(
            expanded, disease_df, ortho_map
        ):
            if src in node_ids:
                edges.append((eid, src, tgt, lbl, props))

    if ppi_df is not None and not ppi_df.empty:
        for eid, src, tgt, lbl, props in generate_ppi_edges(ppi_df, ortho_map):
            if src in node_ids and tgt in node_ids:
                edges.append((eid, src, tgt, lbl, props))

    return edges


def _build_input_label_map(schema_path: Path | str) -> dict[str, str]:
    """Build ``{input_label: schema_key}`` lookup from the schema YAML.

    Args:
        schema_path: Path to the ``schema_config.yaml``.

    Returns:
        Mapping from ``input_label`` values to their schema key names.
    """
    with open(schema_path) as fh:
        schema = yaml.safe_load(fh)
    mapping: dict[str, str] = {}
    for key, cfg in schema.items():
        if isinstance(cfg, dict) and "input_label" in cfg:
            mapping[cfg["input_label"]] = key
    return mapping


def _postprocess_networkx(
    G: nx.DiGraph,
    input_label_map: dict[str, str],
) -> nx.DiGraph:
    """Normalise BioCypher ``to_networkx()`` output for backward compat.

    BioCypher stores the node type under ``node_label`` and the edge
    type under ``relationship_label``.  Downstream code in this project
    expects a ``label`` attribute whose value is the ``input_label``
    (e.g. ``gene``, ``cell_type``, ``gene_cell_type_association``).

    This function copies the input_label into a ``label`` attribute on
    every node and edge so that existing analytics and visualisation code
    continues to work unchanged.

    Args:
        G: NetworkX DiGraph returned by ``BioCypher.to_networkx()``.
        input_label_map: Mapping ``{input_label: schema_key}``.

    Returns:
        The same graph, mutated in-place, with ``label`` attributes set.
    """
    # Reverse map: schema_key → input_label
    schema_to_input: dict[str, str] = {v: k for k, v in input_label_map.items()}

    for _, d in G.nodes(data=True):
        schema_name = d.get("node_label", "")
        d["label"] = schema_to_input.get(schema_name, schema_name.replace(" ", "_"))
        # GraphML does not support None values — replace with empty string.
        for key in list(d):
            if d[key] is None:
                d[key] = ""

    for _, _, d in G.edges(data=True):
        schema_name = d.get("relationship_label", "")
        d["label"] = schema_to_input.get(schema_name, schema_name.replace(" ", "_"))
        for key in list(d):
            if d[key] is None:
                d[key] = ""

    return G


def build_micro_ckg(
    stabl_result: dict[str, Any],
    adata: ad.AnnData,
    schema_path: Path | str | None = None,
    cluster_annotation: dict[str, str] | None = None,
    min_genes: int = _DEFAULT_MIN_GENES,
    drug_df: pd.DataFrame | None = None,
    ortho_map: dict[str, str] | None = None,
    enrich_df: pd.DataFrame | None = None,
    disease_df: pd.DataFrame | None = None,
    ppi_df: pd.DataFrame | None = None,
) -> nx.DiGraph:
    """Construct a BioCypher-backed Micro-CKG as a NetworkX DiGraph.

    Pipes node and edge generators through the real ``BioCypher`` ETL
    engine, which validates entities against the Biolink ontology via
    ``schema_config.yaml``, deduplicates nodes, and produces a
    schema-validated NetworkX DiGraph.

    Performs Wilcoxon DE testing across Leiden clusters and uses the
    results to create statistically-filtered edges.

    When Stabl selects fewer than *min_genes*, the top-ranked genes by
    stability score are included to ensure the graph is informative.

    Biological context (pathways, diseases, PPIs) is added when the
    corresponding DataFrames are supplied.  Drug nodes from ChEMBL are
    added when *drug_df* is supplied.

    Args:
        stabl_result: Dictionary from :func:`run_stabl_cached`.
        adata: AnnData with ``leiden`` clusters and raw expression.
        schema_path: Path to the BioCypher ``schema_config.yaml``.
            Defaults to ``config/schema_config.yaml``.
        cluster_annotation: Dict mapping cluster id to region label
            (from :func:`annotate_clusters`).
        min_genes: Minimum number of gene nodes (default 20).
        drug_df: Optional DataFrame from
            :func:`~src.external_knowledge.get_drug_targets` containing
            drug-target associations from ChEMBL.
        ortho_map: Optional dict mapping mouse gene symbol → human
            ortholog symbol.
        enrich_df: Optional DataFrame from
            :func:`~src.external_knowledge.run_go_enrichment`.
        disease_df: Optional DataFrame from
            :func:`~src.external_knowledge.get_disease_associations`.
        ppi_df: Optional DataFrame from
            :func:`~src.external_knowledge.get_string_ppi`.

    Returns:
        A NetworkX directed graph with Gene, CellType, AnatomicalEntity,
        and optionally Pathway, Disease, and Drug nodes, plus
        DE-filtered and biological-context association edges.
    """
    schema_path = Path(schema_path) if schema_path else _DEFAULT_SCHEMA_PATH

    # Expand gene list if fewer than min_genes were selected
    expanded = _expand_gene_list(stabl_result, min_genes=min_genes)
    n_orig = expanded.get("n_selected_original", len(expanded["selected_genes"]))
    if len(expanded["selected_genes"]) > n_orig:
        print(
            f"  Expanded gene list: {n_orig} Stabl-selected → "
            f"{len(expanded['selected_genes'])} genes (min_genes={min_genes})"
        )

    # Attach ortho_map so generate_gene_nodes can embed human_ortholog
    if ortho_map:
        expanded = dict(expanded)
        expanded["ortho_map"] = ortho_map

    print("  Running DE testing (Wilcoxon rank-sum)...")
    de_df = _run_de_analysis(adata, expanded["selected_genes"])
    sig = de_df[
        (de_df["pval_adj"] < _DE_PVAL_THRESHOLD)
        & (de_df["log2fc"].abs() >= _DE_LOG2FC_THRESHOLD)
    ]
    print(f"  DE results: {len(de_df)} tests, {len(sig)} significant")

    # -- Collect all nodes and edges from generators ------------------
    nodes = _collect_nodes(
        expanded, adata, cluster_annotation, drug_df, enrich_df, disease_df,
    )
    node_ids = {n[0] for n in nodes}
    edges = _collect_edges(
        expanded, adata, cluster_annotation, de_df,
        drug_df, ortho_map, enrich_df, disease_df, ppi_df, node_ids,
    )

    # -- Pipe through BioCypher for schema validation -----------------
    print("  Validating against Biolink ontology via BioCypher...")
    bc = BioCypher(
        offline=False,
        dbms="networkx",
        schema_config_path=str(schema_path),
        biocypher_config_path=str(_DEFAULT_BIOCYPHER_CONFIG_PATH),
    )
    # BioCypher 0.12.x initialises internal lists to None; workaround.
    bc._nodes = []  # noqa: SLF001
    bc._edges = []  # noqa: SLF001

    bc.add_nodes(nodes)
    # BioCypher edge tuples: (optional_id, source, target, input_label, props)
    bc.add_edges(edges)

    G: nx.DiGraph = bc.to_networkx()

    # -- Post-process for backward-compatible ``label`` attribute -----
    il_map = _build_input_label_map(schema_path)
    _postprocess_networkx(G, il_map)

    # -- QC logging ---------------------------------------------------
    bc.log_missing_input_labels()
    bc.log_duplicates()

    # -- Summary print ------------------------------------------------
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    gene_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "gene")
    ct_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "cell_type")
    region_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "anatomical_entity")
    drug_node_count = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "drug")
    pathway_node_count = sum(
        1 for _, d in G.nodes(data=True) if d.get("label") == "biological_process"
    )
    disease_node_count = sum(
        1 for _, d in G.nodes(data=True) if d.get("label") == "disease"
    )

    node_breakdown = f"{gene_nodes} genes, {ct_nodes} cell types, {region_nodes} regions"
    if drug_node_count:
        node_breakdown += f", {drug_node_count} drugs"
    if pathway_node_count:
        node_breakdown += f", {pathway_node_count} pathways"
    if disease_node_count:
        node_breakdown += f", {disease_node_count} diseases"
    print(f"  Micro-CKG: {n_nodes} nodes ({node_breakdown})")
    print(f"  Micro-CKG: {n_edges} edges (DE-filtered, schema-validated)")
    return G


def build_micro_ckg_agent(
    stabl_result: dict[str, Any],
    adata: ad.AnnData,
    schema_path: Path | str | None = None,
    cluster_annotation: dict[str, str] | None = None,
    min_genes: int = _DEFAULT_MIN_GENES,
    drug_df: pd.DataFrame | None = None,
    ortho_map: dict[str, str] | None = None,
    enrich_df: pd.DataFrame | None = None,
    disease_df: pd.DataFrame | None = None,
    ppi_df: pd.DataFrame | None = None,
) -> BCGraph:
    """Build a Micro-CKG using the BioCypher Agent ``Graph`` class.

    Returns a pure-Python ``biocypher.Graph`` instance with built-in
    type-aware indexing, ``find_paths``, ``get_subgraph``, and
    ``to_json`` support — ideal for LLM agent integration and
    interactive exploration.

    The same node / edge generators used in :func:`build_micro_ckg` are
    reused; the only difference is the output container.

    Args:
        stabl_result: Dictionary from :func:`run_stabl_cached`.
        adata: AnnData with ``leiden`` clusters and raw expression.
        schema_path: Path to the ``schema_config.yaml`` (used for
            logging only in this pathway).
        cluster_annotation: Dict mapping cluster id to region label.
        min_genes: Minimum number of gene nodes (default 20).
        drug_df: Optional drug-target DataFrame from ChEMBL.
        ortho_map: Optional mouse → human gene symbol mapping.
        enrich_df: Optional enrichment DataFrame.
        disease_df: Optional disease-association DataFrame.
        ppi_df: Optional STRING PPI DataFrame.

    Returns:
        A ``biocypher.Graph`` directed graph with type indexes,
        path finding, subgraph extraction, and JSON serialisation.
    """
    # Expand gene list
    expanded = _expand_gene_list(stabl_result, min_genes=min_genes)
    n_orig = expanded.get("n_selected_original", len(expanded["selected_genes"]))
    if len(expanded["selected_genes"]) > n_orig:
        print(
            f"  Expanded gene list: {n_orig} Stabl-selected → "
            f"{len(expanded['selected_genes'])} genes (min_genes={min_genes})"
        )
    if ortho_map:
        expanded = dict(expanded)
        expanded["ortho_map"] = ortho_map

    print("  Running DE testing (Wilcoxon rank-sum)...")
    de_df = _run_de_analysis(adata, expanded["selected_genes"])
    sig = de_df[
        (de_df["pval_adj"] < _DE_PVAL_THRESHOLD)
        & (de_df["log2fc"].abs() >= _DE_LOG2FC_THRESHOLD)
    ]
    print(f"  DE results: {len(de_df)} tests, {len(sig)} significant")

    # Collect all nodes and edges
    nodes = _collect_nodes(
        expanded, adata, cluster_annotation, drug_df, enrich_df, disease_df,
    )
    node_ids = {n[0] for n in nodes}
    edges = _collect_edges(
        expanded, adata, cluster_annotation, de_df,
        drug_df, ortho_map, enrich_df, disease_df, ppi_df, node_ids,
    )

    # Build BioCypher Agent Graph
    print("  Building BioCypher Agent Graph...")
    g = BCGraph("micro_ckg", directed=True)

    for node_id, node_type, props in nodes:
        g.add_node(node_id, node_type, props)

    for edge_id, src, tgt, edge_type, props in edges:
        eid = edge_id if edge_id else f"e:{src}_{tgt}_{edge_type}"
        g.add_edge(eid, edge_type, src, tgt, props)

    stats = g.get_statistics()
    basic = stats["basic"]
    print(
        f"  Agent Graph: {basic['nodes']} nodes "
        f"({dict(stats['node_types'])})"
    )
    print(
        f"  Agent Graph: {basic['edges']} edges "
        f"({dict(stats['edge_types'])})"
    )
    return g


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
