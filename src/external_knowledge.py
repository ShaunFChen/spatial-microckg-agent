"""External biological knowledge integration.

Queries public databases to enrich the knowledge graph with:
- Mouse-to-human ortholog mapping (mygene)
- GO / KEGG / Reactome pathway enrichment (gseapy)
- Protein-protein interactions from STRING
- Drug-target associations from ChEMBL
- Gene-disease associations (mygene)

All API results are cached locally under ``cache/external/`` via
:mod:`joblib` to avoid redundant network calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "map_orthologs",
    "run_go_enrichment",
    "get_string_ppi",
    "get_drug_targets",
    "get_disease_associations",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR: Path = Path("cache/external")
_STRING_API_URL: str = "https://string-db.org/api"
_STRING_SPECIES_HUMAN: int = 9606
_STRING_SCORE_THRESHOLD: int = 400
_ENRICHR_GENE_SET_LIBRARIES: list[str] = [
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
]
_ENRICHR_PVAL_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache_path(prefix: str, key: str) -> Path:
    """Build a deterministic cache file path for a query."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return _CACHE_DIR / f"{prefix}_{h}.json"


def _load_cache(path: Path) -> Any | None:
    """Load cached JSON data or return None."""
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_cache(path: Path, data: Any) -> None:
    """Persist data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_orthologs(
    mouse_genes: list[str],
) -> pd.DataFrame:
    """Map mouse gene symbols to human orthologs via mygene.

    Args:
        mouse_genes: List of mouse gene symbols (e.g. ``["Gfap", "Mbp"]``).

    Returns:
        DataFrame with columns ``mouse_symbol``, ``human_symbol``,
        ``human_entrezgene``, ``ensembl_gene``.
    """
    cache_key = ",".join(sorted(mouse_genes))
    cache_file = _cache_path("orthologs", cache_key)
    cached = _load_cache(cache_file)
    if cached is not None:
        print(f"  Orthologs loaded from cache ({len(cached)} mappings)")
        return pd.DataFrame(cached)

    import mygene  # noqa: PLC0415 — lazy import for optional dep
    mg = mygene.MyGeneInfo()

    # Query mouse genes (taxid 10090)
    results = mg.querymany(
        mouse_genes,
        scopes="symbol",
        fields="symbol,homologene",
        species="mouse",
        returnall=False,
    )

    rows: list[dict[str, str | None]] = []
    homologene_ids: list[int] = []
    mouse_map: dict[int, str] = {}

    for hit in results:
        if "notfound" in hit and hit["notfound"]:
            continue
        hg = hit.get("homologene")
        if hg and "id" in hg:
            hg_id = hg["id"]
            homologene_ids.append(hg_id)
            mouse_map[hg_id] = hit.get("symbol", hit.get("query", ""))

    # Fetch human orthologs by homologene id
    if homologene_ids:
        human_results = mg.querymany(
            [str(h) for h in set(homologene_ids)],
            scopes="homologene.id",
            fields="symbol,entrezgene,ensembl.gene",
            species="human",
            returnall=False,
        )
        for hr in human_results:
            if "notfound" in hr and hr["notfound"]:
                continue
            hg_id = int(hr.get("query", 0))
            mouse_sym = mouse_map.get(hg_id)
            ensembl = hr.get("ensembl", {})
            if isinstance(ensembl, list):
                ensembl = ensembl[0]
            rows.append({
                "mouse_symbol": mouse_sym,
                "human_symbol": hr.get("symbol"),
                "human_entrezgene": str(hr.get("entrezgene", "")),
                "ensembl_gene": ensembl.get("gene", "") if isinstance(ensembl, dict) else "",
            })

    df = pd.DataFrame(rows)
    _save_cache(cache_file, rows)
    print(f"  Ortholog mapping: {len(df)}/{len(mouse_genes)} genes mapped to human")
    return df


def run_go_enrichment(
    human_genes: list[str],
    gene_set_libraries: list[str] | None = None,
) -> pd.DataFrame:
    """Run gene-set enrichment analysis via Enrichr.

    Args:
        human_genes: List of human gene symbols.
        gene_set_libraries: Enrichr libraries to query.  Defaults to
            GO BP/MF, KEGG, and Reactome.

    Returns:
        DataFrame with columns including ``Term``, ``Adjusted P-value``,
        ``Gene_set`` (library name), ``Genes``, ``Combined Score``.
    """
    if gene_set_libraries is None:
        gene_set_libraries = _ENRICHR_GENE_SET_LIBRARIES

    cache_key = ",".join(sorted(human_genes)) + "|" + ",".join(gene_set_libraries)
    cache_file = _cache_path("enrichment", cache_key)
    cached = _load_cache(cache_file)
    if cached is not None:
        print(f"  GO enrichment loaded from cache")
        return pd.DataFrame(cached)

    import gseapy  # noqa: PLC0415

    all_results: list[pd.DataFrame] = []
    for lib in gene_set_libraries:
        try:
            enr = gseapy.enrichr(
                gene_list=human_genes,
                gene_sets=lib,
                organism="human",
                outdir=None,
                no_plot=True,
            )
            res_df = enr.results.copy()
            res_df["Gene_set_library"] = lib
            all_results.append(res_df)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: Enrichr query failed for {lib}: {exc}")
            continue

    if all_results:
        df = pd.concat(all_results, ignore_index=True)
    else:
        df = pd.DataFrame()

    sig = df[df["Adjusted P-value"] < _ENRICHR_PVAL_THRESHOLD] if len(df) > 0 else df
    print(f"  GO enrichment: {len(sig)} significant terms (of {len(df)} total)")

    _save_cache(cache_file, df.to_dict(orient="records"))
    return df


def get_string_ppi(
    human_genes: list[str],
    *,
    score_threshold: int = _STRING_SCORE_THRESHOLD,
) -> pd.DataFrame:
    """Fetch protein-protein interactions from STRING.

    Args:
        human_genes: Human gene symbols.
        score_threshold: Minimum combined score (0-1000).

    Returns:
        DataFrame with columns ``protein1``, ``protein2``, ``score``,
        ``gene1``, ``gene2``.
    """
    cache_key = ",".join(sorted(human_genes)) + f"|{score_threshold}"
    cache_file = _cache_path("string_ppi", cache_key)
    cached = _load_cache(cache_file)
    if cached is not None:
        print(f"  STRING PPI loaded from cache ({len(cached)} interactions)")
        return pd.DataFrame(cached)

    import requests  # noqa: PLC0415

    url = f"{_STRING_API_URL}/json/network"
    params = {
        "identifiers": "%0d".join(human_genes),
        "species": _STRING_SPECIES_HUMAN,
        "required_score": score_threshold,
        "caller_identity": "spatial_microckg_agent",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows: list[dict[str, Any]] = []
    for edge in data:
        rows.append({
            "protein1": edge.get("stringId_A", ""),
            "protein2": edge.get("stringId_B", ""),
            "gene1": edge.get("preferredName_A", ""),
            "gene2": edge.get("preferredName_B", ""),
            "score": edge.get("score", 0),
        })

    df = pd.DataFrame(rows)
    _save_cache(cache_file, rows)
    print(f"  STRING PPI: {len(df)} interactions (score >= {score_threshold})")
    return df


def get_drug_targets(
    human_genes: list[str],
) -> pd.DataFrame:
    """Query ChEMBL for approved drugs targeting the given genes.

    Args:
        human_genes: Human gene symbols.

    Returns:
        DataFrame with columns ``gene``, ``drug_name``,
        ``mechanism_of_action``, ``max_phase``.
    """
    cache_key = ",".join(sorted(human_genes))
    cache_file = _cache_path("drug_targets", cache_key)
    cached = _load_cache(cache_file)
    if cached is not None:
        print(f"  Drug targets loaded from cache ({len(cached)} entries)")
        return pd.DataFrame(cached)

    from chembl_webresource_client.new_client import new_client  # noqa: PLC0415

    target_api = new_client.target
    mechanism_api = new_client.mechanism
    molecule_api = new_client.molecule

    rows: list[dict[str, Any]] = []

    for gene in human_genes:
        try:
            targets = target_api.filter(
                target_synonym__icontains=gene,
                target_type="SINGLE PROTEIN",
                organism="Homo sapiens",
            ).only(["target_chembl_id", "pref_name"])

            for tgt in targets[:3]:  # limit to top 3 targets per gene
                tgt_id = tgt["target_chembl_id"]
                mechs = mechanism_api.filter(target_chembl_id=tgt_id)

                for mech in mechs[:5]:  # limit mechanisms
                    mol_id = mech.get("molecule_chembl_id")
                    if not mol_id:
                        continue
                    mol = molecule_api.get(mol_id)
                    if mol is None:
                        continue
                    rows.append({
                        "gene": gene,
                        "drug_name": mol.get("pref_name", mol_id),
                        "mechanism_of_action": mech.get("mechanism_of_action", ""),
                        "max_phase": mol.get("max_phase", 0),
                    })
        except Exception:  # noqa: BLE001
            continue

    df = pd.DataFrame(rows).drop_duplicates(subset=["gene", "drug_name"])
    _save_cache(cache_file, df.to_dict(orient="records"))
    print(f"  Drug targets: {len(df)} gene-drug associations for {df['gene'].nunique() if len(df) else 0} genes")
    return df


def get_disease_associations(
    human_genes: list[str],
) -> pd.DataFrame:
    """Fetch gene-disease associations via mygene.

    Args:
        human_genes: Human gene symbols.

    Returns:
        DataFrame with columns ``gene``, ``disease_name``,
        ``disease_id``.
    """
    cache_key = ",".join(sorted(human_genes))
    cache_file = _cache_path("disease_assoc", cache_key)
    cached = _load_cache(cache_file)
    if cached is not None:
        print(f"  Disease associations loaded from cache ({len(cached)} entries)")
        return pd.DataFrame(cached)

    import mygene  # noqa: PLC0415
    mg = mygene.MyGeneInfo()

    results = mg.querymany(
        human_genes,
        scopes="symbol",
        fields="symbol,disease",
        species="human",
        returnall=False,
    )

    rows: list[dict[str, str]] = []
    for hit in results:
        if "notfound" in hit and hit["notfound"]:
            continue
        symbol = hit.get("symbol", hit.get("query", ""))
        diseases = hit.get("disease", [])
        if isinstance(diseases, dict):
            diseases = [diseases]
        for d in diseases:
            if isinstance(d, dict):
                rows.append({
                    "gene": symbol,
                    "disease_name": d.get("name", ""),
                    "disease_id": d.get("id", ""),
                })

    df = pd.DataFrame(rows)
    _save_cache(cache_file, df.to_dict(orient="records"))
    print(f"  Disease associations: {len(df)} for {df['gene'].nunique() if len(df) else 0} genes")
    return df
