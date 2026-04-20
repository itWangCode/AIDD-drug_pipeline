"""
Stage 1: Hub Genes → UniProt IDs → ChEMBL IDs → Bioactivity Data
Robust network calls with retry, caching, and checkpoint support.
"""

import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DIR, PROCESSED_DIR, CHEMBL_CONFIG, PIC50_ACTIVE_CUTOFF,
    HUB_GENES_FILE,
)
from src.utils import setup_logger, Checkpoint, retry, safe_save_csv, run_stage, chunked

logger = setup_logger("stage1_data_acquisition", "stage1.log")


# ─── Gene → UniProt ───────────────────────────────────────────────────────────
@retry(max_attempts=8, delay=3.0, backoff=2.0, exceptions=(requests.RequestException, Exception), logger=logger)
def gene_to_uniprot(gene_symbol: str, organism: str = "Homo sapiens") -> dict:
    """
    Query UniProt REST API to map gene symbol → UniProt accession.
    Returns the reviewed (Swiss-Prot) entry preferentially.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f'gene_exact:{gene_symbol} AND organism_name:"{organism}" AND reviewed:true',
        "format": "json",
        "fields": "accession,id,gene_names,protein_name,organism_name,length",
        "size": 5,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])

    if not results:
        # Try unreviewed
        params["query"] = f'gene_exact:{gene_symbol} AND organism_name:"{organism}"'
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

    if not results:
        logger.warning(f"No UniProt entry for gene: {gene_symbol}")
        return {}

    entry = results[0]
    accession = entry.get("primaryAccession", "")
    entry_name = entry.get("uniProtkbId", "")
    protein_name = ""
    try:
        protein_name = entry["proteinDescription"]["recommendedName"]["fullName"]["value"]
    except (KeyError, IndexError):
        pass

    return {
        "gene_symbol": gene_symbol,
        "uniprot_accession": accession,
        "uniprot_entry": entry_name,
        "protein_name": protein_name,
        "organism": organism,
    }


def load_hub_genes(filepath: Path) -> list:
    """Load hub genes from text file (one per line, comma- or newline-separated)."""
    if not filepath.exists():
        logger.error(f"Hub genes file not found: {filepath}")
        raise FileNotFoundError(f"{filepath}")
    text = filepath.read_text().strip()
    # Support comma, newline, space separation
    import re
    genes = [g.strip() for g in re.split(r"[,\n\r\t ]+", text) if g.strip()]
    logger.info(f"Loaded {len(genes)} hub genes: {genes}")
    return genes


def genes_to_uniprot_table(genes: list) -> pd.DataFrame:
    """Map all hub genes to UniProt. Robust, with per-gene retry."""
    records = []
    for gene in tqdm(genes, desc="Gene→UniProt"):
        try:
            rec = gene_to_uniprot(gene)
            if rec:
                records.append(rec)
                logger.info(f"  {gene} → {rec['uniprot_accession']} ({rec['protein_name'][:50]})")
            else:
                logger.warning(f"  {gene} → NOT FOUND")
                records.append({"gene_symbol": gene, "uniprot_accession": "", "uniprot_entry": "",
                                 "protein_name": "", "organism": "Homo sapiens"})
        except Exception as e:
            logger.error(f"  {gene} → ERROR: {e}")
            records.append({"gene_symbol": gene, "uniprot_accession": "", "uniprot_entry": "",
                             "protein_name": "", "organism": "Homo sapiens"})
        time.sleep(0.3)  # polite rate limiting

    df = pd.DataFrame(records)
    safe_save_csv(df, RAW_DIR / "uniprot_mapping.csv")
    logger.info(f"UniProt mapping: {len(df)} genes, {(df.uniprot_accession!='').sum()} mapped")
    return df


# ─── UniProt → ChEMBL Target ──────────────────────────────────────────────────
@retry(max_attempts=8, delay=3.0, backoff=2.0, exceptions=(Exception,), logger=logger)
def uniprot_to_chembl_targets(uniprot_id: str) -> list:
    """Query ChEMBL REST API for targets by UniProt accession."""
    url = "https://www.ebi.ac.uk/chembl/api/data/target"
    params = {
        "target_components__accession": uniprot_id,
        "format": "json",
        "limit": 50,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    targets = data.get("targets", [])
    # Prefer SINGLE PROTEIN
    single = [t for t in targets if t.get("target_type") == "SINGLE PROTEIN"]
    return single if single else targets


# ─── ChEMBL Bioactivity ───────────────────────────────────────────────────────
@retry(max_attempts=8, delay=5.0, backoff=2.0, exceptions=(Exception,), logger=logger)
def fetch_bioactivities_for_target(chembl_target_id: str, offset: int = 0, limit: int = 1000) -> dict:
    """Fetch IC50 bioactivities from ChEMBL REST API with pagination."""
    url = "https://www.ebi.ac.uk/chembl/api/data/activity"
    params = {
        "target_chembl_id": chembl_target_id,
        "type": "IC50",
        "relation": "=",
        "assay_type": "B",
        "standard_units": "nM",
        "format": "json",
        "limit": limit,
        "offset": offset,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_all_bioactivities(chembl_target_id: str) -> pd.DataFrame:
    """Fetch ALL bioactivities for a target using pagination."""
    logger.info(f"Fetching bioactivities for {chembl_target_id}...")
    all_activities = []
    offset = 0
    limit = 1000

    # Get total count first
    first_page = fetch_bioactivities_for_target(chembl_target_id, offset=0, limit=1)
    total = first_page.get("page_meta", {}).get("total_count", 0)
    logger.info(f"  Total activities: {total}")

    for offset in tqdm(range(0, total, limit), desc=f"  {chembl_target_id} bioactivities"):
        page = fetch_bioactivities_for_target(chembl_target_id, offset=offset, limit=limit)
        activities = page.get("activities", [])
        all_activities.extend(activities)
        time.sleep(0.5)

    if not all_activities:
        return pd.DataFrame()

    df = pd.DataFrame(all_activities)
    return df


# ─── Compound SMILES from ChEMBL ─────────────────────────────────────────────
@retry(max_attempts=8, delay=3.0, backoff=2.0, exceptions=(Exception,), logger=logger)
def fetch_compound_smiles_batch(chembl_ids: list) -> pd.DataFrame:
    """Fetch SMILES for a batch of ChEMBL molecule IDs."""
    ids_str = ",".join(chembl_ids)
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule"
    params = {
        "molecule_chembl_id__in": ids_str,
        "format": "json",
        "limit": len(chembl_ids),
        "only": "molecule_chembl_id,molecule_structures",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    records = []
    for mol in data.get("molecules", []):
        structs = mol.get("molecule_structures") or {}
        records.append({
            "molecule_chembl_id": mol["molecule_chembl_id"],
            "smiles": structs.get("canonical_smiles", ""),
        })
    return pd.DataFrame(records)


def fetch_all_smiles(chembl_ids: list, batch_size: int = 100) -> pd.DataFrame:
    """Fetch SMILES for all compounds in batches."""
    all_records = []
    batches = list(chunked(chembl_ids, batch_size))
    for batch in tqdm(batches, desc="Fetching SMILES"):
        try:
            df = fetch_compound_smiles_batch(batch)
            all_records.append(df)
        except Exception as e:
            logger.error(f"Batch SMILES fetch failed: {e}")
        time.sleep(0.5)
    if all_records:
        return pd.concat(all_records, ignore_index=True)
    return pd.DataFrame(columns=["molecule_chembl_id", "smiles"])


# ─── IC50 → pIC50 ─────────────────────────────────────────────────────────────
def ic50_to_pic50(ic50_nm: float) -> float:
    """Convert IC50 (nM) to pIC50."""
    return 9.0 - math.log10(ic50_nm)


# ─── Bioactivity Preprocessing ────────────────────────────────────────────────
def preprocess_bioactivities(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess raw bioactivity data."""
    if df.empty:
        return df

    required = ["molecule_chembl_id", "standard_value", "standard_units"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}")
        return pd.DataFrame()

    df = df.copy()
    # Keep only nM
    df = df[df["standard_units"] == "nM"]
    # Convert standard_value to float
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df.dropna(subset=["standard_value", "molecule_chembl_id"], inplace=True)
    df = df[df["standard_value"] > 0]
    # Rename
    df.rename(columns={"standard_value": "IC50_nM"}, inplace=True)
    # Add pIC50
    df["pIC50"] = df["IC50_nM"].apply(ic50_to_pic50)
    # Remove duplicates: keep best (highest pIC50) per molecule
    df.sort_values("pIC50", ascending=False, inplace=True)
    df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─── Main Stage 1 Function ────────────────────────────────────────────────────
def run_data_acquisition(force_rerun: bool = False) -> dict:
    """
    Full Stage 1: Hub genes → UniProt → ChEMBL → Bioactivities + SMILES.
    Returns dict: {gene: DataFrame with columns [molecule_chembl_id, IC50_nM, pIC50, smiles, ...]}
    """
    # Step 1a: Load genes
    genes = load_hub_genes(HUB_GENES_FILE)

    # Step 1b: Gene → UniProt
    ckpt_uniprot = Checkpoint("s1_uniprot")
    if ckpt_uniprot.exists() and not force_rerun:
        uniprot_df = ckpt_uniprot.load()
        logger.info("Loaded UniProt mapping from checkpoint")
    else:
        uniprot_df = genes_to_uniprot_table(genes)
        ckpt_uniprot.save(uniprot_df)

    # Filter genes with valid UniProt IDs
    valid = uniprot_df[uniprot_df["uniprot_accession"] != ""].copy()
    logger.info(f"Valid UniProt mappings: {len(valid)}/{len(uniprot_df)}")

    all_results = {}

    for _, row in valid.iterrows():
        gene = row["gene_symbol"]
        uniprot_id = row["uniprot_accession"]

        ckpt_gene = Checkpoint(f"s1_gene_{gene}")
        if ckpt_gene.exists() and not force_rerun:
            gene_df = ckpt_gene.load()
            logger.info(f"[{gene}] Loaded from checkpoint: {len(gene_df)} compounds")
            all_results[gene] = gene_df
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {gene} (UniProt: {uniprot_id})")

        try:
            # UniProt → ChEMBL target
            targets = uniprot_to_chembl_targets(uniprot_id)
            if not targets:
                logger.warning(f"[{gene}] No ChEMBL targets found")
                continue
            chembl_target_id = targets[0]["target_chembl_id"]
            logger.info(f"[{gene}] ChEMBL target: {chembl_target_id}")

            # Bioactivities
            bio_df = fetch_all_bioactivities(chembl_target_id)
            if bio_df.empty:
                logger.warning(f"[{gene}] No bioactivities found")
                continue
            bio_df = preprocess_bioactivities(bio_df)
            logger.info(f"[{gene}] Preprocessed bioactivities: {len(bio_df)}")

            # SMILES
            mol_ids = bio_df["molecule_chembl_id"].tolist()
            smiles_df = fetch_all_smiles(mol_ids)
            if smiles_df.empty:
                logger.warning(f"[{gene}] No SMILES fetched")
                continue

            # Merge
            gene_df = bio_df.merge(smiles_df, on="molecule_chembl_id", how="inner")
            gene_df = gene_df[gene_df["smiles"].notna() & (gene_df["smiles"] != "")]
            gene_df["gene_symbol"] = gene
            gene_df["uniprot_accession"] = uniprot_id
            gene_df["chembl_target_id"] = chembl_target_id
            gene_df["protein_name"] = row["protein_name"]

            safe_save_csv(gene_df, RAW_DIR / f"{gene}_bioactivities_raw.csv")
            ckpt_gene.save(gene_df)
            all_results[gene] = gene_df
            logger.info(f"[{gene}] Final compounds: {len(gene_df)}")

        except Exception as e:
            logger.error(f"[{gene}] Stage 1 failed: {e}")
            continue

    # Save combined
    if all_results:
        combined = pd.concat(all_results.values(), ignore_index=True)
        safe_save_csv(combined, RAW_DIR / "all_bioactivities_raw.csv")
        safe_save_csv(uniprot_df, RAW_DIR / "uniprot_mapping.csv")

    return all_results


if __name__ == "__main__":
    result = run_data_acquisition()
    for gene, df in result.items():
        print(f"{gene}: {len(df)} compounds")
