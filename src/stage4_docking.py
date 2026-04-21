"""
Stage 4 — Professional Molecular Docking Pipeline (Python 3.8 compatible)
==========================================================================
No meeko dependency. Pure RDKit + AutoDock Vina Python API.

Features:
  • Python 3.8+ fully compatible (no dict[str, Any] type hints)
  • One folder per PDB structure per gene
  • Protein: cleaned PDB → PDBQT (waters/HETATM removed, AD atom types assigned)
  • Ligand: SMILES → 3D conformer (ETKDGv3) → PDBQT (via RDKit only, no meeko)
  • 9 poses docked, split into pose_1.pdbqt ... pose_9.pdbqt
  • Best pose: {mol_id}_best_pose.pdbqt
  • All poses: {mol_id}_docked_all.pdbqt
  • Complex PDB: protein ATOM + ligand HETATM → {mol_id}_complex.pdb
  • SMILES + all 9 scores + RMSD → {mol_id}_docked_smiles.txt
  • Cartesian product: every PDB x every top compound
  • Top-6 auto-selected per PDB by best affinity
  • top6_best_compounds/ with rank1...rank6 complex PDB files
  • top6_summary.csv + top6_docked_smiles.txt
  • all_docking_results.csv per PDB
  • Nature-grade publication figures (dpi=300)

Install requirements:
  pip install vina          # AutoDock Vina Python API
  pip install rdkit         # already required by pipeline
  # Do NOT install meeko — not needed and Python 3.8 incompatible
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

try:
    from vina import Vina
    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DIR, RESULTS_DIR, FIGURES_DIR,
    TOP_N_COMPOUNDS, PLOT_CONFIG, NATURE_COLORS,
)
from src.utils import setup_logger, safe_save_csv, Checkpoint

logger = setup_logger("stage4_docking", "stage4_docking.log")

# ─── Pipeline constants ───────────────────────────────────────────────────────
N_POSES        = 9      # number of docking poses to generate
TOP_K          = 6      # top-affinity compounds to highlight / copy
EXHAUSTIVENESS = 8      # Vina exhaustiveness (8 = default; 16 for production)
BOX_PADDING    = 6.0    # Angstrom padding added to ligand extent for box
MIN_BOX_SIDE   = 15.0   # minimum box dimension in Angstrom


# ══════════════════════════════════════════════════════════════════════════════
# AutoDock4 atom-type tables
# ══════════════════════════════════════════════════════════════════════════════

# element (uppercase) -> (AD4_type, default_partial_charge)
_ELEMENT_TO_AD4 = {
    "C":  ("C",  0.000),
    "N":  ("N", -0.100),
    "O":  ("OA",-0.200),
    "S":  ("SA", 0.000),
    "H":  ("H",  0.100),
    "P":  ("P",  0.100),
    "F":  ("F", -0.100),
    "CL": ("Cl",-0.050),
    "BR": ("Br",-0.050),
    "I":  ("I", -0.050),
    "MG": ("Mg", 0.000),
    "ZN": ("Zn", 0.000),
    "FE": ("Fe", 0.000),
    "CA": ("Ca", 0.000),
    "MN": ("Mn", 0.000),
    "NA": ("NA",-0.100),
}

# RDKit atomic number -> AD4 type (for ligand PDBQT generation)
_ATOMNUM_TO_AD4 = {
    6: "C", 7: "N", 8: "OA", 16: "SA", 1: "H",
    15: "P", 9: "F", 17: "Cl", 35: "Br", 53: "I",
    12: "Mg", 30: "Zn", 26: "Fe", 20: "Ca", 11: "Na", 19: "K",
}

# Approximate partial charge by AD4 type (Gasteiger fallback)
_AD4_CHARGE = {
    "C": 0.000, "N": -0.100, "NA": -0.100,
    "OA": -0.200, "SA": -0.050, "S": -0.050,
    "H": 0.100, "HD": 0.200,
    "P": 0.100, "F": -0.100, "Cl": -0.050, "Br": -0.050, "I": -0.050,
}


def _elem_from_pdb_line(line: str) -> str:
    """Read element symbol from PDB/PDBQT ATOM line."""
    # cols 77-78 (0-indexed 76-78)
    if len(line) >= 78:
        elem = line[76:78].strip().upper()
        if elem:
            return elem
    # Fallback: first alpha characters of atom name (cols 13-16)
    aname = line[12:16].strip()
    return re.sub(r"[^A-Z]", "", aname.upper())[:2] or "C"


def _ad4_from_elem(elem: str) -> Tuple[str, float]:
    """Return (AD4_type, charge) for an element string."""
    return _ELEMENT_TO_AD4.get(elem.upper(), ("C", 0.000))


def _ad4_from_rdkit_atom(atom) -> Tuple[str, float]:
    """Return (AD4_type, charge) for an RDKit atom object."""
    atomic_num = atom.GetAtomicNum()
    ad4 = _ATOMNUM_TO_AD4.get(atomic_num, "C")

    # Refine N/O/S types
    if atomic_num == 7:
        # donor nitrogen has attached H
        ad4 = "NA" if atom.GetTotalNumHs() > 0 else "N"
    elif atomic_num == 8:
        ad4 = "OA"   # all O are H-bond acceptors in AD4
    elif atomic_num == 16:
        ad4 = "S" if atom.GetTotalNumHs() > 0 else "SA"

    # Partial charge: try Gasteiger, else lookup table
    try:
        charge = float(atom.GetDoubleProp("_GasteigerCharge"))
        if math.isnan(charge) or math.isinf(charge):
            charge = _AD4_CHARGE.get(ad4, 0.000)
    except (KeyError, ValueError):
        charge = _AD4_CHARGE.get(ad4, 0.000)

    return ad4, charge


# ══════════════════════════════════════════════════════════════════════════════
# Protein preparation
# ══════════════════════════════════════════════════════════════════════════════

def clean_protein_pdb(raw_pdb: Path, clean_pdb: Path) -> Path:
    """
    Clean a PDB file for docking:
      - Retain only ATOM records (remove HETATM, water, ions)
      - Remove explicit hydrogen atoms
      - Renumber atoms sequentially from 1
      - Append TER + END

    Parameters
    ----------
    raw_pdb : Path  Input PDB (may contain ligands, solvent, HETATM).
    clean_pdb : Path  Output cleaned PDB.

    Returns
    -------
    Path  Path to clean_pdb.
    """
    out = []
    idx = 1
    water_names = {"HOH", "WAT", "DOD", "H2O", "SOL"}

    with open(raw_pdb) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec != "ATOM":
                continue
            resname = line[17:20].strip()
            if resname in water_names:
                continue
            # Skip hydrogens
            aname = line[12:16].strip()
            elem_col = line[76:78].strip() if len(line) >= 78 else ""
            if aname.startswith("H") or elem_col.upper() == "H":
                continue
            out.append("ATOM  {:5d}{}\n".format(idx, line[11:66]).rstrip())
            idx += 1

    out += ["TER", "END"]
    clean_pdb.write_text("\n".join(out) + "\n")
    logger.info("Cleaned PDB: %s  (%d heavy atoms)", clean_pdb.name, idx - 1)
    return clean_pdb


def pdb_to_pdbqt_protein(pdb_path: Path, pdbqt_path: Path) -> Path:
    """
    Convert a cleaned protein PDB to AutoDock Vina PDBQT format.

    PDBQT format (80 cols):
      cols  1-66  : standard PDB ATOM record
      cols 67-70  : spaces
      cols 71-76  : partial charge  (%6.3f)
      cols 77-78  : AD4 atom type   (%-2s)

    IMPORTANT: Vina's receptor parser rejects HEADER / REMARK / TITLE lines.
    Write ONLY ATOM and TER / END lines.

    Parameters
    ----------
    pdb_path : Path   Cleaned protein PDB.
    pdbqt_path : Path  Output PDBQT.

    Returns
    -------
    Path  Path to pdbqt_path.
    """
    out = []
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                elem = _elem_from_pdb_line(line)
                ad4, charge = _ad4_from_elem(elem)
                base = line[:66].ljust(66)
                out.append("{base}    {charge:6.3f} {ad4:<2s}".format(
                    base=base, charge=charge, ad4=ad4))
            elif rec in ("TER", "END"):
                out.append(line.rstrip())

    out.append("END")
    pdbqt_path.write_text("\n".join(out) + "\n")
    logger.info("Protein PDBQT: %s", pdbqt_path.name)
    return pdbqt_path


# ══════════════════════════════════════════════════════════════════════════════
# Binding box
# ══════════════════════════════════════════════════════════════════════════════

def detect_hetatm_ligand(pdb_path: Path) -> Optional[str]:
    """
    Auto-detect the most atom-rich HETATM residue that is not a solvent / ion.
    Returns the 3-letter residue name, or None.
    """
    exclude = {
        "HOH","WAT","DOD","H2O","SOL",
        "EDO","GOL","PEG","PGE","DMS","MPD","ACT","ACE","FMT",
        "SO4","PO4","CL","NA","K","ZN","MG","CA","MN","FE",
        "MES","TRS","EPE","CSD","CSO","CME","MSE","SEP","PTR",
    }
    counts = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("HETATM"):
                rn = line[17:20].strip()
                if rn not in exclude:
                    counts[rn] = counts.get(rn, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else None


def compute_pocket_box(pdb_path: Path,
                        ligand_resname: Optional[str] = None) -> Dict:
    """
    Compute docking box from co-crystallized ligand HETATM coordinates,
    or from all ATOM heavy-atom coordinates as fallback.

    Returns dict: center_x/y/z, size_x/y/z, ligand_resname, n_atoms_used.
    """
    coords = []

    if ligand_resname:
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith("HETATM") and line[17:20].strip() == ligand_resname:
                    try:
                        coords.append([float(line[30:38]),
                                        float(line[38:46]),
                                        float(line[46:54])])
                    except ValueError:
                        pass

    if not coords:
        logger.warning("No HETATM for '%s' in %s — using ATOM centroid.",
                        ligand_resname, pdb_path.name)
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    try:
                        coords.append([float(line[30:38]),
                                        float(line[38:46]),
                                        float(line[46:54])])
                    except ValueError:
                        pass

    if not coords:
        logger.error("No coords in %s. Using origin box.", pdb_path.name)
        return {
            "center_x": 0.0, "center_y": 0.0, "center_z": 0.0,
            "size_x": 25.0,  "size_y": 25.0,  "size_z": 25.0,
            "ligand_resname": ligand_resname or "UNKNOWN",
            "n_atoms_used": 0,
        }

    arr = np.array(coords)
    center = arr.mean(axis=0)
    sizes  = np.maximum(arr.max(axis=0) - arr.min(axis=0) + BOX_PADDING * 2.0,
                         MIN_BOX_SIDE)
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x":   float(sizes[0]),
        "size_y":   float(sizes[1]),
        "size_z":   float(sizes[2]),
        "ligand_resname": ligand_resname or "ATOM_CENTROID",
        "n_atoms_used": len(coords),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Ligand preparation  (pure RDKit — no meeko, Python 3.8 safe)
# ══════════════════════════════════════════════════════════════════════════════

def _mol_to_pdbqt_string(mol_h) -> Optional[str]:
    """
    Convert an RDKit mol (with 3D conformer + Hs) to PDBQT-format string.
    Uses Gasteiger charges. No meeko needed. Python 3.8 compatible.
    """
    try:
        AllChem.ComputeGasteigerCharges(mol_h)
        conf = mol_h.GetConformer()
        lines = [
            "REMARK  PDBQT generated by Drug Discovery Pipeline (RDKit)",
            "ROOT",
        ]
        for i, atom in enumerate(mol_h.GetAtoms()):
            pos  = conf.GetAtomPosition(i)
            ad4, charge = _ad4_from_rdkit_atom(atom)
            elem = atom.GetSymbol()
            # Build atom name: element + serial (max 4 chars)
            aname = "{}{:d}".format(elem[:1], i + 1)
            aname = "{:<4s}".format(aname[:4])
            line = (
                "ATOM  {idx:5d} {aname:<4s} LIG A   1    "
                "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    "
                "{charge:6.3f} {ad4:<2s}"
            ).format(
                idx=i + 1, aname=aname,
                x=pos.x, y=pos.y, z=pos.z,
                charge=charge, ad4=ad4,
            )
            lines.append(line)
        lines += ["ENDROOT", "TORSDOF 0", ""]
        return "\n".join(lines)
    except Exception as e:
        logger.error("_mol_to_pdbqt_string failed: %s", e)
        return None


def smiles_to_pdbqt(smiles: str, mol_id: str,
                     out_dir: Path) -> Tuple[Optional[Path], Optional[object]]:
    """
    Generate a 3D conformer from SMILES and write a PDBQT file.
    No meeko. Pure RDKit. Python 3.8 compatible.

    Strategy (in order):
      1. ETKDGv3 (best, RDKit >= 2020)
      2. ETKDG   (fallback)
      3. ETKDGv2 (last resort)
    Then MMFF94s geometry optimisation.

    Returns (pdbqt_path, rdkit_mol) or (None, None).
    """
    pdbqt_path = out_dir / "{}_ligand.pdbqt".format(mol_id)

    # Return cached
    if pdbqt_path.exists() and pdbqt_path.stat().st_size > 50:
        mol = Chem.MolFromSmiles(smiles)
        return pdbqt_path, mol

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("[%s] Invalid SMILES: %.60s", mol_id, smiles)
            return None, None

        mol_h = Chem.AddHs(mol)

        # Attempt 1: ETKDGv3
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        p.numThreads = 0
        ret = AllChem.EmbedMolecule(mol_h, p)

        # Attempt 2: ETKDG
        if ret == -1:
            p2 = AllChem.ETKDG()
            p2.randomSeed = 42
            ret = AllChem.EmbedMolecule(mol_h, p2)

        # Attempt 3: ETKDGv2
        if ret == -1:
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv2())

        if mol_h.GetNumConformers() == 0:
            logger.warning("[%s] 3D embedding failed for: %.60s", mol_id, smiles)
            return None, None

        # Geometry optimisation
        try:
            ff_res = AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94s",
                                                    maxIters=2000)
            if ff_res == -1:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=2000)
        except Exception:
            pass  # 3D coords exist even if optimisation failed

        pdbqt_str = _mol_to_pdbqt_string(mol_h)
        if pdbqt_str is None:
            return None, None

        pdbqt_path.write_text(pdbqt_str)
        return pdbqt_path, mol

    except Exception as e:
        logger.error("[%s] smiles_to_pdbqt: %s", mol_id, e)
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Pose utilities
# ══════════════════════════════════════════════════════════════════════════════

def split_pdbqt_poses(all_pdbqt: Path, out_dir: Path, mol_id: str) -> int:
    """
    Split a multi-MODEL PDBQT file into individual files:
      {mol_id}_pose_1.pdbqt ... {mol_id}_pose_N.pdbqt

    Returns number of pose files written.
    """
    text   = all_pdbqt.read_text()
    models = re.split(r"(?=^MODEL\s)", text, flags=re.MULTILINE)
    models = [m.strip() for m in models if m.strip()]

    for i, block in enumerate(models, 1):
        if not block.startswith("MODEL"):
            block = "MODEL     {}\n{}".format(i, block)
        if "ENDMDL" not in block:
            block += "\nENDMDL"
        (out_dir / "{}_pose_{}.pdbqt".format(mol_id, i)).write_text(block + "\n")

    return len(models)


def write_complex_pdb(receptor_pdbqt: Path, ligand_pdbqt: Path,
                       out_pdb: Path, mol_id: str, pdb_id: str) -> Path:
    """
    Merge protein (ATOM) and best-pose ligand (HETATM) into a standard PDB file.
    Chain A = protein, Chain L = ligand (resname LIG).

    Strict PDB column layout (80 cols):
      1-6   record type
      7-11  serial
      12    space
      13-16 atom name (left-aligned for C/N/O, right-padded)
      17    alt loc (space)
      18-20 residue name
      21    space
      22    chain ID
      23-26 residue seq number
      27    insertion code (space)
      28-30 spaces
      31-38 x (8.3f)
      39-46 y (8.3f)
      47-54 z (8.3f)
      55-60 occupancy (6.2f)
      61-66 b-factor (6.2f)
      67-76 spaces
      77-78 element symbol (right-justified)
      79-80 charge (spaces)

    Viewable in PyMOL, UCSF ChimeraX, VMD, and parsable by PLIP/RDKit.
    """
    # AD4 atom type -> proper PDB element symbol
    _AD4_TO_ELEM = {
        "C": "C",  "A": "C",    # carbon (aromatic = C)
        "N": "N",  "NA": "N",   # nitrogen
        "OA": "O", "OS": "O",   # oxygen
        "SA": "S", "S": "S",    # sulfur
        "H": "H",  "HD": "H",   # hydrogen
        "P": "P",
        "F": "F",
        "Cl": "Cl", "CL": "Cl",
        "Br": "Br", "BR": "Br",
        "I": "I",
        "Mg": "Mg", "MG": "Mg",
        "Zn": "Zn", "ZN": "Zn",
        "Fe": "Fe", "FE": "Fe",
        "Ca": "Ca", "CA": "Ca",
        "Mn": "Mn", "MN": "Mn",
    }

    def _ad4_to_pdb_elem(ad4_str):
        """Convert AD4 type to proper 2-char PDB element, right-justified."""
        elem = _AD4_TO_ELEM.get(ad4_str.strip(), "C")
        return "{:>2s}".format(elem)

    def _format_atom_name(raw_name, elem):
        """
        Format atom name to PDB cols 13-16.
        Single-char elements: space + name padded to 3 chars (e.g. ' CA ')
        Two-char elements:    name left-padded to 4 chars  (e.g. 'FE  ')
        """
        name = raw_name.strip()
        if len(elem.strip()) == 1:
            return " {:<3s}".format(name[:3])
        else:
            return "{:<4s}".format(name[:4])

    def _pdb_atom_line(record, serial, aname_raw, resname, chain, resi, x, y, z,
                        occ, bfac, elem_raw):
        """Build one 80-column PDB ATOM/HETATM line."""
        elem = _ad4_to_pdb_elem(elem_raw)
        aname = _format_atom_name(aname_raw, _AD4_TO_ELEM.get(elem_raw.strip(), "C"))
        return (
            "{rec:<6s}{serial:5d} {aname:<4s}{alt:1s}{resn:<3s} {chain:1s}"
            "{resi:4d}{icode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
            "{occ:6.2f}{bfac:6.2f}          {elem:>2s}  "
        ).format(
            rec=record, serial=serial, aname=aname.strip(), alt=" ",
            resn=resname, chain=chain, resi=resi, icode=" ",
            x=x, y=y, z=z, occ=occ, bfac=bfac, elem=elem.strip(),
        )

    lines = [
        "REMARK   1 PROTEIN-LIGAND DOCKING COMPLEX",
        "REMARK   2 PDB template : {}".format(pdb_id),
        "REMARK   3 Ligand       : {}".format(mol_id),
        "REMARK   4 Software     : AutoDock Vina",
        "REMARK   5 Chain A = Protein | Chain L = Ligand (resname LIG)",
        "REMARK   6 Generated by Computational Drug Discovery Pipeline",
    ]
    idx = 1

    # ── Protein ATOM records from receptor PDBQT ────────────────────────────
    for line in receptor_pdbqt.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        try:
            aname_raw = line[12:16]
            resname   = line[17:20].strip()
            chain     = line[21].strip() or "A"
            resi      = int(line[22:26])
            x         = float(line[30:38])
            y         = float(line[38:46])
            z         = float(line[46:54])
            ad4       = line[77:79].strip() if len(line) >= 79 else "C"
            pdb_line  = _pdb_atom_line("ATOM", idx, aname_raw, resname,
                                        chain, resi, x, y, z, 1.0, 20.0, ad4)
            lines.append(pdb_line)
            idx += 1
        except (ValueError, IndexError):
            continue

    lines.append("TER   {:5d}".format(idx))
    idx += 1

    # ── Ligand HETATM records from best-pose PDBQT ──────────────────────────
    lig_serial = 1
    for line in ligand_pdbqt.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        try:
            aname_raw = line[12:16]
            x         = float(line[30:38])
            y         = float(line[38:46])
            z         = float(line[46:54])
            ad4       = line[77:79].strip() if len(line) >= 79 else "C"
            pdb_line  = _pdb_atom_line("HETATM", idx, aname_raw, "LIG",
                                        "L", lig_serial, x, y, z, 1.0, 20.0, ad4)
            lines.append(pdb_line)
            idx += 1
            lig_serial += 1
        except (ValueError, IndexError):
            continue

    lines.append("END")
    out_pdb.write_text("\n".join(lines) + "\n")
    return out_pdb


def write_docked_smiles_txt(out_file: Path, smiles: str, mol_id: str,
                              scores: List[float], energies: List[List[float]],
                              box: Dict, pdb_id: str) -> Path:
    """
    Write {mol_id}_docked_smiles.txt containing:
      - SMILES string
      - Binding pocket box parameters and reference ligand
      - Table of all N poses: affinity (kcal/mol), RMSD lb, RMSD ub
      - List of generated files

    Returns out_file.
    """
    sep = "=" * 72
    rows = [
        sep,
        "  DOCKING RESULTS",
        "  Compound  : {}".format(mol_id),
        "  PDB target: {}".format(pdb_id),
        sep,
        "",
        "SMILES:",
        "  {}".format(smiles),
        "",
        "Binding Pocket:",
        "  Reference ligand / pocket : {}".format(box.get("ligand_resname", "N/A")),
        "  Box center  : ({:.3f}, {:.3f}, {:.3f}) Angstrom".format(
            box["center_x"], box["center_y"], box["center_z"]),
        "  Box size    : {:.1f} x {:.1f} x {:.1f} Angstrom".format(
            box["size_x"], box["size_y"], box["size_z"]),
        "  Atoms used  : {}".format(box.get("n_atoms_used", "N/A")),
        "",
        "{:<6}  {:<22}  {:<14}  {:<14}".format(
            "Pose", "Affinity (kcal/mol)", "RMSD lb (A)", "RMSD ub (A)"),
        "-" * 60,
    ]
    for i, e in enumerate(energies, 1):
        aff = "{:.3f}".format(e[0]) if len(e) > 0 else "N/A"
        lb  = "{:.3f}".format(e[1]) if len(e) > 1 else "N/A"
        ub  = "{:.3f}".format(e[2]) if len(e) > 2 else "N/A"
        rows.append("{:<6d}  {:<22}  {:<14}  {:<14}".format(i, aff, lb, ub))

    best = "{:.3f} kcal/mol".format(scores[0]) if scores else "N/A"
    rows += [
        "",
        "Best binding affinity (Pose 1) : {}".format(best),
        "",
        "Files in this directory:",
        "  {m}_ligand.pdbqt         — prepared ligand PDBQT".format(m=mol_id),
        "  {m}_docked_all.pdbqt     — all {n} poses (multi-MODEL)".format(m=mol_id, n=len(scores)),
        "  {m}_pose_1.pdbqt         — best pose".format(m=mol_id),
        "  {m}_pose_{n}.pdbqt       — worst pose".format(m=mol_id, n=len(scores)),
        "  {m}_best_pose.pdbqt      — best pose (single MODEL)".format(m=mol_id),
        "  {m}_complex.pdb          — protein + best pose (PyMOL ready)".format(m=mol_id),
        sep,
    ]
    out_file.write_text("\n".join(rows) + "\n")
    return out_file


# ══════════════════════════════════════════════════════════════════════════════
# Result record builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_record(mol_id, smiles, pdb_id, scores, energies,
                   all_pdbqt, best_pdbqt, complex_pdb, box, status, error=""):
    """Build a flat dict for one compound x one PDB docking result."""
    return {
        "molecule_id":         mol_id,
        "smiles":              smiles,
        "pdb_id":              pdb_id,
        "status":              status,
        "error":               error,
        "best_affinity_kcal":  scores[0] if scores else None,
        "affinity_pose2_kcal": scores[1] if len(scores) > 1 else None,
        "affinity_pose3_kcal": scores[2] if len(scores) > 2 else None,
        "affinity_pose4_kcal": scores[3] if len(scores) > 3 else None,
        "affinity_pose5_kcal": scores[4] if len(scores) > 4 else None,
        "all_scores":          scores,
        "n_poses":             len(scores),
        "pocket_ligand":       box.get("ligand_resname", ""),
        "pocket_center":       "{:.2f},{:.2f},{:.2f}".format(
                                   box["center_x"], box["center_y"], box["center_z"]),
        "box_size":            "{:.1f}x{:.1f}x{:.1f}".format(
                                   box["size_x"], box["size_y"], box["size_z"]),
        "all_poses_file":      str(all_pdbqt)   if all_pdbqt   else "",
        "best_pose_file":      str(best_pdbqt)  if best_pdbqt  else "",
        "complex_pdb":         str(complex_pdb) if complex_pdb else "",
    }


def _parse_vina_scores(pdbqt_path: Path) -> List[float]:
    """Parse affinity scores from REMARK VINA RESULT lines."""
    scores = []
    for line in pdbqt_path.read_text().splitlines():
        if "VINA RESULT" in line:
            parts = line.split()
            try:
                scores.append(float(parts[3]))
            except (IndexError, ValueError):
                pass
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Core docking function
# ══════════════════════════════════════════════════════════════════════════════

def run_single_docking(receptor_pdbqt: Path, ligand_pdbqt: Path,
                        box: Dict, mol_dir: Path, mol_id: str,
                        smiles: str, pdb_id: str,
                        n_poses: int = N_POSES,
                        exhaustiveness: int = EXHAUSTIVENESS) -> Dict:
    """
    Dock one ligand vs one receptor with AutoDock Vina Python API.

    Writes to mol_dir/:
      {mol_id}_docked_all.pdbqt     all n_poses
      {mol_id}_pose_1.pdbqt         best pose
      ...
      {mol_id}_pose_{n}.pdbqt       worst pose
      {mol_id}_best_pose.pdbqt      best pose (single MODEL)
      {mol_id}_complex.pdb          protein + best pose PDB
      {mol_id}_docked_smiles.txt    SMILES + all scores + box info

    Returns a flat dict with docking result fields.
    """
    all_pdbqt   = mol_dir / "{}_docked_all.pdbqt".format(mol_id)
    best_pdbqt  = mol_dir / "{}_best_pose.pdbqt".format(mol_id)
    smiles_txt  = mol_dir / "{}_docked_smiles.txt".format(mol_id)
    complex_pdb = mol_dir / "{}_complex.pdb".format(mol_id)

    # Return cached result if already done
    if all_pdbqt.exists() and smiles_txt.exists():
        cached = _parse_vina_scores(all_pdbqt)
        logger.debug("[%s] Using cached docking result", mol_id)
        return _build_record(mol_id, smiles, pdb_id, cached,
                              [[s, 0.0, 0.0] for s in cached],
                              all_pdbqt, best_pdbqt, complex_pdb, box, "cached")

    if not VINA_AVAILABLE:
        msg = "vina package not installed. Run: pip install vina"
        logger.error(msg)
        return _build_record(mol_id, smiles, pdb_id, [], [],
                              None, None, None, box, "skipped", msg)

    try:
        v = Vina(sf_name="vina", verbosity=0, seed=42)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(str(ligand_pdbqt))
        v.compute_vina_maps(
            center=[box["center_x"], box["center_y"], box["center_z"]],
            box_size=[box["size_x"], box["size_y"], box["size_z"]],
        )
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

        energies = v.energies(n_poses=n_poses)   # [[aff, rmsd_lb, rmsd_ub], ...]
        scores   = [float(e[0]) for e in energies]

        # Write outputs
        v.write_poses(str(all_pdbqt),  n_poses=n_poses, overwrite=True)
        v.write_poses(str(best_pdbqt), n_poses=1,       overwrite=True)

        split_pdbqt_poses(all_pdbqt, mol_dir, mol_id)
        write_complex_pdb(receptor_pdbqt, best_pdbqt, complex_pdb, mol_id, pdb_id)
        write_docked_smiles_txt(smiles_txt, smiles, mol_id, scores, energies, box, pdb_id)

        logger.info("  [%s] Best: %.3f kcal/mol | all: %s",
                    mol_id, scores[0],
                    " ".join(["{:.2f}".format(s) for s in scores]))

        return _build_record(mol_id, smiles, pdb_id, scores, energies,
                              all_pdbqt, best_pdbqt, complex_pdb, box, "success")

    except Exception as e:
        logger.error("  [%s] Docking FAILED: %s", mol_id, e)
        return _build_record(mol_id, smiles, pdb_id, [], [],
                              None, None, None, box, "failed", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Cartesian-product orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_full_docking_pipeline(
    pdb_configs: List[Dict],
    compounds_df: pd.DataFrame,
    gene: str,
    base_out_dir: Path,
    top_n: int = TOP_N_COMPOUNDS,
    n_poses: int = N_POSES,
    exhaustiveness: int = EXHAUSTIVENESS,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """
    Cartesian-product docking: every PDB x every top compound.

    Output layout:
      base_out_dir/
        {pdb_id}/
          protein_clean.pdb
          protein.pdbqt
          box_info.json
          {mol_id}/
            {mol_id}_ligand.pdbqt
            {mol_id}_docked_all.pdbqt
            {mol_id}_pose_1.pdbqt
            ...
            {mol_id}_pose_9.pdbqt
            {mol_id}_best_pose.pdbqt
            {mol_id}_complex.pdb
            {mol_id}_docked_smiles.txt
          all_docking_results.csv
          top6_summary.csv
          top6_docked_smiles.txt
          top6_best_compounds/
            rank1_{mol_id}_complex.pdb
            ...

    Parameters
    ----------
    pdb_configs : list of dict
        Each: {"pdb_id": str, "pdb_path": Path or str, "ligand_resname": str or None}
    compounds_df : pd.DataFrame
        Must have columns: molecule_chembl_id, smiles, [pIC50, qed, composite_score]
    gene : str
    base_out_dir : Path
    top_n : int  Number of top compounds to dock.
    n_poses : int  Poses per compound.
    exhaustiveness : int  Vina exhaustiveness.
    force_rerun : bool

    Returns
    -------
    pd.DataFrame  All docking results.
    """
    base_out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Select top-N by composite_score or pIC50
    sort_col = "composite_score" if "composite_score" in compounds_df.columns else "pIC50"
    top_cpds = (
        compounds_df.dropna(subset=["smiles"])
        .nlargest(min(top_n, len(compounds_df)), sort_col)
        .reset_index(drop=True)
    )

    logger.info("[%s] Docking %d compounds x %d PDB structures",
                gene, len(top_cpds), len(pdb_configs))

    for pdb_cfg in pdb_configs:
        pdb_id   = pdb_cfg["pdb_id"]
        pdb_path = Path(pdb_cfg["pdb_path"])
        lig_rn   = pdb_cfg.get("ligand_resname", None)

        pdb_out = base_out_dir / pdb_id
        pdb_out.mkdir(exist_ok=True)

        logger.info("\n%s", "=" * 60)
        logger.info("[%s] PDB: %s  |  Reference ligand: %s",
                    gene, pdb_id, lig_rn or "AUTO-DETECT")

        # ── Protein preparation ──────────────────────────────────────────────
        clean_pdb  = pdb_out / "protein_clean.pdb"
        prot_pdbqt = pdb_out / "protein.pdbqt"
        box_json   = pdb_out / "box_info.json"

        if not prot_pdbqt.exists() or force_rerun:
            try:
                clean_protein_pdb(pdb_path, clean_pdb)
                pdb_to_pdbqt_protein(clean_pdb, prot_pdbqt)
            except Exception as e:
                logger.error("[%s/%s] Protein prep failed: %s", gene, pdb_id, e)
                continue

        # ── Binding box ──────────────────────────────────────────────────────
        if box_json.exists() and not force_rerun:
            with open(box_json) as fj:
                box = json.load(fj)
        else:
            detected = lig_rn or detect_hetatm_ligand(pdb_path)
            box = compute_pocket_box(pdb_path, detected)
            with open(box_json, "w") as fj:
                json.dump(box, fj, indent=2)

        logger.info(
            "  Pocket centre: (%.1f, %.1f, %.1f) | box: %.1f x %.1f x %.1f A | ref: %s",
            box["center_x"], box["center_y"], box["center_z"],
            box["size_x"],   box["size_y"],   box["size_z"],
            box.get("ligand_resname", "?")
        )

        # ── Dock each compound ───────────────────────────────────────────────
        pdb_results = []

        for _, row in tqdm(
            top_cpds.iterrows(), total=len(top_cpds),
            desc="  [{}/{}]".format(gene, pdb_id),
        ):
            mol_id = str(row.get("molecule_chembl_id",
                                  "mol_{}".format(row.name)))
            smiles = str(row["smiles"])

            mol_dir = pdb_out / mol_id
            mol_dir.mkdir(exist_ok=True)

            # Check cache
            all_pdbqt_cached = mol_dir / "{}_docked_all.pdbqt".format(mol_id)
            if all_pdbqt_cached.exists() and not force_rerun:
                cached_scores = _parse_vina_scores(all_pdbqt_cached)
                rec = _build_record(
                    mol_id, smiles, pdb_id, cached_scores,
                    [[s, 0.0, 0.0] for s in cached_scores],
                    all_pdbqt_cached,
                    mol_dir / "{}_best_pose.pdbqt".format(mol_id),
                    mol_dir / "{}_complex.pdb".format(mol_id),
                    box, "cached"
                )
            else:
                lig_pdbqt, _ = smiles_to_pdbqt(smiles, mol_id, mol_dir)
                if lig_pdbqt is None:
                    logger.warning("  Skipping %s: ligand prep failed", mol_id)
                    continue
                rec = run_single_docking(
                    prot_pdbqt, lig_pdbqt, box, mol_dir,
                    mol_id, smiles, pdb_id, n_poses, exhaustiveness
                )

            rec["gene"]             = gene
            rec["pIC50"]            = row.get("pIC50")
            rec["qed"]              = row.get("qed")
            rec["mw"]               = row.get("mw")
            rec["logp"]             = row.get("logp")
            rec["composite_score"]  = row.get("composite_score")
            rec["pred_active_prob"] = row.get("pred_active_prob")

            pdb_results.append(rec)
            all_results.append(rec)

        # ── Per-PDB CSV ──────────────────────────────────────────────────────
        df_pdb = pd.DataFrame(pdb_results)
        if df_pdb.empty:
            logger.warning("[%s/%s] No docking results", gene, pdb_id)
            continue
        df_pdb["best_affinity_kcal"] = pd.to_numeric(
            df_pdb["best_affinity_kcal"], errors="coerce")
        safe_save_csv(df_pdb, pdb_out / "all_docking_results.csv")

        # ── Top-6 selection ──────────────────────────────────────────────────
        df_valid = df_pdb[df_pdb["best_affinity_kcal"].notna()]
        if df_valid.empty:
            continue

        top6 = df_valid.nsmallest(TOP_K, "best_affinity_kcal")
        top6_dir = pdb_out / "top6_best_compounds"
        top6_dir.mkdir(exist_ok=True)

        logger.info("\n  %s", "+" * 56)
        logger.info("  TOP-%d  BEST AFFINITY  |  %s x %s", TOP_K, gene, pdb_id)
        logger.info("  %s", "+" * 56)

        top6_rows = []
        for rank, (_, r) in enumerate(top6.iterrows(), 1):
            aff_s  = "{:.3f}".format(r["best_affinity_kcal"])
            pic_s  = "{:.2f}".format(r["pIC50"]) if pd.notna(r.get("pIC50")) else "N/A"
            qed_s  = "{:.3f}".format(r["qed"])   if pd.notna(r.get("qed"))   else "N/A"
            logger.info("  #%d  %-22s  %s kcal/mol  pIC50=%s  QED=%s",
                         rank, str(r["molecule_id"])[:22], aff_s, pic_s, qed_s)
            src = Path(r["complex_pdb"]) if r.get("complex_pdb") else None
            if src and src.exists():
                dst = top6_dir / "rank{}_{}_complex.pdb".format(
                    rank, r["molecule_id"])
                shutil.copy(src, dst)
            top6_rows.append(r.to_dict())

        # Save top-6 files
        df_top6 = pd.DataFrame(top6_rows)
        safe_save_csv(df_top6, pdb_out / "top6_summary.csv")

        with open(pdb_out / "top6_docked_smiles.txt", "w") as f:
            f.write("TOP-{} BEST-AFFINITY COMPOUNDS\n".format(TOP_K))
            f.write("Gene: {}  |  PDB: {}\n".format(gene, pdb_id))
            f.write("=" * 72 + "\n")
            f.write("{:<5}  {:<12}  {:<8}  {:<8}  SMILES\n".format(
                "Rank", "Affinity", "pIC50", "QED"))
            f.write("-" * 72 + "\n")
            for rank, r in enumerate(top6_rows, 1):
                aff = "{:.3f}".format(r["best_affinity_kcal"]) \
                      if r.get("best_affinity_kcal") is not None else "N/A"
                pic = "{:.2f}".format(r["pIC50"]) if pd.notna(r.get("pIC50")) else "N/A"
                qed = "{:.3f}".format(r["qed"])   if pd.notna(r.get("qed"))   else "N/A"
                f.write("#{:<4d}  {:<12}  {:<8}  {:<8}  {}\n".format(
                    rank, aff, pic, qed, str(r["smiles"])[:80]))
            f.write("=" * 72 + "\n")

    # ── Global CSV ───────────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_results)
    if not df_all.empty:
        df_all["best_affinity_kcal"] = pd.to_numeric(
            df_all["best_affinity_kcal"], errors="coerce")
        out_csv = RESULTS_DIR / "{}_docking_all.csv".format(gene)
        safe_save_csv(df_all, out_csv)
        logger.info("[%s] Global docking results: %s", gene, out_csv)
    return df_all


# ══════════════════════════════════════════════════════════════════════════════
# Publication-quality figures
# ══════════════════════════════════════════════════════════════════════════════

def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.3, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linestyle": "--",
        "figure.dpi": 300, "savefig.dpi": 300,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), bbox_inches="tight", dpi=300)
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved: %s", path.name)


def plot_docking_panel(df_in: pd.DataFrame, gene: str, pdb_id: str) -> None:
    """
    6-panel docking results figure for one gene x one PDB.
    A: affinity ranking bar (gold border = top-6)
    B: affinity category pie
    C: affinity vs pIC50 scatter (colour = QED, star = top-6)
    D: 9-pose violin (top-8 compounds)
    E: top-6 horizontal bars with pIC50 + QED labels
    F: QED vs affinity scatter (colour = pIC50)
    """
    _style()
    df = df_in.copy()
    df["best_affinity_kcal"] = pd.to_numeric(df["best_affinity_kcal"], errors="coerce")
    df["pIC50"] = pd.to_numeric(df.get("pIC50", pd.Series(dtype=float)), errors="coerce")
    df["qed"]   = pd.to_numeric(df.get("qed",   pd.Series(dtype=float)), errors="coerce")
    df = df[df["best_affinity_kcal"].notna()].copy()
    if df.empty:
        return

    df_s  = df.nsmallest(len(df), "best_affinity_kcal").reset_index(drop=True)
    top6  = df.nsmallest(TOP_K,   "best_affinity_kcal").reset_index(drop=True)
    fig   = plt.figure(figsize=(24, 18))
    gs    = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.42)
    n     = len(df_s)

    # A ── ranking bar
    ax_a = fig.add_subplot(gs[0, :2])
    cb   = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n))
    bars = ax_a.bar(range(n), df_s["best_affinity_kcal"].values,
                    color=cb, edgecolor="white", linewidth=0.8, zorder=2)
    for i in range(min(TOP_K, n)):
        bars[i].set_edgecolor("gold")
        bars[i].set_linewidth(3.0)
    ax_a.set_xticks(range(n))
    ax_a.set_xticklabels([str(df_s.loc[i, "molecule_id"])[:11] for i in range(n)],
                          rotation=50, ha="right", fontsize=8.5)
    ax_a.set_ylabel("Binding Affinity (kcal/mol)", fontsize=13, fontweight="bold")
    cutoff = top6["best_affinity_kcal"].max()
    ax_a.axhline(y=cutoff, color="gold", lw=2.2, ls="--", alpha=0.85,
                  label="Top-{} cutoff ({:.2f} kcal/mol)".format(TOP_K, cutoff))
    ax_a.set_title(
        "A   AutoDock Vina Ranking — {}  |  {}\n"
        "Gold border = Top-{}  (lower = stronger binding)".format(gene, pdb_id, TOP_K),
        fontsize=13, fontweight="bold", loc="left")
    ax_a.legend(fontsize=11)
    for i in range(min(3, n)):
        v = df_s.loc[i, "best_affinity_kcal"]
        ax_a.text(i, v - 0.04, "#{}\n{:.2f}".format(i + 1, v),
                  ha="center", va="top", fontsize=8.5, fontweight="bold", color="white")

    # B ── pie
    ax_b = fig.add_subplot(gs[0, 2])
    bin_defs = [(-999, -9, "< -9 excellent"), (-9, -7, "-9 to -7 strong"),
                (-7, -5, "-7 to -5 moderate"), (-5, 0, "> -5 weak")]
    cnts = []; lbls = []
    for lo, hi, lbl in bin_defs:
        nb = ((df["best_affinity_kcal"] >= lo) & (df["best_affinity_kcal"] < hi)).sum()
        if nb > 0:
            cnts.append(nb)
            lbls.append("{}\n(n={})".format(lbl, nb))
    if cnts:
        ax_b.pie(cnts, labels=lbls,
                  colors=[NATURE_COLORS[i % len(NATURE_COLORS)] for i in range(len(cnts))],
                  autopct="%1.0f%%", startangle=90,
                  textprops={"fontsize": 10},
                  wedgeprops={"linewidth": 2, "edgecolor": "white"})
    ax_b.set_title("B   Affinity\nCategory", fontsize=13, fontweight="bold", loc="left")

    # C ── affinity vs pIC50
    ax_c = fig.add_subplot(gs[1, 0])
    dv = df[df["pIC50"].notna() & df["qed"].notna()]
    if not dv.empty:
        sc = ax_c.scatter(dv["best_affinity_kcal"], dv["pIC50"],
                           s=80, alpha=0.75, c=dv["qed"], cmap="RdYlGn",
                           vmin=0, vmax=1, edgecolors="white", linewidths=0.5)
        plt.colorbar(sc, ax=ax_c, label="QED", shrink=0.85, pad=0.02)
        if not top6.empty:
            valid_t6 = top6[top6["pIC50"].notna()]
            ax_c.scatter(valid_t6["best_affinity_kcal"], valid_t6["pIC50"],
                          s=220, marker="*", c="gold", edgecolors="darkorange",
                          linewidths=1.5, zorder=5,
                          label="Top-{}".format(TOP_K))
            for rank, (_, r) in enumerate(valid_t6.iterrows(), 1):
                ax_c.annotate("#{}".format(rank),
                               xy=(r["best_affinity_kcal"], r["pIC50"]),
                               fontsize=8.5, fontweight="bold", color="darkorange",
                               xytext=(4, 4), textcoords="offset points")
    ax_c.axhline(y=6.3, color="crimson", lw=1.8, ls="--", alpha=0.7,
                  label="Active (pIC50>=6.3)")
    ax_c.set_xlabel("Best Affinity (kcal/mol)", fontsize=13, fontweight="bold")
    ax_c.set_ylabel("Measured pIC50", fontsize=13, fontweight="bold")
    ax_c.set_title("C   Affinity vs pIC50\nStar=Top-{}, Colour=QED".format(TOP_K),
                    fontsize=13, fontweight="bold", loc="left")
    ax_c.legend(fontsize=10)

    # D ── 9-pose violin
    ax_d = fig.add_subplot(gs[1, 1:])
    pd_list = []; pl_list = []
    for _, r in df.nsmallest(min(8, len(df)), "best_affinity_kcal").iterrows():
        sc_l = r.get("all_scores", [])
        if isinstance(sc_l, list) and sc_l:
            pd_list.append(sc_l)
            pl_list.append(str(r["molecule_id"])[:12])
    if pd_list:
        parts = ax_d.violinplot(pd_list, showmedians=True, showextrema=True, widths=0.7)
        vc = [NATURE_COLORS[i % len(NATURE_COLORS)] for i in range(len(pd_list))]
        for pc, col in zip(parts["bodies"], vc):
            pc.set_facecolor(col); pc.set_alpha(0.65); pc.set_edgecolor(col)
        parts["cmedians"].set_colors("black"); parts["cmedians"].set_linewidth(2.5)
        parts["cmaxes"].set_colors("gray");   parts["cmins"].set_colors("gray")
        ax_d.set_xticks(range(1, len(pl_list) + 1))
        ax_d.set_xticklabels(pl_list, rotation=35, ha="right", fontsize=9.5)
        ax_d.set_ylabel("Pose Affinity (kcal/mol)", fontsize=13, fontweight="bold")
    ax_d.set_title("D   9-Pose Score Distribution (Top-8)\n"
                    "Spread = conformational flexibility in pocket",
                    fontsize=13, fontweight="bold", loc="left")

    # E ── top-6 horizontal bars
    ax_e = fig.add_subplot(gs[2, :2])
    t6  = df.nsmallest(TOP_K, "best_affinity_kcal").reset_index(drop=True)
    pal = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(t6)))
    bars_e = ax_e.barh(range(len(t6)), t6["best_affinity_kcal"].values,
                        color=pal, edgecolor="white", height=0.65)
    ax_e.set_yticks(range(len(t6)))
    ax_e.set_yticklabels(
        ["#{} {}".format(i + 1, str(r["molecule_id"])[:18])
         for i, r in t6.iterrows()], fontsize=11)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("Best Affinity (kcal/mol)", fontsize=13, fontweight="bold")
    ax_e.set_title("E   Top-{}  Compounds — {} x {}".format(TOP_K, gene, pdb_id),
                    fontsize=14, fontweight="bold", loc="left")
    for bar, (_, r) in zip(bars_e, t6.iterrows()):
        v = r["best_affinity_kcal"]
        ax_e.text(v - 0.04, bar.get_y() + bar.get_height() / 2,
                  "{:.3f}".format(v), va="center", ha="right",
                  fontsize=11, fontweight="bold", color="white")
        lbl = ""
        if pd.notna(r.get("pIC50")):
            lbl += "pIC50={:.2f}  ".format(r["pIC50"])
        if pd.notna(r.get("qed")):
            lbl += "QED={:.3f}".format(r["qed"])
        ax_e.text(0.01, bar.get_y() + bar.get_height() / 2 + 0.22,
                  lbl, transform=ax_e.get_yaxis_transform(),
                  fontsize=8.5, color="#333")
    ax_e.set_xlim(ax_e.get_xlim()[0], ax_e.get_xlim()[1] + 1.0)

    # F ── QED vs affinity
    ax_f = fig.add_subplot(gs[2, 2])
    dv2 = df[df["qed"].notna() & df["best_affinity_kcal"].notna()]
    if not dv2.empty:
        c_f = dv2["pIC50"] if dv2["pIC50"].notna().any() else "steelblue"
        sc2 = ax_f.scatter(dv2["best_affinity_kcal"], dv2["qed"],
                            s=70, alpha=0.65, c=c_f, cmap="plasma",
                            edgecolors="white", linewidths=0.5)
        if dv2["pIC50"].notna().any():
            plt.colorbar(sc2, ax=ax_f, label="pIC50", shrink=0.85)
    if not t6.empty and "qed" in t6.columns:
        ax_f.scatter(t6["best_affinity_kcal"], t6["qed"], s=200, marker="*",
                      c="gold", edgecolors="darkorange", linewidths=1.8,
                      zorder=5, label="Top-{}".format(TOP_K))
    ax_f.set_xlabel("Best Affinity (kcal/mol)", fontsize=13, fontweight="bold")
    ax_f.set_ylabel("QED Score", fontsize=13, fontweight="bold")
    ax_f.set_title("F   QED vs Affinity\nOptimal: high QED + low affinity",
                    fontsize=13, fontweight="bold", loc="left")
    ax_f.legend(fontsize=10)

    fig.suptitle(
        "Molecular Docking — {}  x  {}\n"
        "AutoDock Vina | {} poses/compound | Top-{} highlighted".format(
            gene, pdb_id, N_POSES, TOP_K),
        fontsize=18, fontweight="bold")

    _save(fig, FIGURES_DIR / "{}_{}_docking.pdf".format(gene, pdb_id))


def plot_top6_comparison(df_in: pd.DataFrame, gene: str) -> None:
    """Top-6 horizontal bar comparison across all PDB structures."""
    _style()
    df = df_in.copy()
    df["best_affinity_kcal"] = pd.to_numeric(df["best_affinity_kcal"], errors="coerce")
    df["pIC50"] = pd.to_numeric(df.get("pIC50", pd.Series(dtype=float)), errors="coerce")
    df["qed"]   = pd.to_numeric(df.get("qed",   pd.Series(dtype=float)), errors="coerce")

    pdb_ids = df["pdb_id"].dropna().unique().tolist()
    if not pdb_ids:
        return

    fig, axes = plt.subplots(1, len(pdb_ids), figsize=(9 * len(pdb_ids), 9))
    if len(pdb_ids) == 1:
        axes = [axes]

    for ax, pid in zip(axes, pdb_ids):
        sub = df[(df["pdb_id"] == pid) & df["best_affinity_kcal"].notna()]
        if sub.empty:
            continue
        t6 = sub.nsmallest(TOP_K, "best_affinity_kcal").reset_index(drop=True)
        pal = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(t6)))
        bars = ax.barh(range(len(t6)), t6["best_affinity_kcal"].values,
                        color=pal, edgecolor="white", height=0.68)
        ax.set_yticks(range(len(t6)))
        ax.set_yticklabels(
            ["Rank #{}\n{}".format(i + 1, str(r["molecule_id"])[:16])
             for i, r in t6.iterrows()], fontsize=10.5)
        ax.invert_yaxis()
        ax.set_xlabel("Binding Affinity (kcal/mol)", fontsize=13, fontweight="bold")
        ax.set_title("{}\nTop-{} vs {}".format(gene, TOP_K, pid),
                      fontsize=14, fontweight="bold")
        for bar, (_, r) in zip(bars, t6.iterrows()):
            v = r["best_affinity_kcal"]
            ax.text(v - 0.03, bar.get_y() + bar.get_height() / 2,
                    "{:.3f}".format(v), va="center", ha="right",
                    fontsize=11, fontweight="bold", color="white")
            lbl = ""
            if pd.notna(r.get("pIC50")):
                lbl += "pIC50={:.1f}  ".format(r["pIC50"])
            if pd.notna(r.get("qed")):
                lbl += "QED={:.2f}".format(r["qed"])
            ax.text(0.02, bar.get_y() + bar.get_height() / 2,
                    lbl, transform=ax.get_yaxis_transform(),
                    fontsize=9, color="#222", va="center")
        pocket_info = sub["pocket_ligand"].iloc[0] if "pocket_ligand" in sub.columns else "N/A"
        box_info    = sub["box_size"].iloc[0]       if "box_size"       in sub.columns else "N/A"
        ax.text(0.98, 0.02,
                "Pocket: {}\nBox: {}".format(pocket_info, box_info),
                transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                          edgecolor="goldenrod", alpha=0.9))
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 1.5)

    fig.suptitle("Top-{}  Best-Affinity Candidates — {}\n"
                  "AutoDock Vina | Cartesian-product docking".format(TOP_K, gene),
                  fontsize=18, fontweight="bold")
    plt.tight_layout()
    _save(fig, FIGURES_DIR / "{}_top6_docking.pdf".format(gene))


def plot_pose_profiles(df_in: pd.DataFrame, gene: str, top_n: int = 3) -> None:
    """Line chart of 9-pose affinity for top-N compounds (each PDB row)."""
    _style()
    df = df_in.copy()
    df["best_affinity_kcal"] = pd.to_numeric(df["best_affinity_kcal"], errors="coerce")
    df["pIC50"] = pd.to_numeric(df.get("pIC50", pd.Series(dtype=float)), errors="coerce")
    df = df[df["best_affinity_kcal"].notna()].copy()
    if df.empty:
        return

    pdb_ids = df["pdb_id"].dropna().unique().tolist()
    nr, nc = len(pdb_ids), top_n
    fig, axes = plt.subplots(nr, nc, figsize=(7 * nc, 6 * nr), squeeze=False)

    for ri, pid in enumerate(pdb_ids):
        sub  = df[df["pdb_id"] == pid]
        cpds = sub.nsmallest(min(top_n, len(sub)), "best_affinity_kcal").reset_index(drop=True)
        for ci in range(nc):
            ax = axes[ri][ci]
            if ci >= len(cpds):
                ax.set_visible(False)
                continue
            row = cpds.iloc[ci]
            sc_l = row.get("all_scores", [])
            if not isinstance(sc_l, list) or not sc_l:
                ax.set_visible(False)
                continue
            poses = list(range(1, len(sc_l) + 1))
            col   = NATURE_COLORS[(ri * nc + ci) % len(NATURE_COLORS)]
            ax.plot(poses, sc_l, "o-", color=col, lw=2.5, ms=9,
                    markeredgecolor="white", markeredgewidth=1.2)
            ax.fill_between(poses, sc_l, min(sc_l) - 0.3, alpha=0.15, color=col)
            ax.scatter([1], [sc_l[0]], s=180, color="gold", edgecolors="darkorange",
                        zorder=5, label="Best: {:.3f}".format(sc_l[0]))
            pic_s = "  pIC50={:.1f}".format(row["pIC50"]) if pd.notna(row.get("pIC50")) else ""
            ax.set_title("{} Rank #{} | {}\n{}{}".format(
                gene, ci + 1, pid, str(row["molecule_id"])[:16], pic_s),
                fontsize=12, fontweight="bold")
            ax.set_xlabel("Pose Number", fontsize=12, fontweight="bold")
            ax.set_ylabel("Affinity (kcal/mol)", fontsize=12, fontweight="bold")
            ax.set_xticks(range(1, len(sc_l) + 1))
            ax.set_xlim(0.5, len(sc_l) + 0.5)
            ax.legend(fontsize=10)

    fig.suptitle("Pose-by-Pose Affinity Profile — {}\n"
                  "{} conformations per compound".format(gene, N_POSES),
                  fontsize=17, fontweight="bold")
    plt.tight_layout()
    _save(fig, FIGURES_DIR / "{}_pose_profiles.pdf".format(gene))


def plot_docking_heatmap(df_in: pd.DataFrame, gene: str) -> None:
    """Compounds x PDB heatmap of best binding affinity."""
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn not installed; skipping heatmap (pip install seaborn)")
        return

    _style()
    df = df_in.copy()
    df["best_affinity_kcal"] = pd.to_numeric(df["best_affinity_kcal"], errors="coerce")
    df = df[df["best_affinity_kcal"].notna()].copy()
    if df.empty:
        return

    pivot = df.pivot_table(index="molecule_id", columns="pdb_id",
                            values="best_affinity_kcal", aggfunc="min")
    n_show = min(40, len(pivot))
    pivot  = pivot.loc[pivot.mean(axis=1).nsmallest(n_show).index]

    fig_h = max(10, n_show * 0.40)
    fig_w = max(8, len(pivot.columns) * 3.5 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn_r",
                 annot=(len(pivot.columns) <= 8), fmt=".2f",
                 linewidths=0.6, linecolor="white",
                 cbar_kws={"label": "Affinity (kcal/mol)", "shrink": 0.80},
                 annot_kws={"fontsize": 9, "fontweight": "bold"})
    ax.set_xlabel("PDB Structure", fontsize=13, fontweight="bold")
    ax.set_ylabel("Compound",      fontsize=13, fontweight="bold")
    ax.set_title("Cartesian-Product Docking Heatmap — {}\n"
                  "{} compounds x {} PDB structures | Best affinity (kcal/mol)".format(
                      gene, n_show, len(pivot.columns)),
                  fontsize=14, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=11, fontweight="bold")
    plt.yticks(fontsize=9)
    _save(fig, FIGURES_DIR / "{}_docking_heatmap.pdf".format(gene))


# ══════════════════════════════════════════════════════════════════════════════
# Main entry for main.py
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Automatic PDB retrieval from RCSB
# ══════════════════════════════════════════════════════════════════════════════

# Known high-quality EGFR structures with ligand resnames
# Used as fallback when no UniProt-based search is possible
_KNOWN_PDB_FOR_GENE = {
    "EGFR": [
        ("3W32", "W32"), ("4HJO", "WZ4"), ("2ITQ", "IRE"),
        ("3POZ", "03P"), ("4I23", "0WM"),
    ],
    "TRIM22":   [("2LMP", None), ("6FLO", None)],
    "PPP2R5A":  [("2IAE", None), ("6NTS", None)],
    "TP53":     [("2OCJ", None), ("3ZME", None)],
    "BRCA1":    [("1JNX", None)],
    "MYC":      [("6XRZ", None)],
    "KRAS":     [("4OBE", "GDP"), ("6OIM", "AMG")],
    "BRAF":     [("4RIW", "PLX"), ("5CSW", "VEM")],
    "CDK2":     [("1FIN", "ATP"), ("2C5Y", "SC2")],
    "CDK4":     [("2W96", "AQ4"), ("3G33", "P04")],
    "PIK3CA":   [("4JPS", "PIZ"), ("2ENQ", None)],
    "PTEN":     [("1D5R", None)],
    "AKT1":     [("4EJN", "AHA"), ("3CQW", None)],
    "MTOR":     [("4JSP", None)],
    "HER2":     [("3PP0", "FMM"), ("2A91", None)],
    "ALK":      [("2YFX", "TAE"), ("4ANS", "3LZ")],
    "MET":      [("2WGJ", "Y39"), ("3ZXZ", None)],
    "VEGFR2":   [("4ASD", "AXI"), ("3VHE", None)],
    "PDGFRA":   [("5GRN", None)],
    "FGFR1":    [("4V05", None)],
    "ABL1":     [("2HYY", "STI"), ("3CS9", "NIL")],
    "JAK2":     [("3KRR", "VX6"), ("4C62", None)],
    "STAT3":    [("6NJS", None)],
    "MDM2":     [("1YCR", None), ("4HG7", None)],
    "BCL2":     [("4LVT", "ABT"), ("2O21", None)],
    "HDAC1":    [("4BKX", "VK1")],
    "DNMT3A":   [("5YX2", None)],
}


def _fetch_pdb_from_rcsb(pdb_id: str, out_path: Path) -> bool:
    """
    Download a PDB file from RCSB. Returns True on success.
    Tries both PDB and mmCIF format (converted internally if needed).
    """
    import urllib.request
    import urllib.error

    if out_path.exists() and out_path.stat().st_size > 1000:
        logger.info("  [PDB] %s already cached at %s", pdb_id, out_path.name)
        return True

    urls = [
        "https://files.rcsb.org/download/{}.pdb".format(pdb_id.upper()),
        "https://files.rcsb.org/view/{}.pdb".format(pdb_id.upper()),
        "https://www.rcsb.org/pdb/files/{}.pdb".format(pdb_id.upper()),
    ]

    for url in urls:
        try:
            logger.info("  [PDB] Downloading %s from %s", pdb_id, url)
            req = urllib.request.Request(url, headers={"User-Agent": "DrugDiscoveryPipeline/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                content = response.read().decode("utf-8", errors="ignore")

            if len(content) < 500 or "ATOM" not in content:
                logger.warning("  [PDB] %s: empty or invalid response from %s", pdb_id, url)
                continue

            out_path.write_text(content)
            n_atoms = content.count("\nATOM")
            logger.info("  [PDB] %s downloaded: %d ATOM records, %.1f KB",
                        pdb_id, n_atoms, len(content) / 1024)
            return True

        except Exception as e:
            logger.warning("  [PDB] %s failed from %s: %s", pdb_id, url, e)
            continue

    logger.error("  [PDB] Could not download %s from any URL", pdb_id)
    return False


def _query_rcsb_for_uniprot(uniprot_id: str, n_top: int = 5) -> List[Dict]:
    """
    Query RCSB Search API to find the best PDB structures for a UniProt ID.
    Returns list of {"pdb_id": str, "resolution": float, "has_ligand": bool}.
    Filters: X-ray only, resolution <= 3.0 A, single chain.
    Falls back to empty list on any network error.
    """
    import urllib.request
    import urllib.error

    query_body = json.dumps({
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers"
                                     ".reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": uniprot_id,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 3.0,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True,
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined",
                       "direction": "asc"}],
        },
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            data=query_body,
            headers={"Content-Type": "application/json",
                     "User-Agent": "DrugDiscoveryPipeline/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        hits = data.get("result_set", [])
        logger.info("RCSB returned %d structures for UniProt %s", len(hits), uniprot_id)
        return [{"pdb_id": h["identifier"]} for h in hits[:n_top]]

    except Exception as e:
        logger.warning("RCSB search failed for %s: %s", uniprot_id, e)
        return []


def _build_pdb_configs_for_gene(
    gene: str,
    uniprot_row: Optional[pd.Series],
    pdb_store_dir: Path,
    n_structures: int = 3,
) -> List[Dict]:
    """
    Resolve PDB configs for one gene using this priority:
      1. Local files in pdb_store_dir/{gene}/ or pdb_store_dir/
      2. RCSB API search by UniProt accession
      3. Hard-coded fallback table (_KNOWN_PDB_FOR_GENE)

    Downloads any missing PDB files from RCSB automatically.

    Returns list of {"pdb_id", "pdb_path", "ligand_resname"}.
    """
    configs = []

    # ── 1. Scan local directories ──────────────────────────────────────────
    for search_dir in [pdb_store_dir / gene, pdb_store_dir]:
        if search_dir.is_dir():
            for pdb_file in sorted(search_dir.glob("*.pdb"))[:n_structures]:
                configs.append({
                    "pdb_id":         pdb_file.stem.upper(),
                    "pdb_path":       pdb_file,
                    "ligand_resname": None,
                })
            if configs:
                logger.info("[%s] Found %d local PDB files in %s",
                            gene, len(configs), search_dir)
                return configs

    # ── 2. RCSB API search by UniProt ──────────────────────────────────────
    pdb_store_dir.mkdir(parents=True, exist_ok=True)
    gene_pdb_dir = pdb_store_dir / gene
    gene_pdb_dir.mkdir(exist_ok=True)

    uniprot_id = None
    if uniprot_row is not None:
        uniprot_id = str(uniprot_row.get("uniprot_accession", "")).strip()
        if uniprot_id and uniprot_id != "nan":
            logger.info("[%s] Searching RCSB for UniProt %s ...", gene, uniprot_id)
            hits = _query_rcsb_for_uniprot(uniprot_id, n_top=n_structures + 5)

            for hit in hits[:n_structures]:
                pid = hit["pdb_id"].upper()
                pdb_path = gene_pdb_dir / "{}.pdb".format(pid)
                ok = _fetch_pdb_from_rcsb(pid, pdb_path)
                if ok:
                    configs.append({
                        "pdb_id":         pid,
                        "pdb_path":       pdb_path,
                        "ligand_resname": None,  # auto-detect
                    })
                if len(configs) >= n_structures:
                    break

    # ── 3. Hard-coded fallback ──────────────────────────────────────────────
    if not configs:
        known = _KNOWN_PDB_FOR_GENE.get(gene, [])
        if not known:
            # Generic fallback: try common EGFR structures
            logger.warning("[%s] No known PDB IDs. Add entries to _KNOWN_PDB_FOR_GENE "
                           "or place .pdb files in data/raw/pdb/%s/", gene, gene)
            return []

        logger.info("[%s] Using hard-coded PDB list: %s", gene,
                    [p[0] for p in known[:n_structures]])
        for pid, lig_rn in known[:n_structures]:
            pdb_path = gene_pdb_dir / "{}.pdb".format(pid)
            ok = _fetch_pdb_from_rcsb(pid, pdb_path)
            if ok:
                configs.append({
                    "pdb_id":         pid,
                    "pdb_path":       pdb_path,
                    "ligand_resname": lig_rn,
                })

    return configs


def run_docking_pipeline(
    scored_data: Dict,
    uniprot_df: pd.DataFrame,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """
    Stage 4 entry point called from main.py.

    Automatically:
      1. Checks for local PDB files  (data/raw/pdb/{GENE}/ or data/raw/pdb/)
      2. If none found, queries RCSB Search API by UniProt accession
      3. If RCSB unavailable, uses hard-coded high-quality structures
      4. Downloads missing PDB files directly from RCSB files server
      5. Prepares protein PDBQT, computes binding box, docks all top compounds
      6. Selects top-6 by best affinity, copies complex PDB files
      7. Generates publication-quality figures

    Output per gene per PDB:
      data/results/docking/{GENE}/{PDB_ID}/
        protein_clean.pdb                 cleaned protein
        protein.pdbqt                     Vina-ready receptor
        box_info.json                     pocket centre + box size
        {mol_id}/
          {mol_id}_ligand.pdbqt           prepared ligand
          {mol_id}_docked_all.pdbqt       all 9 poses
          {mol_id}_pose_1.pdbqt           best pose
          ...
          {mol_id}_pose_9.pdbqt
          {mol_id}_best_pose.pdbqt
          {mol_id}_complex.pdb            docked complex (PyMOL ready)
          {mol_id}_docked_smiles.txt      SMILES + all scores + pocket info
        all_docking_results.csv
        top6_summary.csv
        top6_docked_smiles.txt
        top6_best_compounds/
          rank1_{mol_id}_complex.pdb
          ...
          rank6_{mol_id}_complex.pdb

    Parameters
    ----------
    scored_data : dict
        {gene: {"df_scored": pd.DataFrame, ...}} from Stage 3.
    uniprot_df : pd.DataFrame
        UniProt mapping with columns gene_symbol, uniprot_accession.
    force_rerun : bool
        Re-dock even if checkpoints exist.

    Returns
    -------
    pd.DataFrame
        All docking results concatenated across genes and PDB structures.
    """
    if not VINA_AVAILABLE:
        logger.error(
            "AutoDock Vina Python package not installed.\n"
            "Install with:  pip install vina\n"
            "Skipping Stage 4."
        )
        return pd.DataFrame()

    docking_dir = RESULTS_DIR / "docking"
    docking_dir.mkdir(exist_ok=True)

    pdb_store = RAW_DIR / "pdb"
    pdb_store.mkdir(exist_ok=True)

    # Load manual PDB registry (highest priority, overrides everything)
    registry_path = RAW_DIR / "pdb_registry.json"
    manual_registry = {}
    if registry_path.exists():
        with open(registry_path) as fj:
            manual_registry = json.load(fj)
        logger.info("Loaded manual PDB registry: %d genes", len(manual_registry))

    all_dfs = []

    for gene, result in scored_data.items():
        df_scored = result["df_scored"] if isinstance(result, dict) else result

        # ── Checkpoint ──────────────────────────────────────────────────────
        ckpt = Checkpoint("s4_dock_{}".format(gene))
        if ckpt.exists() and not force_rerun:
            df_res = ckpt.load()
            logger.info("[%s] Loaded docking from checkpoint (%d rows)", gene, len(df_res))
            all_dfs.append(df_res)
            continue

        # ── Build PDB config list ────────────────────────────────────────────
        pdb_configs = []

        # Priority 1: manual registry
        if gene in manual_registry:
            for entry in manual_registry[gene]:
                pdb_configs.append({
                    "pdb_id":         entry["pdb_id"],
                    "pdb_path":       Path(entry["pdb_path"]),
                    "ligand_resname": entry.get("ligand_resname", None),
                })
            logger.info("[%s] Using %d structures from manual registry",
                        gene, len(pdb_configs))

        # Priority 2 + 3: auto-resolve (local → RCSB API → hard-coded)
        if not pdb_configs:
            uniprot_row = None
            if uniprot_df is not None and not uniprot_df.empty:
                mask = uniprot_df["gene_symbol"] == gene
                if mask.any():
                    uniprot_row = uniprot_df[mask].iloc[0]

            pdb_configs = _build_pdb_configs_for_gene(
                gene, uniprot_row, pdb_store, n_structures=3
            )

        if not pdb_configs:
            logger.error(
                "[%s] No PDB structures available after all attempts.\n"
                "  Manual option: place PDB files in  data/raw/pdb/%s/*.pdb",
                gene, gene
            )
            continue

        logger.info("[%s] Will dock against %d PDB structure(s): %s",
                    gene, len(pdb_configs),
                    [c["pdb_id"] for c in pdb_configs])

        # ── Run Cartesian-product docking ────────────────────────────────────
        df_res = run_full_docking_pipeline(
            pdb_configs    = pdb_configs,
            compounds_df   = df_scored,
            gene           = gene,
            base_out_dir   = docking_dir / gene,
            top_n          = TOP_N_COMPOUNDS,
            n_poses        = N_POSES,
            exhaustiveness = EXHAUSTIVENESS,
            force_rerun    = force_rerun,
        )

        ckpt.save(df_res)
        if not df_res.empty:
            all_dfs.append(df_res)

        # ── Figures ──────────────────────────────────────────────────────────
        for pid in df_res["pdb_id"].dropna().unique():
            try:
                plot_docking_panel(df_res[df_res["pdb_id"] == pid], gene, pid)
            except Exception as e:
                logger.warning("[%s/%s] plot_docking_panel: %s", gene, pid, e)
        try:
            plot_top6_comparison(df_res, gene)
        except Exception as e:
            logger.warning("[%s] plot_top6_comparison: %s", gene, e)
        try:
            plot_pose_profiles(df_res, gene)
        except Exception as e:
            logger.warning("[%s] plot_pose_profiles: %s", gene, e)
        try:
            plot_docking_heatmap(df_res, gene)
        except Exception as e:
            logger.warning("[%s] plot_docking_heatmap: %s", gene, e)

    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if not df_all.empty:
        safe_save_csv(df_all, RESULTS_DIR / "docking_results_all.csv")

    return df_all
