#!/usr/bin/env python
"""
Generate auxiliary molecular features for monomer structures.

This script:
1. Converts XYZ to SDF format using Open Babel
2. Fixes bond orders and formal charges using RDKit
3. Extracts molecular features (basic, chemical, geometric)
4. Saves features to pickle files for use in training

Usage:
    python generate_features.py --input all_monomers.xyz --output_dir ./features
    python generate_features.py --input all_monomers.xyz --output_dir ./features --coarse_grained cg.extxyz
"""
import argparse
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from ase.io import read
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from tqdm import tqdm

from common import (
    adjust_formal_charges_neutralize,
    fix_carbon_valency,
    fix_mol_bonds,
    fix_nitrogen_valency,
    zero_formal_charges,
)

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")


# Feature names for extraction
BASIC_FEATURES = [
    "num_heavy_atoms",
    "molecular_weight",
]

CHEMICAL_FEATURES = [
    "is_chiral",
    "num_hbd",
    "num_hba",
    "num_rotatable_bonds",
    "num_rings",
    "num_aromatic_rings",
    "logp",
    "tpsa",
]

GEOMETRIC_FEATURES = [
    "radius_of_gyration",
    "asphericity",
    "eccentricity",
    "planarity",
]


def xyz_to_sdf(xyz_path: str, sdf_path: str) -> bool:
    """
    Convert XYZ file to SDF using Open Babel.

    Args:
        xyz_path: Path to input XYZ file.
        sdf_path: Path to output SDF file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        cmd = ["obabel", xyz_path, "-O", sdf_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Open Babel error: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        print("Error: Open Babel (obabel) not found. Please install it:")
        print("  conda install -c conda-forge openbabel")
        print("  or: apt-get install openbabel")
        return False


def calculate_geometric_descriptors(positions: np.ndarray) -> Dict[str, float]:
    """Calculate geometric descriptors from atomic positions."""
    positions = np.asarray(positions)
    centered_pos = positions - positions.mean(axis=0)
    cov_matrix = np.dot(centered_pos.T, centered_pos) / len(centered_pos)

    eigenvals = np.linalg.eigvalsh(cov_matrix)
    eigenvals = np.sort(eigenvals)[::-1]  # Descending order

    I1, I2, I3 = eigenvals
    radius_of_gyration = np.sqrt(np.sum(eigenvals))
    asphericity = I1 - 0.5 * (I2 + I3)
    eccentricity = (I1 - I2) / I1 if I1 > 0 else 0.0
    planarity = 1.0 - (I3 / I1) if I1 > 0 else 0.0

    return {
        "radius_of_gyration": float(radius_of_gyration),
        "principal_moment_1": float(I1),
        "principal_moment_2": float(I2),
        "principal_moment_3": float(I3),
        "asphericity": float(asphericity),
        "eccentricity": float(eccentricity),
        "planarity": float(planarity),
    }


def calculate_rdkit_descriptors(mol: Chem.Mol) -> Dict[str, Union[int, float, bool]]:
    """Calculate RDKit molecular descriptors."""
    if mol is None:
        return {}

    descriptors = {}
    try:
        descriptors["num_atoms"] = mol.GetNumAtoms()
        descriptors["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
        descriptors["num_rings"] = rdMolDescriptors.CalcNumRings(mol)
        descriptors["num_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        descriptors["num_hbd"] = rdMolDescriptors.CalcNumHBD(mol)
        descriptors["num_hba"] = rdMolDescriptors.CalcNumHBA(mol)
        descriptors["num_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        descriptors["molecular_weight"] = Descriptors.MolWt(mol)
        descriptors["exact_molecular_weight"] = Descriptors.ExactMolWt(mol)
        descriptors["tpsa"] = rdMolDescriptors.CalcTPSA(mol)
        descriptors["logp"] = Crippen.MolLogP(mol)
        descriptors["is_chiral"] = float(
            bool(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        )

        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        descriptors["num_chiral_centers"] = len(chiral_centers)
        descriptors["num_R_centers"] = sum(1 for _, cfg in chiral_centers if cfg == "R")
        descriptors["num_S_centers"] = sum(1 for _, cfg in chiral_centers if cfg == "S")

        try:
            descriptors["spherocity"] = rdMolDescriptors.CalcSpherocityIndex(mol)
        except:
            descriptors["spherocity"] = 0.0

    except Exception as e:
        pass

    return descriptors


def extract_features_single_molecule(
    positions: np.ndarray, mol: Chem.Mol
) -> Dict[str, Union[int, float, bool]]:
    """Extract all features for a single molecule."""
    features = {}

    try:
        positions = np.asarray(positions)
        features.update(calculate_geometric_descriptors(positions))
        features.update(calculate_rdkit_descriptors(mol))
    except Exception as e:
        pass

    return features


def generate_features(
    input_xyz: str,
    output_dir: str,
    coarse_grained_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Generate auxiliary molecular features from monomer XYZ file.

    Args:
        input_xyz: Path to input XYZ file containing monomers.
        output_dir: Directory to save output files.
        coarse_grained_path: Optional path to coarse-grained structures
                             (to get num_bbs per structure).
        verbose: Whether to print progress information.

    Returns:
        Dictionary with processing statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert XYZ to SDF
    sdf_path = output_dir / "all_monomers.sdf"
    if verbose:
        print(f"Converting {input_xyz} to SDF format...")

    if not xyz_to_sdf(input_xyz, str(sdf_path)):
        raise RuntimeError("Failed to convert XYZ to SDF")

    if verbose:
        print(f"Created {sdf_path}")

    # Step 2: Load structures
    if verbose:
        print("Loading structures...")

    xyz_mols = read(input_xyz, index=":")
    sdf_supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    sdf_mols = list(sdf_supplier)

    if len(xyz_mols) != len(sdf_mols):
        print(f"Warning: {len(xyz_mols)} XYZ structures vs {len(sdf_mols)} SDF molecules")

    # Step 3: Fix bonds and extract SMILES
    if verbose:
        print("Fixing bond orders and charges...")

    fixed_mols = []
    smiles_list = []
    valid_indices = []
    invalid_indices = []

    iterator = tqdm(enumerate(sdf_mols), total=len(sdf_mols), desc="Fixing bonds") if verbose else enumerate(sdf_mols)

    for i, mol in iterator:
        if mol is None:
            invalid_indices.append(i)
            fixed_mols.append(None)
            smiles_list.append(None)
            continue

        try:
            # Get initial SMILES for reference
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)

            # Fix bonds
            fixed_mol = fix_mol_bonds(mol, smiles)

            if fixed_mol is not None:
                fixed_mols.append(fixed_mol)
                valid_indices.append(i)
            else:
                fixed_mols.append(mol)  # Keep original
                invalid_indices.append(i)

        except Exception as e:
            invalid_indices.append(i)
            fixed_mols.append(mol)
            smiles_list.append(None)

    if verbose:
        print(f"Fixed {len(valid_indices)} molecules, {len(invalid_indices)} failed")

    # Save cleaned SDF
    cleaned_sdf_path = output_dir / "all_cleaned_monomers.sdf"
    writer = Chem.SDWriter(str(cleaned_sdf_path))
    for mol in fixed_mols:
        if mol is not None:
            try:
                writer.write(mol)
            except:
                pass
    writer.close()
    if verbose:
        print(f"Saved cleaned molecules to {cleaned_sdf_path}")

    # Save SMILES list
    smiles_path = output_dir / "smiles_list.pkl"
    with open(smiles_path, "wb") as f:
        pickle.dump(smiles_list, f)

    # Save indices
    np.save(str(output_dir / "valid_indices.npy"), np.array(valid_indices))
    np.save(str(output_dir / "invalid_indices.npy"), np.array(invalid_indices))

    # Step 4: Extract features
    if verbose:
        print("Extracting molecular features...")

    all_features = []
    iterator = tqdm(
        zip(xyz_mols, fixed_mols), total=len(xyz_mols), desc="Extracting features"
    ) if verbose else zip(xyz_mols, fixed_mols)

    for atoms, mol in iterator:
        positions = atoms.get_positions()
        features = extract_features_single_molecule(positions, mol)
        all_features.append(features)

    # Save raw features
    features_path = output_dir / "extra_molcrystal_features.pkl"
    with open(features_path, "wb") as f:
        pickle.dump(all_features, f)

    # Step 5: Aggregate features into arrays
    if verbose:
        print("Aggregating features...")

    basic_feats = []
    chemical_feats = []
    geometric_feats = []

    for features in all_features:
        tmp_basic = [features.get(k, 0) for k in BASIC_FEATURES]
        tmp_chemical = [float(features.get(k, 0)) for k in CHEMICAL_FEATURES]
        tmp_geometric = [features.get(k, 0) for k in GEOMETRIC_FEATURES]

        basic_feats.append(tmp_basic)
        chemical_feats.append(tmp_chemical)
        geometric_feats.append(tmp_geometric)

    # Get num_bbs if coarse-grained structures provided
    num_bbs = None
    if coarse_grained_path is not None:
        if verbose:
            print(f"Loading coarse-grained structures from {coarse_grained_path}")
        cg_atoms = read(coarse_grained_path, index=":")
        num_bbs = [len(atoms) for atoms in cg_atoms]

    # Save aggregated features
    agg_dict = {
        "basic_features": np.array(basic_feats),
        "chemical_features": np.array(chemical_feats),
        "geometric_features": np.array(geometric_feats),
    }
    if num_bbs is not None:
        agg_dict["num_bbs"] = np.array(num_bbs)

    agg_path = output_dir / "agg_extra_features.pkl"
    with open(agg_path, "wb") as f:
        pickle.dump(agg_dict, f)

    if verbose:
        print(f"\nSaved files to {output_dir}:")
        print(f"  - all_monomers.sdf: Raw SDF from Open Babel")
        print(f"  - all_cleaned_monomers.sdf: Bond-fixed SDF")
        print(f"  - smiles_list.pkl: SMILES strings")
        print(f"  - valid_indices.npy, invalid_indices.npy: Processing status")
        print(f"  - extra_molcrystal_features.pkl: Raw features per molecule")
        print(f"  - agg_extra_features.pkl: Aggregated feature arrays")

    stats = {
        "total": len(sdf_mols),
        "valid": len(valid_indices),
        "invalid": len(invalid_indices),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate auxiliary molecular features from monomer XYZ file."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input XYZ file containing monomers (e.g., all_monomers.xyz)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="./features",
        help="Directory to save output files (default: ./features)",
    )
    parser.add_argument(
        "--coarse_grained",
        "-c",
        type=str,
        default=None,
        help="Optional path to coarse-grained structures (to get num_bbs)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    generate_features(
        input_xyz=str(input_path),
        output_dir=args.output_dir,
        coarse_grained_path=args.coarse_grained,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
