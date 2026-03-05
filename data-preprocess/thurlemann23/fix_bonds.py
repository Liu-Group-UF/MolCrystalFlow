#!/usr/bin/env python
"""
Multi-step bond order and formal charge fixing for SDF files.

This script implements a comprehensive multi-step bond fixing procedure:
1. Try zeroing formal charges and adjusting to neutralize
2. Try fixing nitrogen and carbon valency issues
3. Try InChI round-trip to recover correct bond orders
4. Try rdDetermineBonds to re-determine bonds from 3D structure

Based on the multi_step_bond_fix.ipynb notebook.

Usage:
    python fix_bonds.py --input all_monomers.sdf --output all_cleaned_monomers.sdf
    python fix_bonds.py --input all_monomers.sdf --output all_cleaned_monomers.sdf --save_indices
"""
import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem.rdchem import BondType
from tqdm import tqdm

from common import (
    adjust_formal_charges_neutralize,
    fix_carbon_valency,
    fix_nitrogen_valency,
    fix_valency,
    zero_formal_charges,
)

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")


def get_inchi_atom_mapping_with_h(mol: Chem.Mol) -> Optional[dict]:
    """
    Get atom mapping between original molecule and InChI-derived molecule.

    Returns:
        Dictionary mapping InChI atom indices to original atom indices, or None if failed.
    """
    try:
        # Get SMILES without sanitization
        smiles = Chem.MolToSmiles(mol)
        # Create mol from SMILES to get consistent atom ordering
        ref_mol = Chem.MolFromSmiles(smiles)
        if ref_mol is None:
            return None

        Chem.RemoveStereochemistry(ref_mol)
        inchi = Chem.MolToInchi(ref_mol)
        if inchi is None:
            return None

        inchi_mol = Chem.MolFromInchi(inchi)
        if inchi_mol is None:
            return None

        inchi_mol = Chem.AddHs(inchi_mol)

        # Simple 1:1 mapping assuming same atom count
        if mol.GetNumAtoms() == inchi_mol.GetNumAtoms():
            return {i: i for i in range(mol.GetNumAtoms())}

        return None
    except:
        return None


def transfer_bonds_and_charges(ref_mol: Chem.Mol, target_mol: Chem.Mol,
                                atom_map: dict) -> Chem.Mol:
    """
    Transfer bond orders and charges from reference molecule to target.

    Args:
        ref_mol: Reference molecule with correct bond orders.
        target_mol: Target molecule to modify.
        atom_map: Dictionary mapping ref_mol atom indices to target_mol indices.

    Returns:
        Modified target molecule.
    """
    target = Chem.RWMol(target_mol)

    # Transfer formal charges
    for ref_idx, target_idx in atom_map.items():
        if ref_idx < ref_mol.GetNumAtoms() and target_idx < target.GetNumAtoms():
            ref_atom = ref_mol.GetAtomWithIdx(ref_idx)
            target_atom = target.GetAtomWithIdx(target_idx)
            target_atom.SetFormalCharge(ref_atom.GetFormalCharge())

    # Transfer bond orders
    for bond in ref_mol.GetBonds():
        ref_begin = bond.GetBeginAtomIdx()
        ref_end = bond.GetEndAtomIdx()

        if ref_begin in atom_map and ref_end in atom_map:
            target_begin = atom_map[ref_begin]
            target_end = atom_map[ref_end]

            target_bond = target.GetBondBetweenAtoms(target_begin, target_end)
            if target_bond is not None:
                target_bond.SetBondType(bond.GetBondType())

    return target.GetMol()


def fix_mol_step1(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Step 1: Zero formal charges and neutralize.
    """
    try:
        mol = zero_formal_charges(mol)
        fixed_mol, _ = adjust_formal_charges_neutralize(mol)
        Chem.SanitizeMol(fixed_mol)
        return fixed_mol
    except:
        return None


def fix_mol_step2(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Step 2: Fix nitrogen and carbon valency, then neutralize.
    """
    try:
        mol = zero_formal_charges(mol)
        mol = fix_nitrogen_valency(mol)
        mol = fix_carbon_valency(mol)
        fixed_mol, _ = adjust_formal_charges_neutralize(mol)
        Chem.SanitizeMol(fixed_mol)
        return fixed_mol
    except:
        return None


def fix_mol_step3(mol: Chem.Mol, inchi: str) -> Optional[Chem.Mol]:
    """
    Step 3: Use InChI round-trip to recover correct bond orders.
    """
    try:
        atom_map = get_inchi_atom_mapping_with_h(mol)
        if atom_map is None:
            return None

        no_atoms_mol = mol.GetNumAtoms()

        # Get mol from InChI
        qm9_mol_from_inchl = Chem.MolFromInchi(inchi)
        if qm9_mol_from_inchl is None:
            return None

        qm9_mol_from_inchl = Chem.AddHs(qm9_mol_from_inchl)
        Chem.Kekulize(qm9_mol_from_inchl, clearAromaticFlags=True)

        no_atoms_qm9 = qm9_mol_from_inchl.GetNumAtoms()
        if no_atoms_mol != no_atoms_qm9:
            return None

        _mol = fix_valency(mol)
        new_mol = transfer_bonds_and_charges(qm9_mol_from_inchl, _mol, atom_map)
        new_mol, _ = adjust_formal_charges_neutralize(new_mol)
        Chem.Kekulize(new_mol, clearAromaticFlags=True)
        Chem.SanitizeMol(new_mol)
        return new_mol
    except:
        return None


def fix_mol_step4(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Step 4: Use rdDetermineBonds to re-determine bonds from 3D structure.
    """
    try:
        mol = zero_formal_charges(mol)
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        fixed_mol, _ = adjust_formal_charges_neutralize(mol)
        Chem.SanitizeMol(fixed_mol)
        return fixed_mol
    except:
        return None


def fix_mol_multistep(mol: Chem.Mol, smiles: str = None) -> Tuple[Chem.Mol, str]:
    """
    Apply multi-step bond fixing to a molecule.

    Args:
        mol: RDKit Mol object with potentially incorrect bonds.
        smiles: Optional SMILES string for InChI derivation.

    Returns:
        Tuple of (fixed_mol, method_used).
    """
    if mol is None:
        return None, "none"

    # Step 1: Zero charges and neutralize
    result = fix_mol_step1(mol)
    if result is not None:
        return result, "step1_neutralize"

    # Step 2: Fix valency then neutralize
    result = fix_mol_step2(mol)
    if result is not None:
        return result, "step2_valency"

    # Step 3: InChI round-trip (if SMILES available)
    if smiles is not None:
        try:
            ref_mol = Chem.MolFromSmiles(smiles)
            if ref_mol is not None:
                Chem.RemoveStereochemistry(ref_mol)
                inchi = Chem.MolToInchi(ref_mol)
                if inchi is not None:
                    result = fix_mol_step3(mol, inchi)
                    if result is not None:
                        return result, "step3_inchi"
        except:
            pass

    # Step 4: rdDetermineBonds
    result = fix_mol_step4(mol)
    if result is not None:
        return result, "step4_determine_bonds"

    # All steps failed, return original
    return mol, "failed"


def fix_sdf_file(
    input_sdf: str,
    output_sdf: str,
    save_indices: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Fix bond orders and charges for all molecules in an SDF file.

    Args:
        input_sdf: Path to input SDF file.
        output_sdf: Path to output SDF file.
        save_indices: Whether to save valid/invalid indices.
        verbose: Whether to print progress.

    Returns:
        Dictionary with processing statistics.
    """
    # Load molecules
    if verbose:
        print(f"Loading molecules from {input_sdf}...")

    supplier = Chem.SDMolSupplier(input_sdf, removeHs=False, sanitize=False)
    mols = list(supplier)

    if verbose:
        print(f"Loaded {len(mols)} molecules")

    # Extract SMILES for each molecule (before fixing)
    smiles_list = []
    for mol in mols:
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)
            except:
                smiles_list.append(None)
        else:
            smiles_list.append(None)

    # Apply multi-step fixing
    fixed_mols = []
    valid_indices = []
    invalid_indices = []
    method_counts = {}

    iterator = tqdm(enumerate(mols), total=len(mols), desc="Fixing bonds") if verbose else enumerate(mols)

    for i, mol in iterator:
        smiles = smiles_list[i] if i < len(smiles_list) else None

        fixed_mol, method = fix_mol_multistep(mol, smiles)

        # Track method usage
        method_counts[method] = method_counts.get(method, 0) + 1

        if method != "failed" and method != "none":
            fixed_mols.append(fixed_mol)
            valid_indices.append(i)
        else:
            # Keep original molecule even if fixing failed
            fixed_mols.append(mol)
            invalid_indices.append(i)

    # Validate fixed molecules
    if verbose:
        print("\nValidating fixed molecules...")

    rdkit_valid = []
    rdkit_invalid = []

    for i, mol in enumerate(fixed_mols):
        if mol is None:
            rdkit_invalid.append(i)
            continue

        try:
            Chem.SanitizeMol(mol)
            rdkit_valid.append(i)
        except:
            rdkit_invalid.append(i)
            # Replace with original
            fixed_mols[i] = mols[i]

    # Save output SDF
    output_path = Path(output_sdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = Chem.SDWriter(str(output_path))
    written = 0
    for mol in fixed_mols:
        if mol is not None:
            try:
                writer.write(mol)
                written += 1
            except:
                pass
    writer.close()

    if verbose:
        print(f"\nSaved {written} molecules to {output_sdf}")

    # Save indices
    if save_indices:
        output_dir = output_path.parent
        np.save(str(output_dir / "valid_indices.npy"), np.array(valid_indices))
        np.save(str(output_dir / "invalid_indices.npy"), np.array(invalid_indices))
        np.save(str(output_dir / "bond_unfixed.npy"), np.array(invalid_indices))

        # Save SMILES
        with open(str(output_dir / "smiles_list.pkl"), "wb") as f:
            pickle.dump(smiles_list, f)

        if verbose:
            print(f"Saved indices to {output_dir}")

    # Print statistics
    if verbose:
        print("\n" + "=" * 60)
        print("Bond Fixing Statistics:")
        print("=" * 60)
        print(f"  Total molecules: {len(mols)}")
        print(f"  Successfully fixed: {len(valid_indices)}")
        print(f"  Failed to fix: {len(invalid_indices)}")
        print(f"  RDKit valid after fixing: {len(rdkit_valid)}")
        print(f"  RDKit invalid after fixing: {len(rdkit_invalid)}")
        print("\nMethod breakdown:")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"  {method}: {count}")

    stats = {
        "total": len(mols),
        "valid": len(valid_indices),
        "invalid": len(invalid_indices),
        "rdkit_valid": len(rdkit_valid),
        "rdkit_invalid": len(rdkit_invalid),
        "methods": method_counts,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Multi-step bond order and formal charge fixing for SDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python fix_bonds.py --input all_monomers.sdf --output all_cleaned_monomers.sdf

  # Save indices and SMILES
  python fix_bonds.py --input all_monomers.sdf --output all_cleaned_monomers.sdf --save_indices

  # Process without verbose output
  python fix_bonds.py -i input.sdf -o output.sdf -q
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input SDF file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output SDF file",
    )
    parser.add_argument(
        "--save_indices",
        "-s",
        action="store_true",
        help="Save valid/invalid indices and SMILES to output directory",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fix_sdf_file(
        input_sdf=str(input_path),
        output_sdf=args.output,
        save_indices=args.save_indices,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
