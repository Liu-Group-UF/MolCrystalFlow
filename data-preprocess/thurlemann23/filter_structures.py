#!/usr/bin/env python
"""
Filter and validate molecular crystal structures from Thurlemann2023 dataset.

This script:
1. Selects only unstrained structures (config_index=2, i.e., lattice scale=1.0)
2. Validates each structure by checking that all molecules have the same chemical formula
3. Validates that the number of detected molecules matches the expected z value
4. Assigns building block indices (bb_indices) to each atom
5. Outputs valid structures with bb_indices and optionally coarse-grained representations

Usage:
    python filter_structures.py --input molecule_crystals.extxyz --output valid_molcrystals.extxyz
    python filter_structures.py --input molecule_crystals.extxyz --output valid.extxyz --coarse_grained cg.extxyz
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from tqdm import tqdm

from common import ELEMENT_CUTOFFS, find_all_neighbors, get_molecule_groups


def atoms_to_formula(atom_list):
    """Convert a list of element symbols to a chemical formula string."""
    element_counts = Counter(atom_list)
    formula = "".join(
        f"{elem}{element_counts[elem] if element_counts[elem] > 1 else ''}"
        for elem in sorted(element_counts.keys(), key=lambda x: (x.islower(), x))
    )
    return formula


def coarse_grain_molecule(atoms, replace_with="C", validate_z=True):
    """
    Replace each molecular group with a single atom at its center of mass.

    Args:
        atoms: ASE Atoms object representing the crystal structure.
        replace_with: Element symbol for the coarse-grained representation.
        validate_z: If True, validate that number of molecules matches z value.

    Returns:
        new_atoms: Coarse-grained ASE Atoms object.
        monomer: Chemical formula of the monomer.
        molecule_groups: List of lists containing atom indices for each molecule.

    Raises:
        AssertionError: If molecules have different chemical formulas or z mismatch.
        ValueError: If element cutoffs are missing.
    """
    molecule_groups, mol_offsets = get_molecule_groups(atoms)
    new_atoms = Atoms()

    for mol, offsets in zip(molecule_groups, mol_offsets):
        ref_pos = atoms.positions[mol[0]]
        nl_pos = ref_pos + offsets
        # Compute center of mass
        com = np.mean(np.vstack([ref_pos, nl_pos]), axis=0)
        new_atoms.extend(Atoms(replace_with, positions=[com]))

    new_atoms.set_cell(atoms.get_cell())
    new_atoms.set_pbc(True)

    # Validate: all monomers must have the same chemical formula
    all_monomers = [atoms_to_formula(atoms.symbols[mol]) for mol in molecule_groups]
    if len(set(all_monomers)) != 1:
        raise AssertionError(f"Multiple monomers found: {set(all_monomers)}")
    monomer = all_monomers[0]

    # Validate: number of molecules must match z value if present
    if validate_z and "z" in atoms.info:
        z_value = atoms.info["z"]
        if len(molecule_groups) != z_value:
            raise AssertionError(
                f"z mismatch: expected {z_value} molecules, found {len(molecule_groups)}"
            )

    return new_atoms, monomer, molecule_groups


def get_all_monomers(atoms):
    """Extract all individual monomers from a crystal structure."""
    molecule_groups, offset_groups = get_molecule_groups(atoms)
    monomers = []

    for mol, offsets in zip(molecule_groups, offset_groups):
        symbols = [atoms[i].symbol for i in mol]
        positions = offsets
        monomer = Atoms(symbols, positions=positions)
        monomer.center()
        monomers.append(monomer)

    return monomers


def assign_group_indices(molecule_groups, num_atoms):
    """
    Assign group IDs to each atom.

    Args:
        molecule_groups: List of lists of atom indices.
        num_atoms: Total number of atoms.

    Returns:
        group_indices: Array of shape (num_atoms,) with group ID for each atom.
    """
    group_indices = np.full(num_atoms, -1, dtype=int)

    for group_id, group in enumerate(molecule_groups):
        for atom_idx in group:
            group_indices[atom_idx] = group_id

    return group_indices


def filter_structures(
    input_path: str,
    output_path: str,
    coarse_grained_path: str = None,
    monomers_path: str = None,
    bad_indices_path: str = None,
    select_unstrained: bool = True,
    verbose: bool = True,
) -> tuple:
    """
    Filter and validate molecular crystal structures.

    Args:
        input_path: Path to input extended XYZ file.
        output_path: Path to output valid structures.
        coarse_grained_path: Optional path to save coarse-grained structures.
        monomers_path: Optional path to save extracted monomers.
        bad_indices_path: Optional path to save indices of invalid structures.
        select_unstrained: If True, select only unstrained (config_index=2) structures.
        verbose: Whether to print progress information.

    Returns:
        Tuple of (num_valid, num_invalid, num_total).
    """
    # Read all structures
    if verbose:
        print(f"Reading structures from {input_path}")

    if select_unstrained:
        # Select every 5th structure starting from index 2 (unstrained, scale=1.0)
        all_atoms = read(input_path, index="2::5", format="extxyz")
        if verbose:
            print(f"Selected {len(all_atoms)} unstrained structures (config_index=2)")
    else:
        all_atoms = read(input_path, index=":", format="extxyz")
        if verbose:
            print(f"Loaded {len(all_atoms)} total structures")

    # Filter structures
    valid_structures = []
    coarse_grained = []
    all_monomers = []
    bad_indices = []
    monomer_formulas = []

    if verbose:
        iterator = tqdm(enumerate(all_atoms), total=len(all_atoms), desc="Filtering")
    else:
        iterator = enumerate(all_atoms)

    for i, atoms in iterator:
        # Original index in full dataset (if unstrained selection was used)
        original_idx = 2 + i * 5 if select_unstrained else i
        if original_idx == 53742:
            atoms.pbc = False
        try:
            cg_atoms, monomer_formula, molecule_groups = coarse_grain_molecule(atoms)
            z = atoms.info['z_value']
            assert len(cg_atoms) == z, f"index {original_idx}: {z} != {len(cg_atoms)}"
        except (AssertionError, ValueError) as e:
            if verbose:
                tqdm.write(f"Index {original_idx}: {e}")
            bad_indices.append(original_idx)
            continue

        # Assign building block indices to each atom
        bb_indices = assign_group_indices(molecule_groups, len(atoms))
        atoms.arrays["bb_indices"] = bb_indices

        # Store monomer formula in structure info
        atoms.info["monomer"] = monomer_formula

        # Valid structure
        valid_structures.append(atoms)
        cg_atoms.info["monomer"] = monomer_formula
        coarse_grained.append(cg_atoms)
        monomer_formulas.append(monomer_formula)

        # Extract monomers if requested
        if monomers_path is not None:
            monomers = get_all_monomers(atoms)
            all_monomers.extend(monomers)

    # Statistics
    num_valid = len(valid_structures)
    num_invalid = len(bad_indices)
    num_total = len(all_atoms)

    if verbose:
        print(f"\nResults:")
        print(f"  Valid structures: {num_valid}")
        print(f"  Invalid structures: {num_invalid}")
        print(f"  Unique monomers: {len(set(monomer_formulas))}")

    # Save outputs
    write(output_path, valid_structures, format="extxyz")
    if verbose:
        print(f"Saved valid structures to {output_path}")

    if coarse_grained_path is not None:
        write(coarse_grained_path, coarse_grained, format="extxyz")
        if verbose:
            print(f"Saved coarse-grained structures to {coarse_grained_path}")

    if monomers_path is not None and all_monomers:
        write(monomers_path, all_monomers, format="extxyz")
        if verbose:
            print(f"Saved {len(all_monomers)} monomers to {monomers_path}")

    if bad_indices_path is not None:
        np.save(bad_indices_path, np.array(bad_indices))
        if verbose:
            print(f"Saved bad indices to {bad_indices_path}")

    return num_valid, num_invalid, num_total


def main():
    parser = argparse.ArgumentParser(
        description="Filter and validate molecular crystal structures from Thurlemann2023 dataset."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input extended XYZ file (from hdf2xyz.py)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="valid_molcrystals.extxyz",
        help="Path to output valid structures (default: valid_molcrystals.extxyz)",
    )
    parser.add_argument(
        "--coarse_grained",
        "-c",
        type=str,
        default=None,
        help="Optional path to save coarse-grained structures",
    )
    parser.add_argument(
        "--monomers",
        "-m",
        type=str,
        default=None,
        help="Optional path to save extracted monomers",
    )
    parser.add_argument(
        "--bad_indices",
        "-b",
        type=str,
        default=None,
        help="Optional path to save indices of invalid structures (.npy)",
    )
    parser.add_argument(
        "--all_configs",
        action="store_true",
        help="Process all configurations instead of only unstrained (config_index=2)",
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

    filter_structures(
        input_path=str(input_path),
        output_path=args.output,
        coarse_grained_path=args.coarse_grained,
        monomers_path=args.monomers,
        bad_indices_path=args.bad_indices,
        select_unstrained=not args.all_configs,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
