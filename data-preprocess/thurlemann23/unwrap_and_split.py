#!/usr/bin/env python
"""
Unwrap molecules and split data into train/validation/test sets.

This script:
1. Unwraps molecules across periodic boundaries so each is fully connected
2. Splits data by unique monomer formulas (no formula appears in multiple splits)
3. Outputs train, validation, and test XYZ files and index arrays

Usage:
    python unwrap_and_split.py --input valid_molcrystals.extxyz --output_dir ./splits
    python unwrap_and_split.py --input valid.extxyz --output_dir ./splits --train_size 10000 --test_size 750
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from tqdm import tqdm

from common import get_molecule_groups


def unwrap_building_blocks(atoms):
    """
    Reassign building block positions using offsets from get_molecule_groups().
    This unwraps molecules across periodic boundaries so each is fully connected.

    Args:
        atoms: ASE Atoms object with periodic boundary conditions.

    Returns:
        unwrapped_atoms: ASE Atoms object with updated positions.
    """
    molecule_groups, offset_groups = get_molecule_groups(atoms)
    unwrapped_positions = np.zeros_like(atoms.positions)

    for bb_id, (mol, offsets) in enumerate(zip(molecule_groups, offset_groups)):
        ref_pos = atoms.positions[mol[0]]
        # Offsets are relative vectors that place all atoms in the same image
        new_positions = ref_pos + offsets
        unwrapped_positions[mol] = new_positions

    # Create a copy so we don't modify the original object in-place
    unwrapped_atoms = atoms.copy()
    unwrapped_atoms.positions = unwrapped_positions
    unwrapped_atoms.center()

    return unwrapped_atoms


def split_by_formula(
    molcrystals: list,
    monomers: list,
    train_size: int = 10000,
    test_size: int = 750,
    seed: int = 42,
) -> tuple:
    """
    Split molecular crystals by unique monomer formulas.

    No formula appears in more than one split, ensuring that the model
    doesn't see the same molecular structure in multiple splits.

    Args:
        molcrystals: List of ASE Atoms objects.
        monomers: List of monomer formula strings.
        train_size: Target number of training samples.
        test_size: Target number of test samples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_inds, val_inds, test_inds) as numpy arrays.
    """
    # Group sample indices by formula
    formula_to_indices = defaultdict(list)
    for i, formula in enumerate(monomers):
        formula_to_indices[formula].append(i)

    # Shuffle unique formulas
    unique_formulas = np.array(list(formula_to_indices.keys()))
    np.random.seed(seed)
    np.random.shuffle(unique_formulas)

    # Allocate formulas greedily to reach target sample counts
    train_formulas, val_formulas, test_formulas = [], [], []
    n_train, n_val, n_test = 0, 0, 0

    for f in unique_formulas:
        count = len(formula_to_indices[f])
        if n_train + count <= train_size:
            train_formulas.append(f)
            n_train += count
        elif n_test + count <= test_size:
            test_formulas.append(f)
            n_test += count
        else:
            val_formulas.append(f)
            n_val += count

    # Collect indices for each split
    train_inds = np.concatenate([formula_to_indices[f] for f in train_formulas]) if train_formulas else np.array([], dtype=int)
    val_inds = np.concatenate([formula_to_indices[f] for f in val_formulas]) if val_formulas else np.array([], dtype=int)
    test_inds = np.concatenate([formula_to_indices[f] for f in test_formulas]) if test_formulas else np.array([], dtype=int)

    return train_inds, val_inds, test_inds, train_formulas, val_formulas, test_formulas


def unwrap_and_split(
    input_path: str,
    output_dir: str,
    coarse_grained_path: str = None,
    monomers_path: str = None,
    train_size: int = 10000,
    test_size: int = 750,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Unwrap molecules and split data into train/validation/test sets.

    Args:
        input_path: Path to input extended XYZ file (with bb_indices).
        output_dir: Directory to save output files.
        coarse_grained_path: Optional path to coarse-grained structures.
        monomers_path: Optional path to monomer formulas pickle file.
        train_size: Target number of training samples.
        test_size: Target number of test samples.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress information.

    Returns:
        Dictionary with split statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read all structures
    if verbose:
        print(f"Reading structures from {input_path}")
    molcrystals = read(input_path, index=":", format="extxyz")
    if verbose:
        print(f"Loaded {len(molcrystals)} structures")

    # Load coarse-grained structures if provided
    cg_molcrystals = None
    if coarse_grained_path is not None:
        if verbose:
            print(f"Reading coarse-grained structures from {coarse_grained_path}")
        cg_molcrystals = read(coarse_grained_path, index=":", format="extxyz")

    # Get monomer formulas
    if monomers_path is not None:
        if verbose:
            print(f"Loading monomer formulas from {monomers_path}")
        with open(monomers_path, "rb") as f:
            monomers = pickle.load(f)
    else:
        # Extract monomer formulas from coarse-grained info or structure info
        monomers = []
        for atoms in molcrystals:
            if "monomer" in atoms.info:
                monomers.append(atoms.info["monomer"])
            else:
                # Fall back: no monomer info available
                monomers.append("unknown")
        if verbose and monomers[0] == "unknown":
            print("Warning: No monomer formulas found. All structures will be treated as same formula.")

    assert len(monomers) == len(molcrystals), "Mismatch between monomers and structures"

    # Step 1: Unwrap molecules
    if verbose:
        print("Unwrapping molecules across periodic boundaries...")
        iterator = tqdm(molcrystals, desc="Unwrapping")
    else:
        iterator = molcrystals

    unwrapped_molcrystals = [unwrap_building_blocks(atoms) for atoms in iterator]

    # Save unwrapped structures
    unwrapped_path = output_dir / "unwrapped_molcrystals.extxyz"
    write(str(unwrapped_path), unwrapped_molcrystals, format="extxyz")
    if verbose:
        print(f"Saved unwrapped structures to {unwrapped_path}")

    # Step 2: Split by formula
    if verbose:
        print(f"\nSplitting data (train_size={train_size}, test_size={test_size}, seed={seed})...")

    train_inds, val_inds, test_inds, train_formulas, val_formulas, test_formulas = split_by_formula(
        unwrapped_molcrystals, monomers, train_size=train_size, test_size=test_size, seed=seed
    )

    # Convert to numpy arrays for indexing
    unwrapped_arr = np.array(unwrapped_molcrystals, dtype=object)
    monomers_arr = np.array(monomers)

    # Create splits
    train_data = unwrapped_arr[train_inds]
    val_data = unwrapped_arr[val_inds]
    test_data = unwrapped_arr[test_inds]

    # Save split indices
    np.save(str(output_dir / "train_inds.npy"), train_inds)
    np.save(str(output_dir / "val_inds.npy"), val_inds)
    np.save(str(output_dir / "test_inds.npy"), test_inds)

    # Save split XYZ files
    if len(train_data) > 0:
        write(str(output_dir / "train_molcrystal.extxyz"), list(train_data), format="extxyz")
    if len(val_data) > 0:
        write(str(output_dir / "val_molcrystal.extxyz"), list(val_data), format="extxyz")
    if len(test_data) > 0:
        write(str(output_dir / "test_molcrystal.extxyz"), list(test_data), format="extxyz")

    # Save coarse-grained splits if available
    if cg_molcrystals is not None:
        cg_arr = np.array(cg_molcrystals, dtype=object)
        if len(train_inds) > 0:
            write(str(output_dir / "train_cg_molcrystal.extxyz"), list(cg_arr[train_inds]), format="extxyz")
        if len(val_inds) > 0:
            write(str(output_dir / "val_cg_molcrystal.extxyz"), list(cg_arr[val_inds]), format="extxyz")
        if len(test_inds) > 0:
            write(str(output_dir / "test_cg_molcrystal.extxyz"), list(cg_arr[test_inds]), format="extxyz")

    # Save monomer formulas for each split
    with open(str(output_dir / "train_monomers.pkl"), "wb") as f:
        pickle.dump(list(monomers_arr[train_inds]), f)
    with open(str(output_dir / "val_monomers.pkl"), "wb") as f:
        pickle.dump(list(monomers_arr[val_inds]), f)
    with open(str(output_dir / "test_monomers.pkl"), "wb") as f:
        pickle.dump(list(monomers_arr[test_inds]), f)

    # Statistics
    stats = {
        "total": len(molcrystals),
        "train": len(train_inds),
        "val": len(val_inds),
        "test": len(test_inds),
        "train_formulas": len(train_formulas),
        "val_formulas": len(val_formulas),
        "test_formulas": len(test_formulas),
        "unique_formulas": len(set(monomers)),
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Train: {stats['train']} samples ({stats['train_formulas']} unique formulas)")
        print(f"  Val:   {stats['val']} samples ({stats['val_formulas']} unique formulas)")
        print(f"  Test:  {stats['test']} samples ({stats['test_formulas']} unique formulas)")
        print(f"\nSaved files to {output_dir}:")
        print(f"  - train_molcrystal.extxyz, val_molcrystal.extxyz, test_molcrystal.extxyz")
        print(f"  - train_inds.npy, val_inds.npy, test_inds.npy")
        print(f"  - train_monomers.pkl, val_monomers.pkl, test_monomers.pkl")
        if cg_molcrystals is not None:
            print(f"  - train_cg_molcrystal.extxyz, val_cg_molcrystal.extxyz, test_cg_molcrystal.extxyz")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Unwrap molecules and split data into train/validation/test sets."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input extended XYZ file (output from filter_structures.py)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="./splits",
        help="Directory to save output files (default: ./splits)",
    )
    parser.add_argument(
        "--coarse_grained",
        "-c",
        type=str,
        default=None,
        help="Optional path to coarse-grained structures file",
    )
    parser.add_argument(
        "--monomers",
        "-m",
        type=str,
        default=None,
        help="Optional path to monomer formulas pickle file",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Target number of training samples (default: 10000)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=750,
        help="Target number of test samples (default: 750)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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

    unwrap_and_split(
        input_path=str(input_path),
        output_dir=args.output_dir,
        coarse_grained_path=args.coarse_grained,
        monomers_path=args.monomers,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
