#!/usr/bin/env python
"""
Standardize lattice vectors for molecular crystal structures.

This script:
1. Sorts lattice vectors so that |a| <= |b| <= |c|
2. Transforms to lower-triangular form with positive diagonal
3. Consistently rotates atomic positions

Usage:
    python standardize_lattice.py --input train_molcrystal.extxyz --output standardized_train.extxyz
    python standardize_lattice.py --input_dir ./splits --output_dir ./standardized
"""
import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write
from tqdm import tqdm


def sort_cell_left_matrix(atoms):
    """
    Standardize the unit cell lattice vectors.

    This function:
    1. Sorts lattice vectors so that |a| <= |b| <= |c|
    2. Transforms to lower-triangular form via QR decomposition
    3. Ensures positive diagonal elements
    4. Consistently rotates atomic positions

    Args:
        atoms: ASE Atoms object.

    Returns:
        Modified Atoms object with standardized cell.
    """
    # ASE cell: 3x3, rows are lattice vectors a, b, c (left-matrix convention)
    L = atoms.cell.array.copy()
    lengths = np.linalg.norm(L, axis=1)  # |a|, |b|, |c|
    order = np.argsort(lengths)  # ensure a <= b <= c
    Lp = L[order, :]

    # Compute orthogonal Q so L_lower = Lp @ Q is lower-triangular with positive diagonal
    Q, R = np.linalg.qr(Lp.T)  # Lp^T = Q R  => L_lower = R^T
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    D = np.diag(s)
    Q = Q @ D
    R = D @ R

    L_lower = R.T
    L_lower[np.triu_indices(3, k=1)] = 0.0  # clean numerical noise above diagonal

    # Rotate Cartesian positions consistently with the cell transformation
    X = atoms.get_positions()
    X_rot = X @ Q

    # Apply new cell and positions; do not scale or wrap atoms; leave pbc unchanged
    atoms.set_cell(L_lower, scale_atoms=False)
    atoms.set_positions(X_rot)

    return atoms


def standardize_file(input_path: str, output_path: str, verbose: bool = True) -> int:
    """
    Standardize lattice vectors for all structures in an XYZ file.

    Args:
        input_path: Path to input extended XYZ file.
        output_path: Path to output extended XYZ file.
        verbose: Whether to print progress information.

    Returns:
        Number of structures processed.
    """
    if verbose:
        print(f"Reading structures from {input_path}")

    images = read(input_path, index=":")

    if verbose:
        print(f"Standardizing {len(images)} structures...")
        iterator = tqdm(images, desc="Standardizing")
    else:
        iterator = images

    for atoms in iterator:
        sort_cell_left_matrix(atoms)

    write(output_path, images, format="extxyz")

    if verbose:
        print(f"Saved standardized structures to {output_path}")

    return len(images)


def standardize_directory(
    input_dir: str,
    output_dir: str,
    pattern: str = "*_molcrystal.extxyz",
    verbose: bool = True,
) -> dict:
    """
    Standardize lattice vectors for all XYZ files matching a pattern.

    Args:
        input_dir: Directory containing input XYZ files.
        output_dir: Directory to save output XYZ files.
        pattern: Glob pattern for input files.
        verbose: Whether to print progress information.

    Returns:
        Dictionary with processing statistics per file.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(input_dir.glob(pattern))

    if not input_files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return {}

    if verbose:
        print(f"Found {len(input_files)} files to process")

    stats = {}
    for input_path in input_files:
        output_path = output_dir / f"standardized_{input_path.name}"
        n_processed = standardize_file(str(input_path), str(output_path), verbose=verbose)
        stats[input_path.name] = n_processed

    if verbose:
        print(f"\nProcessed {len(stats)} files:")
        for name, count in stats.items():
            print(f"  {name}: {count} structures")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Standardize lattice vectors for molecular crystal structures."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input extended XYZ file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output extended XYZ file",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing input XYZ files (for batch processing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output XYZ files (for batch processing)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_molcrystal.extxyz",
        help="Glob pattern for input files when using --input_dir (default: *_molcrystal.extxyz)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.input is not None and args.output is not None:
        # Single file mode
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        standardize_file(
            input_path=str(input_path),
            output_path=args.output,
            verbose=not args.quiet,
        )

    elif args.input_dir is not None and args.output_dir is not None:
        # Batch mode
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        standardize_directory(
            input_dir=str(input_dir),
            output_dir=args.output_dir,
            pattern=args.pattern,
            verbose=not args.quiet,
        )

    else:
        parser.error(
            "Must specify either (--input and --output) or (--input_dir and --output_dir)"
        )


if __name__ == "__main__":
    main()
