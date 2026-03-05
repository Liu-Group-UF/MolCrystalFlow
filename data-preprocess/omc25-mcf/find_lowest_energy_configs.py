#!/usr/bin/env python
"""
Find the lowest-energy configuration per CSD refcode from OMC25 AseDB splits.

For each CSD refcode the script keeps the structure with the lowest
per-molecule energy (total energy / z_value) among those whose maximum
per-atom force norm is below a threshold.  Train and validation splits are
processed independently and then merged into a single combined extxyz file.

Usage:
    python find_lowest_energy_configs.py \
        --train_dir /path/to/OMC25/train \
        --val_dir   /path/to/OMC25/val \
        --output    omc25_lowest_energy.extxyz

    # Only train split
    python find_lowest_energy_configs.py \
        --train_dir /path/to/OMC25/train \
        --output    omc25_lowest_energy.extxyz

    # Custom force threshold
    python find_lowest_energy_configs.py \
        --train_dir /path/to/OMC25/train \
        --val_dir   /path/to/OMC25/val \
        --output    omc25_lowest_energy.extxyz \
        --fmax 0.1
"""
import argparse
from pathlib import Path

import numpy as np
from ase.io import write
from fairchem.core.datasets import AseDBDataset


def find_lowest_energy(dataset_path: str,
                       fmax_threshold: float = 0.05,
                       verbose: bool = True) -> dict:
    """Scan an AseDB split and keep the lowest per-molecule-energy structure
    per CSD refcode, subject to a maximum force constraint.

    Args:
        dataset_path: Path to the AseDB directory (e.g. ``OMC25/train``).
        fmax_threshold: Maximum allowed per-atom force norm (eV/Å).
        verbose: Print progress every 100 000 structures.

    Returns:
        Dictionary mapping ``csd_refcode`` → ``(atoms, energy_per_mol)``.
    """
    dataset = AseDBDataset({"src": dataset_path})
    n = len(dataset)
    if verbose:
        print(f"  Loaded {n} structures from {dataset_path}")

    best: dict = {}  # csd_refcode -> (atoms, energy_per_mol)

    for i in range(n):
        if verbose and i % 100_000 == 0:
            print(f"    {i:>9d} / {n}")

        atoms = dataset.get_atoms(i)
        csd_code = atoms.info["csd_refcode"]
        z_value = atoms.info["z_value"]
        energy_per_mol = atoms.get_potential_energy() / z_value

        # Check force criterion
        forces = atoms.get_forces()
        max_force = np.linalg.norm(forces, axis=1).max()
        if max_force >= fmax_threshold:
            continue

        # Keep if first seen or lower energy than current best
        if csd_code not in best or energy_per_mol < best[csd_code][1]:
            best[csd_code] = (atoms, energy_per_mol)

    if verbose:
        print(f"  Found {len(best)} unique refcodes passing fmax < {fmax_threshold}")
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Find lowest-energy OMC25 configs per CSD refcode and "
                    "write a combined extxyz file.",
    )
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Path to OMC25 train AseDB directory.")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Path to OMC25 val AseDB directory.")
    parser.add_argument("--output", "-o", type=str,
                        default="omc25_lowest_energy.extxyz",
                        help="Output extxyz path (default: omc25_lowest_energy.extxyz).")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="Max per-atom force norm threshold in eV/Å (default: 0.05).")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output.")
    args = parser.parse_args()

    if args.train_dir is None and args.val_dir is None:
        parser.error("Provide at least one of --train_dir or --val_dir.")

    verbose = not args.quiet
    combined: dict = {}  # csd_refcode -> (atoms, energy_per_mol)

    for label, path in [("train", args.train_dir), ("val", args.val_dir)]:
        if path is None:
            continue
        if not Path(path).exists():
            print(f"WARNING: {label} path does not exist: {path}, skipping.")
            continue
        if verbose:
            print(f"Processing {label} split …")
        best = find_lowest_energy(path, fmax_threshold=args.fmax, verbose=verbose)

        # Merge: keep the lower energy across splits for the same refcode
        for code, (atoms, energy) in best.items():
            if code not in combined or energy < combined[code][1]:
                combined[code] = (atoms, energy)

    if verbose:
        print(f"\nTotal unique refcodes after merging: {len(combined)}")

    # Write output
    structures = [atoms for atoms, _ in combined.values()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), structures)
    if verbose:
        print(f"Saved {len(structures)} structures to {out_path}")


if __name__ == "__main__":
    main()
