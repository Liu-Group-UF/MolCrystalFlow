#!/usr/bin/env python
"""
Generate training-ready pkl.gz files from preprocessed XYZ files.

This script converts standardized XYZ files (with bb_indices) into compressed
pickle files ready for training. It follows the same preprocessing logic as
data/dataset.py:preprocess_all_structures.

Usage:
    # Single file
    python generate_training_data.py \
        --input standardized_train.extxyz \
        --output train_molcrystal.pkl.gz \
        --features agg_extra_features.pkl \
        --indices train_inds.npy

    # Batch mode (all splits)
    python generate_training_data.py \
        --input_dir ./standardized \
        --output_dir ./processed \
        --features ./features/agg_extra_features.pkl \
        --indices_dir ./splits
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from ase.data import atomic_numbers
from tqdm import tqdm

from common import preprocess_structure, save_pkl_gz


def preprocess_all_structures(xyz_file, output_file, pkl_file=None,
                              existing_inds=None, verbose=True):
    """
    Preprocess all structures in an XYZ file and save to compressed pickle.

    Args:
        xyz_file: Path to the XYZ file containing multiple structures.
        output_file: Path to save the processed data (.pkl.gz).
        pkl_file: Optional path to pickle file with extra features.
        existing_inds: Optional indices mapping local to global indices.
        verbose: Whether to print progress.
    """
    processed_data = []

    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    # Load extra features if provided
    full_basic_chunks = None
    full_chemical_chunks = None
    full_geometric_chunks = None

    if pkl_file is not None:
        with open(pkl_file, 'rb') as pf:
            extra_features = pickle.load(pf)
        basic_feats = extra_features['basic_features']
        chemical_feats = extra_features['chemical_features']
        geometric_feats = extra_features['geometric_features']
        num_bbs = np.int_(extra_features['num_bbs'])
        split_indices = np.cumsum(num_bbs)[:-1]
        full_basic_chunks = np.split(basic_feats, split_indices)
        full_chemical_chunks = np.split(chemical_feats, split_indices)
        full_geometric_chunks = np.split(geometric_feats, split_indices)

    # First pass: count structures for progress bar
    i = 0
    n_structures = 0
    while i < len(lines):
        try:
            n_atoms = int(lines[i].strip())
            n_structures += 1
            i += n_atoms + 2
        except:
            break

    if verbose:
        print(f"Found {n_structures} structures in {xyz_file}")

    # Second pass: process structures
    i = 0
    iterator = tqdm(range(n_structures), desc="Processing") if verbose else range(n_structures)

    for _ in iterator:
        try:
            n_atoms = int(lines[i].strip())
            comment = lines[i + 1].strip()

            # Extract lattice matrix from comment line
            lattice_str = comment.split('Lattice="')[1].split('"')[0]
            lattice_matrix = np.array([float(x) for x in lattice_str.split()]).reshape(3, 3)

            # Read atomic positions and elements
            atom_positions = []
            atom_elements = []
            bb_indices = []

            for j in range(n_atoms):
                parts = lines[i + 2 + j].strip().split()
                atom_elements.append(parts[0])
                atom_positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                bb_indices.append(int(parts[4]))

            atom_positions = np.array(atom_positions)
            bb_indices = np.array(bb_indices)
            atom_types = np.array([atomic_numbers[element] for element in atom_elements])

            feats = preprocess_structure(lattice_matrix, atom_positions, atom_types, bb_indices)
            processed_data.append(feats)

            i += n_atoms + 2

        except Exception as e:
            if verbose:
                tqdm.write(f"Error at line {i}: {e}")
            break

    # Add extra features if provided
    indices_to_delete = []
    if existing_inds is not None and pkl_file is not None:
        for local_idx, global_idx in enumerate(existing_inds):
            if local_idx >= len(processed_data):
                break

            basic_tensor = torch.tensor(full_basic_chunks[global_idx]).float()
            chemical_tensor = torch.tensor(full_chemical_chunks[global_idx]).float()
            geometric_tensor = torch.tensor(full_geometric_chunks[global_idx]).float()

            try:
                assert basic_tensor.shape[0] == processed_data[local_idx]['trans_1'].shape[0]
                processed_data[local_idx].update({
                    "basic_features": basic_tensor,
                    "chemical_features": chemical_tensor,
                    "geometric_features": geometric_tensor,
                })
            except AssertionError:
                if verbose:
                    print(f"Shape mismatch at idx {local_idx}: "
                          f"{basic_tensor.shape[0]} vs {processed_data[local_idx]['trans_1'].shape[0]}")
                indices_to_delete.append(local_idx)

    # Delete problematic entries in reverse order
    for idx in sorted(indices_to_delete, reverse=True):
        del processed_data[idx]

    if verbose and indices_to_delete:
        print(f"Removed {len(indices_to_delete)} entries due to shape mismatches")

    # Save to compressed pickle
    if verbose:
        print(f"Saving {len(processed_data)} structures to {output_file}")

    save_pkl_gz(processed_data, output_file)

    return len(processed_data)


def process_all_splits(input_dir, output_dir, features_pkl=None, indices_dir=None, verbose=True):
    """
    Process all splits (train, val, test) in batch mode.

    Args:
        input_dir: Directory containing standardized XYZ files.
        output_dir: Directory to save output pkl.gz files.
        features_pkl: Optional path to auxiliary features pickle.
        indices_dir: Optional directory containing *_inds.npy files.
        verbose: Whether to print progress.

    Returns:
        Dictionary with processing statistics.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'val', 'test']
    stats = {}

    for split in splits:
        # Find input file
        input_patterns = [
            f"standardized_{split}_molcrystal.extxyz",
            f"{split}_molcrystal.extxyz",
        ]

        input_file = None
        for pattern in input_patterns:
            candidate = input_dir / pattern
            if candidate.exists():
                input_file = candidate
                break

        if input_file is None:
            if verbose:
                print(f"No input file found for {split} split, skipping...")
            continue

        output_file = output_dir / f"{split}_molcrystal.pkl.gz"

        # Find indices file
        existing_inds = None
        if indices_dir is not None:
            indices_patterns = [f"{split}_inds.npy", f"valid_inds.npy" if split == "val" else None]
            for pattern in indices_patterns:
                if pattern is None:
                    continue
                candidate = Path(indices_dir) / pattern
                if candidate.exists():
                    existing_inds = np.load(candidate)
                    break

        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {split} split")
            print(f"  Input: {input_file}")
            print(f"  Output: {output_file}")
            if existing_inds is not None:
                print(f"  Indices: {len(existing_inds)} entries")

        n_processed = preprocess_all_structures(
            xyz_file=str(input_file),
            output_file=str(output_file),
            pkl_file=features_pkl,
            existing_inds=existing_inds,
            verbose=verbose,
        )

        stats[split] = n_processed

    if verbose:
        print(f"\n{'='*60}")
        print("Summary:")
        for split, count in stats.items():
            print(f"  {split}: {count} structures")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate training-ready pkl.gz files from preprocessed XYZ files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode
  python generate_training_data.py \\
      --input standardized_train.extxyz \\
      --output train_molcrystal.pkl.gz \\
      --features agg_extra_features.pkl \\
      --indices train_inds.npy

  # Batch mode (all splits)
  python generate_training_data.py \\
      --input_dir ./standardized \\
      --output_dir ./processed \\
      --features ./features/agg_extra_features.pkl \\
      --indices_dir ./splits
        """
    )

    # Single file mode
    parser.add_argument("--input", "-i", type=str,
                        help="Path to input XYZ file (single file mode)")
    parser.add_argument("--output", "-o", type=str,
                        help="Path to output pkl.gz file (single file mode)")

    # Batch mode
    parser.add_argument("--input_dir", type=str,
                        help="Directory containing standardized XYZ files (batch mode)")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save output pkl.gz files (batch mode)")

    # Common options
    parser.add_argument("--features", "-f", type=str, default=None,
                        help="Path to auxiliary features pickle (agg_extra_features.pkl)")
    parser.add_argument("--indices", type=str, default=None,
                        help="Path to indices npy file (single file mode)")
    parser.add_argument("--indices_dir", type=str, default=None,
                        help="Directory containing *_inds.npy files (batch mode)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    # Single file mode
    if args.input is not None and args.output is not None:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        existing_inds = None
        if args.indices is not None:
            existing_inds = np.load(args.indices)

        preprocess_all_structures(
            xyz_file=str(input_path),
            output_file=args.output,
            pkl_file=args.features,
            existing_inds=existing_inds,
            verbose=not args.quiet,
        )

    # Batch mode
    elif args.input_dir is not None and args.output_dir is not None:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)

        process_all_splits(
            input_dir=str(input_dir),
            output_dir=args.output_dir,
            features_pkl=args.features,
            indices_dir=args.indices_dir,
            verbose=not args.quiet,
        )

    else:
        parser.error("Must specify either (--input and --output) or (--input_dir and --output_dir)")


if __name__ == "__main__":
    main()
