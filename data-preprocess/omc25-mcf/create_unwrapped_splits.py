#!/usr/bin/env python
"""
Create train/val/test splits from the unwrapped (but not standardized) OMC25-MCF data.

This script takes the unwrapped_molcrystals.extxyz file and creates splits
using the same indices as the standardized splits, ensuring consistency.

Usage:
    python create_unwrapped_splits.py
    python create_unwrapped_splits.py --output_dir ./preprocessed/unwrapped_splits
    python create_unwrapped_splits.py --fresh_split  # Create new random split
"""
import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write


def create_splits_from_indices(
    unwrapped_path: str,
    indices_dir: str,
    output_dir: str,
    verbose: bool = True,
):
    """
    Create train/val/test splits from unwrapped data using existing indices.
    
    Args:
        unwrapped_path: Path to unwrapped_molcrystals.extxyz
        indices_dir: Directory containing train_inds.npy, val_inds.npy, test_inds.npy
        output_dir: Output directory for the splits
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Loading unwrapped structures from {unwrapped_path}...")
    
    images = read(unwrapped_path, index=":")
    n_total = len(images)
    
    if verbose:
        print(f"  Loaded {n_total} structures")
    
    # Load existing indices
    indices_dir = Path(indices_dir)
    train_inds = np.load(indices_dir / "train_inds.npy")
    val_inds = np.load(indices_dir / "val_inds.npy")
    test_inds = np.load(indices_dir / "test_inds.npy")
    
    if verbose:
        print(f"  Using existing indices: train={len(train_inds)}, val={len(val_inds)}, test={len(test_inds)}")
    
    # Validate indices
    all_inds = np.concatenate([train_inds, val_inds, test_inds])
    if len(all_inds) != n_total:
        print(f"  WARNING: Index count ({len(all_inds)}) != structure count ({n_total})")
        print(f"  This may happen if the unwrapped data differs from the standardized data.")
    
    # Create splits
    for split_name, split_inds in [("train", train_inds), ("val", val_inds), ("test", test_inds)]:
        # Filter valid indices (in case unwrapped has fewer structures)
        valid_inds = split_inds[split_inds < n_total]
        
        if len(valid_inds) < len(split_inds):
            print(f"  WARNING: {split_name}: using {len(valid_inds)}/{len(split_inds)} indices (some out of range)")
        
        split_images = [images[i] for i in valid_inds]
        split_file = output_dir / f"{split_name}_unwrapped.extxyz"
        write(str(split_file), split_images, format="extxyz")
        
        # Also save the indices used
        np.save(str(output_dir / f"{split_name}_inds.npy"), valid_inds)
        
        if verbose:
            print(f"  Saved {len(split_images)} structures to {split_file}")
    
    if verbose:
        print(f"\nDone! Unwrapped splits saved to {output_dir}")


def create_fresh_splits(
    unwrapped_path: str,
    output_dir: str,
    test_size: int = 1000,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Create fresh train/val/test splits from unwrapped data.
    
    Args:
        unwrapped_path: Path to unwrapped_molcrystals.extxyz
        output_dir: Output directory for the splits
        test_size: Number of test samples
        seed: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Loading unwrapped structures from {unwrapped_path}...")
    
    images = read(unwrapped_path, index=":")
    n_total = len(images)
    
    if verbose:
        print(f"  Loaded {n_total} structures")
    
    # Calculate split sizes: 90% train, test_size test, rest val
    n_train = int(n_total * 0.9)
    n_test = min(test_size, n_total - n_train)
    n_val = n_total - n_train - n_test
    
    if verbose:
        print(f"  Split sizes: train={n_train}, val={n_val}, test={n_test}")
    
    # Shuffle indices
    np.random.seed(seed)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    # Split
    train_inds = indices[:n_train]
    val_inds = indices[n_train:n_train + n_val]
    test_inds = indices[n_train + n_val:]
    
    # Create splits
    for split_name, split_inds in [("train", train_inds), ("val", val_inds), ("test", test_inds)]:
        split_images = [images[i] for i in split_inds]
        split_file = output_dir / f"{split_name}_unwrapped.extxyz"
        write(str(split_file), split_images, format="extxyz")
        np.save(str(output_dir / f"{split_name}_inds.npy"), split_inds)
        
        if verbose:
            print(f"  Saved {len(split_images)} structures to {split_file}")
    
    if verbose:
        print(f"\nDone! Fresh unwrapped splits saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from unwrapped OMC25-MCF data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing split indices (default)
  python create_unwrapped_splits.py

  # Create fresh random splits
  python create_unwrapped_splits.py --fresh_split

  # Custom output directory
  python create_unwrapped_splits.py --output_dir ./my_splits

  # Custom paths
  python create_unwrapped_splits.py \\
      --unwrapped_path ./preprocessed/unwrapped/unwrapped_molcrystals.extxyz \\
      --indices_dir ./preprocessed/splits \\
      --output_dir ./preprocessed/unwrapped_splits
        """,
    )
    
    parser.add_argument(
        "--unwrapped_path",
        type=str,
        default="./preprocessed/unwrapped/unwrapped_molcrystals.extxyz",
        help="Path to unwrapped structures extxyz file",
    )
    parser.add_argument(
        "--indices_dir",
        type=str,
        default="./preprocessed/splits",
        help="Directory containing existing split indices (train_inds.npy, etc.)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./preprocessed/unwrapped_splits",
        help="Output directory for unwrapped splits",
    )
    parser.add_argument(
        "--fresh_split",
        action="store_true",
        help="Create fresh random splits instead of using existing indices",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Number of test samples (only for --fresh_split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (only for --fresh_split)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CREATE UNWRAPPED SPLITS FOR OMC25-MCF")
    print("=" * 60)
    
    if args.fresh_split:
        print("\nCreating fresh random splits...")
        create_fresh_splits(
            unwrapped_path=args.unwrapped_path,
            output_dir=args.output_dir,
            test_size=args.test_size,
            seed=args.seed,
            verbose=not args.quiet,
        )
    else:
        print("\nUsing existing split indices...")
        create_splits_from_indices(
            unwrapped_path=args.unwrapped_path,
            indices_dir=args.indices_dir,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
