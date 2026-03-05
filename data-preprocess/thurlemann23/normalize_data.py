#!/usr/bin/env python
"""
Normalize training pkl.gz files for consistent local coordinate frames.

This script applies normalization to ensure consistent local coordinate frames
across building blocks within each structure:
1. For blocks with the same axis flip pattern, align first atoms to reference block
2. Detect 2-axis flips and apply 180° rotation about remaining axis

The output files (*_normalized.pkl.gz) are the final training-ready data files.

Based on renormalize_data.ipynb.

Usage:
    # Single file
    python normalize_data.py --input train_molcrystal.pkl.gz --output train_molcrystal_normalized.pkl.gz

    # Batch mode (all splits)
    python normalize_data.py --input_dir ./processed --output_dir ./normalized

    # In-place normalization (adds _normalized suffix)
    python normalize_data.py --input_dir ./processed
"""
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from common import (
    load_pkl_gz,
    renormalize_single_datapoint,
    rotation_180_about_axis,
    save_pkl_gz,
)


def normalize_dataset(
    input_path: str,
    output_path: str,
    verbose: bool = True,
) -> int:
    """
    Normalize all structures in a pkl.gz file.
    
    Args:
        input_path: Path to input pkl.gz file.
        output_path: Path to output pkl.gz file.
        verbose: Whether to print progress.
        
    Returns:
        Number of structures normalized.
    """
    if verbose:
        print(f"Loading {input_path}...")
    
    processed_data = load_pkl_gz(input_path)
    
    if verbose:
        print(f"Loaded {len(processed_data)} structures")
    
    # Normalize each structure
    iterator = tqdm(range(len(processed_data)), desc="Normalizing") if verbose else range(len(processed_data))
    
    for i in iterator:
        processed_data[i] = renormalize_single_datapoint(processed_data[i])
    
    # Save normalized data
    if verbose:
        print(f"Saving to {output_path}...")
    
    save_pkl_gz(processed_data, output_path)
    
    if verbose:
        print(f"Saved {len(processed_data)} normalized structures")
    
    return len(processed_data)


def normalize_all_splits(
    input_dir: str,
    output_dir: str = None,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Normalize all split files (train, val, test).
    
    Args:
        input_dir: Directory containing pkl.gz files.
        output_dir: Directory to save normalized files. If None, saves in input_dir.
        verbose: Whether to print progress.
        
    Returns:
        Dictionary with counts per split.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    stats = {}
    
    for split in splits:
        # Find input file
        input_patterns = [
            f"{split}_molcrystal.pkl.gz",
            f"{split}.pkl.gz",
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
        
        # Determine output filename
        output_file = output_dir / f"{split}_molcrystal_normalized.pkl.gz"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {split} split")
            print(f"  Input: {input_file}")
            print(f"  Output: {output_file}")
        
        n_processed = normalize_dataset(
            input_path=str(input_file),
            output_path=str(output_file),
            verbose=verbose,
        )
        
        stats[split] = n_processed
    
    if verbose:
        print(f"\n{'='*60}")
        print("Summary:")
        for split, count in stats.items():
            print(f"  {split}: {count} structures normalized")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Normalize training pkl.gz files for consistent local coordinate frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script is the final step in data preprocessing. The output files
(*_normalized.pkl.gz) are ready for training MolCrystalFlow models.

Examples:
  # Single file
  python normalize_data.py --input train_molcrystal.pkl.gz --output train_normalized.pkl.gz

  # Batch mode (all splits)
  python normalize_data.py --input_dir ./processed --output_dir ./normalized

  # In-place normalization (creates *_normalized.pkl.gz in same directory)
  python normalize_data.py --input_dir ./processed
        """,
    )
    
    # Single file mode
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input pkl.gz file (single file mode)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output pkl.gz file (single file mode)",
    )
    
    # Batch mode
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing pkl.gz files (batch mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save normalized files (batch mode). If not specified, uses input_dir.",
    )
    
    # Common options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Single file mode
    if args.input is not None:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Default output path
        if args.output is None:
            output_path = input_path.parent / f"{input_path.stem}_normalized.pkl.gz"
        else:
            output_path = Path(args.output)
        
        normalize_dataset(
            input_path=str(input_path),
            output_path=str(output_path),
            verbose=not args.quiet,
        )
    
    # Batch mode
    elif args.input_dir is not None:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        normalize_all_splits(
            input_dir=str(input_dir),
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )
    
    else:
        parser.error("Must specify either --input or --input_dir")


if __name__ == "__main__":
    main()
