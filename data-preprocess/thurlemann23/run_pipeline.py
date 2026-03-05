#!/usr/bin/env python
"""
All-in-one preprocessing pipeline for Thurlemann2023 molecular crystal dataset.

This script runs the complete preprocessing pipeline from raw HDF5 file to
final training-ready pkl.gz files, organizing all intermediate outputs into
a structured directory hierarchy.

Directory Structure:
    output_dir/
    ├── xyz/
    │   ├── molecule_crystals.extxyz      # Raw converted from HDF5
    │   ├── valid_molcrystals.extxyz      # Filtered and validated
    │   ├── coarse_grained.extxyz         # Coarse-grained representations
    │   ├── all_monomers.xyz              # Extracted monomers (standard xyz for obabel)
    │   ├── reconstructed_train.xyz       # Reconstructed from pkl.gz (for verification)
    │   ├── reconstructed_val.xyz
    │   └── reconstructed_test.xyz
    ├── splits/
    │   ├── train_molcrystal.extxyz       # Train split (unwrapped)
    │   ├── val_molcrystal.extxyz         # Validation split
    │   ├── test_molcrystal.extxyz        # Test split
    │   ├── train_inds.npy                # Indices into original dataset
    │   ├── val_inds.npy
    │   └── test_inds.npy
    ├── standardized/
    │   ├── train_molcrystal.extxyz       # Lattice-standardized
    │   ├── val_molcrystal.extxyz
    │   └── test_molcrystal.extxyz
    ├── sdf/
    │   ├── all_monomers.sdf              # Raw SDF from obabel
    │   ├── all_cleaned_monomers.sdf      # Bond-fixed SDF
    │   ├── smiles_list.pkl               # SMILES strings
    │   ├── valid_indices.npy             # Valid molecule indices
    │   └── invalid_indices.npy           # Invalid molecule indices
    ├── features/
    │   ├── extra_molcrystal_features.pkl # Raw features per molecule
    │   └── agg_extra_features.pkl        # Aggregated features
    ├── final/
    │   ├── train_molcrystal.pkl.gz       # Pre-normalization training data
    │   ├── val_molcrystal.pkl.gz         # Pre-normalization validation data
    │   └── test_molcrystal.pkl.gz        # Pre-normalization test data
    └── normalized/
        ├── train_molcrystal_normalized.pkl.gz  # FINAL training data
        ├── val_molcrystal_normalized.pkl.gz    # FINAL validation data
        └── test_molcrystal_normalized.pkl.gz   # FINAL test data

Usage:
    python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed
    python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed --train_size 10000 --test_size 750
    python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed --skip_to step4
    python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed --skip_to step7
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd, description, verbose=True):
    """Run a shell command and handle errors."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Step: {description}")
        print(f"Command: {' '.join(cmd)}")
        print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    if verbose and result.stdout:
        print(result.stdout)
    
    return True


def run_python_script(script_path, args, description, verbose=True):
    """Run a Python script with arguments."""
    cmd = [sys.executable, str(script_path)] + args
    return run_command(cmd, description, verbose)


def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if Path(filepath).exists():
        size = Path(filepath).stat().st_size
        print(f"  ✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"  ✗ {description}: {filepath} NOT FOUND")
        return False


def run_pipeline(
    input_hdf5: str,
    output_dir: str,
    train_size: int = 10000,
    test_size: int = 750,
    seed: int = 42,
    skip_to: str = None,
    verbose: bool = True,
):
    """
    Run the complete preprocessing pipeline.
    
    Args:
        input_hdf5: Path to input HDF5 file.
        output_dir: Base output directory.
        train_size: Number of training samples.
        test_size: Number of test samples.
        seed: Random seed for reproducibility.
        skip_to: Skip to a specific step (step1-step7).
        verbose: Whether to print progress.
    """
    # Get script directory
    script_dir = Path(__file__).parent.resolve()
    
    # Create output directories
    output_dir = Path(output_dir).resolve()
    xyz_dir = output_dir / "xyz"
    splits_dir = output_dir / "splits"
    standardized_dir = output_dir / "standardized"
    sdf_dir = output_dir / "sdf"
    features_dir = output_dir / "features"
    final_dir = output_dir / "final"
    normalized_dir = output_dir / "normalized"
    
    for d in [xyz_dir, splits_dir, standardized_dir, sdf_dir, features_dir, final_dir, normalized_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Define step order
    steps = ["step1", "step2", "step3", "step4", "step5", "step6", "step7"]
    start_step = 0
    if skip_to:
        if skip_to in steps:
            start_step = steps.index(skip_to)
            print(f"Skipping to {skip_to}")
        else:
            print(f"Unknown step: {skip_to}. Valid steps: {steps}")
            return False
    
    # Track timing
    start_time = datetime.now()
    
    if verbose:
        print("\n" + "="*60)
        print("MOLECULAR CRYSTAL PREPROCESSING PIPELINE")
        print("="*60)
        print(f"Input HDF5: {input_hdf5}")
        print(f"Output directory: {output_dir}")
        print(f"Train size: {train_size}, Test size: {test_size}")
        print(f"Random seed: {seed}")
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================================
    # Step 1: Convert HDF5 to XYZ
    # =========================================================================
    if start_step <= 0:
        success = run_python_script(
            script_dir / "hdf2xyz.py",
            [
                "--input", str(input_hdf5),
                "--output", str(xyz_dir / "molecule_crystals.extxyz"),
            ],
            "Step 1: Convert HDF5 to XYZ",
            verbose
        )
        if not success:
            return False
    
    # =========================================================================
    # Step 2: Filter and validate structures
    # =========================================================================
    if start_step <= 1:
        success = run_python_script(
            script_dir / "filter_structures.py",
            [
                "--input", str(xyz_dir / "molecule_crystals.extxyz"),
                "--output", str(xyz_dir / "valid_molcrystals.extxyz"),
                "--coarse_grained", str(xyz_dir / "coarse_grained.extxyz"),
                "--monomers", str(xyz_dir / "all_monomers.xyz"),
                "--bad_indices", str(xyz_dir / "bad_indices.npy"),
            ],
            "Step 2: Filter and validate structures",
            verbose
        )
        if not success:
            return False
    
    # =========================================================================
    # Step 3: Unwrap molecules and split data
    # =========================================================================
    if start_step <= 2:
        success = run_python_script(
            script_dir / "unwrap_and_split.py",
            [
                "--input", str(xyz_dir / "valid_molcrystals.extxyz"),
                "--output_dir", str(splits_dir),
                "--coarse_grained", str(xyz_dir / "coarse_grained.extxyz"),
                "--train_size", str(train_size),
                "--test_size", str(test_size),
                "--seed", str(seed),
            ],
            "Step 3: Unwrap molecules and split data",
            verbose
        )
        if not success:
            return False
    
    # =========================================================================
    # Step 4: Standardize lattice vectors
    # =========================================================================
    if start_step <= 3:
        success = run_python_script(
            script_dir / "standardize_lattice.py",
            [
                "--input_dir", str(splits_dir),
                "--output_dir", str(standardized_dir),
                "--pattern", "*_molcrystal.extxyz",
            ],
            "Step 4: Standardize lattice vectors",
            verbose
        )
        if not success:
            return False
    
    # =========================================================================
    # Step 5: Generate molecular features (bond fixing + feature extraction)
    # =========================================================================
    if start_step <= 4:
        # Step 5a: Convert XYZ to SDF using obabel
        if verbose:
            print(f"\n{'='*60}")
            print("Step 5a: Convert XYZ to SDF using Open Babel")
            print('='*60)
        
        obabel_cmd = [
            "obabel",
            str(xyz_dir / "all_monomers.xyz"),
            "-O", str(sdf_dir / "all_monomers.sdf"),
        ]
        result = subprocess.run(obabel_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: obabel conversion may have issues: {result.stderr}")
        
        # Step 5b: Fix bond orders
        success = run_python_script(
            script_dir / "fix_bonds.py",
            [
                "--input", str(sdf_dir / "all_monomers.sdf"),
                "--output", str(sdf_dir / "all_cleaned_monomers.sdf"),
                "--save_indices",
            ],
            "Step 5b: Fix bond orders and charges",
            verbose
        )
        if not success:
            print("Warning: Bond fixing had issues, continuing...")
        
        # Step 5c: Generate features
        success = run_python_script(
            script_dir / "generate_features.py",
            [
                "--input", str(xyz_dir / "all_monomers.xyz"),
                "--output_dir", str(features_dir),
                "--coarse_grained", str(xyz_dir / "coarse_grained.extxyz"),
            ],
            "Step 5c: Generate molecular features",
            verbose
        )
        if not success:
            return False
    
    # =========================================================================
    # Step 6: Generate training pkl.gz files (final output)
    # =========================================================================
    if start_step <= 5:
        success = run_python_script(
            script_dir / "generate_training_data.py",
            [
                "--input_dir", str(standardized_dir),
                "--output_dir", str(final_dir),
                "--features", str(features_dir / "agg_extra_features.pkl"),
                "--indices_dir", str(splits_dir),
            ],
            "Step 6: Generate training pkl.gz files (final output)",
            verbose
        )
        if not success:
            return False
    
    # =========================================================================
    # Step 7: Normalize local coordinate frames
    # =========================================================================
    if start_step <= 6:
        success = run_python_script(
            script_dir / "normalize_data.py",
            [
                "--input_dir", str(final_dir),
                "--output_dir", str(normalized_dir),
            ],
            "Step 7: Normalize local coordinate frames",
            verbose
        )
        if not success:
            return False
    
    # =========================================================================
    # Verify: Export reconstructed XYZ files for visual comparison
    # =========================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("Verification: Exporting reconstructed XYZ files")
        print('='*60)

    for split in ["train", "val", "test"]:
        pkl_path = normalized_dir / f"{split}_molcrystal_normalized.pkl.gz"
        xyz_out = xyz_dir / f"reconstructed_{split}.xyz"
        if pkl_path.exists():
            run_python_script(
                script_dir / "check_data.py",
                [
                    "--input", str(pkl_path),
                    "--export_xyz", str(xyz_out),
                    "-q",
                ],
                f"Export reconstructed {split} XYZ",
                verbose,
            )
    
    # =========================================================================
    # Summary
    # =========================================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    if verbose:
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Duration: {duration}")
        print(f"\nOutput files:")
        print("-" * 40)
        
        # Check final outputs
        print("\n[Final Training Data (normalized)]")
        check_file_exists(normalized_dir / "train_molcrystal_normalized.pkl.gz", "Training data")
        check_file_exists(normalized_dir / "val_molcrystal_normalized.pkl.gz", "Validation data")
        check_file_exists(normalized_dir / "test_molcrystal_normalized.pkl.gz", "Test data")
        
        print("\n[Pre-normalization Data]")
        check_file_exists(final_dir / "train_molcrystal.pkl.gz", "Training data (raw)")
        check_file_exists(final_dir / "val_molcrystal.pkl.gz", "Validation data (raw)")
        check_file_exists(final_dir / "test_molcrystal.pkl.gz", "Test data (raw)")
        
        print("\n[Intermediate XYZ Files]")
        check_file_exists(xyz_dir / "molecule_crystals.extxyz", "Raw crystals")
        check_file_exists(xyz_dir / "valid_molcrystals.extxyz", "Valid crystals")
        check_file_exists(xyz_dir / "coarse_grained.extxyz", "Coarse-grained")
        check_file_exists(xyz_dir / "all_monomers.xyz", "Monomers")
        
        print("\n[Reconstructed XYZ (for verification)]")
        check_file_exists(xyz_dir / "reconstructed_train.xyz", "Reconstructed train")
        check_file_exists(xyz_dir / "reconstructed_val.xyz", "Reconstructed val")
        check_file_exists(xyz_dir / "reconstructed_test.xyz", "Reconstructed test")

        print("\n[Split Files]")
        check_file_exists(splits_dir / "train_molcrystal.extxyz", "Train XYZ")
        check_file_exists(splits_dir / "val_molcrystal.extxyz", "Val XYZ")
        check_file_exists(splits_dir / "test_molcrystal.extxyz", "Test XYZ")
        check_file_exists(splits_dir / "train_inds.npy", "Train indices")
        
        print("\n[Feature Files]")
        check_file_exists(features_dir / "agg_extra_features.pkl", "Aggregated features")
        check_file_exists(sdf_dir / "all_cleaned_monomers.sdf", "Cleaned SDF")
        
        print("\n" + "="*60)
        print("To use the final data for training, load:")
        print(f"  {normalized_dir / 'train_molcrystal_normalized.pkl.gz'}")
        print(f"  {normalized_dir / 'val_molcrystal_normalized.pkl.gz'}")
        print(f"  {normalized_dir / 'test_molcrystal_normalized.pkl.gz'}")
        print()
        print("To verify data consistency, compare reconstructed vs standardized XYZ:")
        print(f"  diff <(sort {xyz_dir / 'reconstructed_test.xyz'}) "
              f"<(sort {standardized_dir / 'test_molcrystal.extxyz'})")
        print("  Or load both in OVITO / ASE and visually inspect.")
        print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="All-in-one preprocessing pipeline for Thurlemann2023 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed

  # Custom split sizes
  python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed \\
      --train_size 10000 --test_size 750

  # Resume from a specific step
  python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed \\
      --skip_to step4

Steps:
  step1: Convert HDF5 to XYZ
  step2: Filter and validate structures
  step3: Unwrap molecules and split data
  step4: Standardize lattice vectors
  step5: Generate molecular features (bond fixing + extraction)
  step6: Generate training pkl.gz files (final output)
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input HDF5 file (crystal_dataset.hdf5)",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./preprocessed",
        help="Output directory for all generated files (default: ./preprocessed)",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Number of training samples (default: 10000)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=750,
        help="Number of test samples (default: 750)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip_to",
        type=str,
        default=None,
        choices=["step1", "step2", "step3", "step4", "step5", "step6"],
        help="Skip to a specific step (assumes previous steps completed)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    success = run_pipeline(
        input_hdf5=str(input_path),
        output_dir=args.output_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        skip_to=args.skip_to,
        verbose=not args.quiet,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
