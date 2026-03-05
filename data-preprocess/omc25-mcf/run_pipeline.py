#!/usr/bin/env python
"""
All-in-one preprocessing pipeline for OMC25-MCF molecular crystal dataset.

This script runs the complete preprocessing pipeline from raw extxyz file
(omc25_lowest_energy.extxyz) to final training-ready pkl.gz files.

Pipeline steps:
  1. Filter and validate structures (assign bb_indices, extract monomers)
  2. Unwrap molecules and reorder atoms by bb_indices
  3. Standardize lattice vectors
  4. Split into train/val/test (90% train, 1000 test, rest val)
  5. Generate molecular features (bond fixing + extraction)
  6. Generate training pkl.gz files
  7. Normalize local coordinate frames

Directory Structure:
    output_dir/
    ├── filtered/
    │   ├── valid_molcrystals.extxyz     # Structures with bb_indices
    │   ├── coarse_grained.extxyz        # Coarse-grained (one atom per molecule)
    │   └── all_monomers.xyz             # Extracted monomers
    ├── unwrapped/
    │   └── unwrapped_molcrystals.extxyz # Unwrapped and bb_indices-ordered
    ├── standardized/
    │   └── standardized.extxyz          # Lattice-standardized
    ├── splits/
    │   ├── train.extxyz, val.extxyz, test.extxyz
    │   └── train_inds.npy, val_inds.npy, test_inds.npy
    ├── features/
    │   └── agg_extra_features.pkl
    ├── final/
    │   ├── train_molcrystal.pkl.gz
    │   ├── val_molcrystal.pkl.gz
    │   └── test_molcrystal.pkl.gz
    └── normalized/
        ├── train_molcrystal_normalized.pkl.gz
        ├── val_molcrystal_normalized.pkl.gz
        └── test_molcrystal_normalized.pkl.gz

Usage:
    python run_pipeline.py --input omc25_lowest_energy.extxyz --output_dir ./preprocessed

    # Resume from a specific step
    python run_pipeline.py --input ... --output_dir ... --skip_to step3
"""
import argparse
import os
import pickle
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from tqdm import tqdm

# Add thurlemann23 to path for common imports
SCRIPT_DIR = Path(__file__).parent.resolve()
THURLEMANN_DIR = SCRIPT_DIR.parent / "thurlemann23"
sys.path.insert(0, str(THURLEMANN_DIR))

from common import get_molecule_groups


def run_python_script(script_path, args, description, verbose=True):
    """Run a Python script with arguments."""
    cmd = [sys.executable, str(script_path)] + args
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


def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if Path(filepath).exists():
        size = Path(filepath).stat().st_size
        print(f"  ✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"  ✗ {description}: {filepath} NOT FOUND")
        return False


# =============================================================================
# Step 1: Filter and validate structures
# =============================================================================

def atoms_to_formula(atom_list):
    """Convert a list of element symbols to a chemical formula string."""
    element_counts = Counter(atom_list)
    formula = "".join(
        f"{elem}{element_counts[elem] if element_counts[elem] > 1 else ''}"
        for elem in sorted(element_counts.keys(), key=lambda x: (x.islower(), x))
    )
    return formula


def filter_and_validate(
    input_path: str,
    output_dir: str,
    verbose: bool = True,
) -> dict:
    """
    Filter and validate OMC25 structures.
    
    - Detect molecular groups (building blocks)
    - Validate all molecules have same formula
    - Validate number of molecules matches z_value
    - Assign bb_indices to each atom
    - Extract monomers
    
    Returns dict with paths to output files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"  Loading structures from {input_path}...")
    
    all_atoms = read(input_path, index=":")
    
    if verbose:
        print(f"  Loaded {len(all_atoms)} structures")
    
    valid_structures = []
    coarse_grained = []
    all_monomers = []
    bad_indices = []
    
    iterator = tqdm(enumerate(all_atoms), total=len(all_atoms), desc="  Filtering") if verbose else enumerate(all_atoms)
    
    for i, atoms in iterator:
        try:
            # Detect molecular groups
            molecule_groups, offset_groups = get_molecule_groups(atoms)
            
            # Validate: all monomers must have same formula
            formulas = [atoms_to_formula([atoms[j].symbol for j in mol]) for mol in molecule_groups]
            if len(set(formulas)) != 1:
                raise AssertionError(f"Multiple monomer formulas: {set(formulas)}")
            
            monomer_formula = formulas[0]
            
            # Validate: number of molecules matches z_value
            z_value = atoms.info.get("z_value", len(molecule_groups))
            if len(molecule_groups) != z_value:
                raise AssertionError(f"z mismatch: expected {z_value}, found {len(molecule_groups)}")
            
            # Assign bb_indices
            bb_indices = np.full(len(atoms), -1, dtype=int)
            for group_id, group in enumerate(molecule_groups):
                for atom_idx in group:
                    bb_indices[atom_idx] = group_id
            
            atoms.arrays["bb_indices"] = bb_indices
            atoms.info["monomer"] = monomer_formula
            
            valid_structures.append(atoms)
            
            # Create coarse-grained representation
            cg_atoms = Atoms()
            for mol, offsets in zip(molecule_groups, offset_groups):
                ref_pos = atoms.positions[mol[0]]
                positions = ref_pos + offsets
                com = np.mean(positions, axis=0)
                cg_atoms.extend(Atoms("C", positions=[com]))
            cg_atoms.set_cell(atoms.get_cell())
            cg_atoms.set_pbc(True)
            cg_atoms.info["monomer"] = monomer_formula
            coarse_grained.append(cg_atoms)
            
            # Extract monomers (centered)
            for mol, offsets in zip(molecule_groups, offset_groups):
                symbols = [atoms[j].symbol for j in mol]
                positions = offsets.copy()
                monomer = Atoms(symbols, positions=positions)
                monomer.center()
                all_monomers.append(monomer)
            
        except (AssertionError, ValueError) as e:
            if verbose:
                tqdm.write(f"  Index {i}: {e}")
            bad_indices.append(i)
            continue
    
    if verbose:
        print(f"  Valid: {len(valid_structures)}, Invalid: {len(bad_indices)}")
    
    # Save outputs
    valid_path = output_dir / "valid_molcrystals.extxyz"
    write(str(valid_path), valid_structures, format="extxyz")
    
    cg_path = output_dir / "coarse_grained.extxyz"
    write(str(cg_path), coarse_grained, format="extxyz")
    
    monomers_path = output_dir / "all_monomers.xyz"
    write(str(monomers_path), all_monomers, format="xyz")
    
    if bad_indices:
        np.save(str(output_dir / "bad_indices.npy"), np.array(bad_indices))
    
    if verbose:
        print(f"  Saved {len(valid_structures)} valid structures to {valid_path}")
        print(f"  Saved {len(coarse_grained)} coarse-grained to {cg_path}")
        print(f"  Saved {len(all_monomers)} monomers to {monomers_path}")
    
    return {
        "valid_path": str(valid_path),
        "cg_path": str(cg_path),
        "monomers_path": str(monomers_path),
        "n_valid": len(valid_structures),
        "n_monomers": len(all_monomers),
    }


# =============================================================================
# Step 2: Unwrap and reorder by bb_indices
# =============================================================================

def unwrap_and_reorder(
    input_path: str,
    output_path: str,
    verbose: bool = True,
) -> int:
    """
    Unwrap molecules across periodic boundaries and reorder atoms by bb_indices.
    
    Returns number of structures processed.
    """
    if verbose:
        print(f"  Loading structures from {input_path}...")
    
    images = read(input_path, index=":")
    
    if verbose:
        print(f"  Processing {len(images)} structures...")
    
    unwrapped = []
    iterator = tqdm(images, desc="  Unwrapping") if verbose else images
    
    for atoms in iterator:
        # Get molecule groups and offsets
        molecule_groups, offset_groups = get_molecule_groups(atoms)
        
        # Unwrap: place all atoms of each molecule in the same image
        unwrapped_positions = np.zeros_like(atoms.positions)
        for mol, offsets in zip(molecule_groups, offset_groups):
            ref_pos = atoms.positions[mol[0]]
            new_positions = ref_pos + offsets
            unwrapped_positions[mol] = new_positions
        
        # Reorder atoms by bb_indices (stable sort)
        bb_indices = atoms.arrays["bb_indices"]
        order = np.argsort(bb_indices, kind="stable")
        
        # Create new atoms object with reordered atoms
        new_atoms = Atoms(
            symbols=[atoms[i].symbol for i in order],
            positions=unwrapped_positions[order],
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )
        
        # Copy arrays (reordered)
        new_atoms.arrays["bb_indices"] = bb_indices[order]
        
        # Copy info
        for key, value in atoms.info.items():
            new_atoms.info[key] = value
        
        # Center the structure
        new_atoms.center()
        
        unwrapped.append(new_atoms)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write(output_path, unwrapped, format="extxyz")
    
    if verbose:
        print(f"  Saved {len(unwrapped)} unwrapped structures to {output_path}")
    
    return len(unwrapped)


# =============================================================================
# Step 4: Split data
# =============================================================================

def split_data(
    input_path: str,
    output_dir: str,
    test_size: int = 1000,
    seed: int = 42,
    verbose: bool = True,
):
    """Split data into train/val/test (90% train, test_size test, rest val)."""
    if verbose:
        print(f"  Loading structures from {input_path}...")
    
    images = read(input_path, index=":")
    n_total = len(images)
    
    # Calculate split sizes: 90% train, n_test test, rest val
    n_train = int(n_total * 0.9)
    n_test = test_size
    n_val = n_total - n_train - n_test
    
    if n_val < 0:
        print(f"  Warning: Not enough data for requested test_size={test_size}")
        n_val = 0
        n_test = n_total - n_train
    
    if verbose:
        print(f"  Total: {n_total}, Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Shuffle indices (matching reference implementation)
    np.random.seed(seed)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    # Split order: train, val, test
    train_inds = indices[:n_train]
    val_inds = indices[n_train:n_train + n_val]
    test_inds = indices[n_train + n_val:]
    
    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_inds in [("train", train_inds), ("val", val_inds), ("test", test_inds)]:
        split_images = [images[i] for i in split_inds]
        split_file = output_dir / f"{split_name}.extxyz"
        write(str(split_file), split_images, format="extxyz")
        np.save(str(output_dir / f"{split_name}_inds.npy"), split_inds)
        
        if verbose:
            print(f"  Saved {len(split_images)} structures to {split_file}")


# =============================================================================
# Step 5: Add num_bbs to features
# =============================================================================

def add_num_bbs_to_features(crystals_path: str, features_path: str, verbose: bool = True):
    """Add num_bbs array to the aggregated features pickle."""
    if verbose:
        print(f"  Loading crystals from {crystals_path}...")
    
    images = read(crystals_path, index=":")
    num_bbs = []
    for atoms in images:
        bb_indices = atoms.arrays.get("bb_indices", None)
        if bb_indices is not None:
            n_bb = len(set(bb_indices))
        else:
            n_bb = atoms.info.get("z_value", 1)
        num_bbs.append(n_bb)
    
    if verbose:
        print(f"  Found {len(num_bbs)} structures, total monomers: {sum(num_bbs)}")
    
    with open(features_path, "rb") as f:
        features = pickle.load(f)
    
    features["num_bbs"] = np.array(num_bbs)
    
    with open(features_path, "wb") as f:
        pickle.dump(features, f)
    
    if verbose:
        print(f"  Updated {features_path} with num_bbs")


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(
    input_path: str,
    output_dir: str,
    test_size: int = 1000,
    seed: int = 42,
    skip_to: str = None,
    verbose: bool = True,
):
    """Run the complete OMC25-MCF preprocessing pipeline."""
    
    # Create output directories
    output_dir = Path(output_dir).resolve()
    filtered_dir = output_dir / "filtered"
    unwrapped_dir = output_dir / "unwrapped"
    standardized_dir = output_dir / "standardized"
    splits_dir = output_dir / "splits"
    features_dir = output_dir / "features"
    final_dir = output_dir / "final"
    normalized_dir = output_dir / "normalized"
    
    for d in [filtered_dir, unwrapped_dir, standardized_dir, splits_dir, 
              features_dir, final_dir, normalized_dir]:
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
        print("OMC25-MCF PREPROCESSING PIPELINE")
        print("="*60)
        print(f"Input: {input_path}")
        print(f"Output directory: {output_dir}")
        print(f"Test size: {test_size}, Seed: {seed}")
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    valid_path = filtered_dir / "valid_molcrystals.extxyz"
    monomers_path = filtered_dir / "all_monomers.xyz"
    unwrapped_path = unwrapped_dir / "unwrapped_molcrystals.extxyz"
    standardized_path = standardized_dir / "standardized.extxyz"
    
    # =========================================================================
    # Step 1: Filter and validate structures
    # =========================================================================
    if start_step <= 0:
        if verbose:
            print(f"\n{'='*60}")
            print("Step 1: Filter and validate structures")
            print('='*60)
        
        filter_and_validate(
            input_path=input_path,
            output_dir=str(filtered_dir),
            verbose=verbose,
        )
    
    # =========================================================================
    # Step 2: Unwrap and reorder by bb_indices
    # =========================================================================
    if start_step <= 1:
        if verbose:
            print(f"\n{'='*60}")
            print("Step 2: Unwrap and reorder by bb_indices")
            print('='*60)
        
        unwrap_and_reorder(
            input_path=str(valid_path),
            output_path=str(unwrapped_path),
            verbose=verbose,
        )
    
    # =========================================================================
    # Step 3: Standardize lattice vectors
    # =========================================================================
    if start_step <= 2:
        success = run_python_script(
            THURLEMANN_DIR / "standardize_lattice.py",
            [
                "--input", str(unwrapped_path),
                "--output", str(standardized_path),
            ],
            "Step 3: Standardize lattice vectors",
            verbose
        )
        if not success:
            return False
    
    # =========================================================================
    # Step 4: Split into train/val/test
    # =========================================================================
    if start_step <= 3:
        if verbose:
            print(f"\n{'='*60}")
            print("Step 4: Split into train/val/test")
            print('='*60)
        
        split_data(
            input_path=str(standardized_path),
            output_dir=str(splits_dir),
            test_size=test_size,
            seed=seed,
            verbose=verbose,
        )
    
    # =========================================================================
    # Step 5: Generate molecular features
    # =========================================================================
    if start_step <= 4:
        success = run_python_script(
            THURLEMANN_DIR / "generate_features.py",
            [
                "--input", str(monomers_path),
                "--output_dir", str(features_dir),
            ],
            "Step 5: Generate molecular features",
            verbose
        )
        if not success:
            return False
        
        if verbose:
            print("\nAdding num_bbs to aggregated features...")
        
        add_num_bbs_to_features(
            crystals_path=str(standardized_path),
            features_path=str(features_dir / "agg_extra_features.pkl"),
            verbose=verbose,
        )
    
    # =========================================================================
    # Step 6: Generate training pkl.gz files
    # =========================================================================
    if start_step <= 5:
        for split in ["train", "val", "test"]:
            split_file = splits_dir / f"{split}.extxyz"
            final_file = final_dir / f"{split}_molcrystal.pkl.gz"
            inds_file = splits_dir / f"{split}_inds.npy"
            
            if not split_file.exists():
                if verbose:
                    print(f"Warning: {split_file} not found, skipping {split} split")
                continue
            
            args_list = [
                "--input", str(split_file),
                "--output", str(final_file),
                "--features", str(features_dir / "agg_extra_features.pkl"),
            ]
            if inds_file.exists():
                args_list.extend(["--indices", str(inds_file)])
            
            success = run_python_script(
                THURLEMANN_DIR / "generate_training_data.py",
                args_list,
                f"Step 6: Generate {split} pkl.gz",
                verbose
            )
            if not success:
                return False
    
    # =========================================================================
    # Step 7: Normalize local coordinate frames
    # =========================================================================
    if start_step <= 6:
        success = run_python_script(
            THURLEMANN_DIR / "normalize_data.py",
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
        
        print("\n[Final Training Data (normalized)]")
        for split in ["train", "val", "test"]:
            check_file_exists(normalized_dir / f"{split}_molcrystal_normalized.pkl.gz", f"{split.capitalize()} data")
        
        print("\n[Pre-normalization Data]")
        for split in ["train", "val", "test"]:
            check_file_exists(final_dir / f"{split}_molcrystal.pkl.gz", f"{split.capitalize()} data (raw)")
        
        print("\n[Split Files]")
        for split in ["train", "val", "test"]:
            check_file_exists(splits_dir / f"{split}.extxyz", f"{split.capitalize()} XYZ")
        
        print("\n[Intermediate Files]")
        check_file_exists(valid_path, "Valid structures")
        check_file_exists(monomers_path, "Monomers")
        check_file_exists(unwrapped_path, "Unwrapped structures")
        check_file_exists(standardized_path, "Standardized structures")
        check_file_exists(features_dir / "agg_extra_features.pkl", "Aggregated features")
        
        print("\n" + "="*60)
        print("To use the final data for training, load:")
        print(f"  {normalized_dir / 'train_molcrystal_normalized.pkl.gz'}")
        print(f"  {normalized_dir / 'val_molcrystal_normalized.pkl.gz'}")
        print(f"  {normalized_dir / 'test_molcrystal_normalized.pkl.gz'}")
        print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="OMC25-MCF preprocessing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  step1: Filter and validate structures (assign bb_indices, extract monomers)
  step2: Unwrap molecules and reorder atoms by bb_indices
  step3: Standardize lattice vectors
  step4: Split into train/val/test (90% train, 1000 test, rest val)
  step5: Generate molecular features (bond fixing + extraction)
  step6: Generate training pkl.gz files
  step7: Normalize local coordinate frames

Examples:
  # Full pipeline
  python run_pipeline.py --input omc25_lowest_energy.extxyz --output_dir ./preprocessed

  # Custom test size
  python run_pipeline.py --input ... --output_dir ... --test_size 500

  # Resume from step 3
  python run_pipeline.py --input ... --output_dir ... --skip_to step3
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input extxyz file (e.g., omc25_lowest_energy.extxyz)",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./preprocessed",
        help="Output directory for all generated files (default: ./preprocessed)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Number of test samples (default: 1000)",
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
        choices=["step1", "step2", "step3", "step4", "step5", "step6", "step7"],
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
        input_path=str(input_path),
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        skip_to=args.skip_to,
        verbose=not args.quiet,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
