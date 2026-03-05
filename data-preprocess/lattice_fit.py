#!/usr/bin/env python
"""
Compute lognormal statistics for lattice parameters from training data.

This script fits lognormal distributions to the lattice cell parameters
(a, b, c, alpha, beta, gamma) from the training splits of molecular crystal
datasets. The resulting statistics can be used as priors for sampling
lattice matrices during generation.

For lognormal distribution:
  - loc = mean(log(x))
  - scale = std(log(x))

Usage:
    python lattice_fit.py --dataset thurlemann
    python lattice_fit.py --dataset omc25
    python lattice_fit.py --all
    python lattice_fit.py --input /path/to/train.extxyz --output lattice_stats.yaml
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml
from ase.io import read


def compute_lattice_stats(
    structures: List,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute lognormal statistics for lattice parameters.
    
    Args:
        structures: List of ASE Atoms objects with cell information.
        verbose: Whether to print progress.
    
    Returns:
        Dictionary with:
            - cellparams: (N, 6) array of cell parameters [a, b, c, alpha, beta, gamma]
            - log_cellparams: (N, 6) array of log cell parameters
            - loc: (6,) array of mean(log(x)) for each parameter
            - scale: (6,) array of std(log(x)) for each parameter
            - lengths_loc: (3,) array for a, b, c
            - lengths_scale: (3,) array for a, b, c
            - angles_loc: (3,) array for alpha, beta, gamma
            - angles_scale: (3,) array for alpha, beta, gamma
    """
    cellparams = []
    
    for atoms in structures:
        cell = atoms.get_cell()
        if cell is not None and cell.volume > 0:
            # cellpar() returns [a, b, c, alpha, beta, gamma]
            params = cell.cellpar()
            cellparams.append(params)
    
    cellparams = np.array(cellparams, dtype=np.float64)
    
    if verbose:
        print(f"  Number of structures: {len(cellparams)}")
        print(f"  Cell parameter ranges:")
        print(f"    a:     {cellparams[:, 0].min():.2f} - {cellparams[:, 0].max():.2f} Å")
        print(f"    b:     {cellparams[:, 1].min():.2f} - {cellparams[:, 1].max():.2f} Å")
        print(f"    c:     {cellparams[:, 2].min():.2f} - {cellparams[:, 2].max():.2f} Å")
        print(f"    alpha: {cellparams[:, 3].min():.2f} - {cellparams[:, 3].max():.2f}°")
        print(f"    beta:  {cellparams[:, 4].min():.2f} - {cellparams[:, 4].max():.2f}°")
        print(f"    gamma: {cellparams[:, 5].min():.2f} - {cellparams[:, 5].max():.2f}°")
    
    # Compute log of cell parameters
    log_cellparams = np.log(cellparams)
    
    # Compute lognormal statistics
    loc = log_cellparams.mean(axis=0)
    scale = log_cellparams.std(axis=0)
    
    if verbose:
        print(f"\n  Lognormal parameters (loc=mean(log(x)), scale=std(log(x))):")
        param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        for i, name in enumerate(param_names):
            print(f"    {name:5s}: loc={loc[i]:.6f}, scale={scale[i]:.6f}")
    
    return {
        "cellparams": cellparams,
        "log_cellparams": log_cellparams,
        "loc": loc,
        "scale": scale,
        "lengths_loc": loc[:3],
        "lengths_scale": scale[:3],
        "angles_loc": loc[3:],
        "angles_scale": scale[3:],
    }


def save_yaml(stats: Dict, output_path: str, dataset_name: str = "dataset"):
    """Save lattice statistics to YAML file (lengths only for lognormal prior)."""
    # Convert numpy arrays to lists for YAML serialization
    # Main lognormal prior uses only lengths (a, b, c), angles use separate prior
    yaml_data = {
        "dataset": dataset_name,
        "n_structures": int(stats["cellparams"].shape[0]),
        "lattice": {
            "lognormal": {
                "loc": stats["lengths_loc"].tolist(),
                "scale": stats["lengths_scale"].tolist(),
            },
        },
        "raw_stats": {
            "lengths_mean": stats["cellparams"][:, :3].mean(axis=0).tolist(),
            "lengths_std": stats["cellparams"][:, :3].std(axis=0).tolist(),
            "lengths_min": stats["cellparams"][:, :3].min(axis=0).tolist(),
            "lengths_max": stats["cellparams"][:, :3].max(axis=0).tolist(),
            "angles_mean": stats["cellparams"][:, 3:].mean(axis=0).tolist(),
            "angles_std": stats["cellparams"][:, 3:].std(axis=0).tolist(),
        },
    }
    
    with open(output_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nSaved to {output_path}")


def print_yaml_snippet(stats: Dict, dataset_name: str):
    """Print YAML snippet for config file (lengths only, angles use separate prior)."""
    print(f"\n{'='*60}")
    print(f"YAML config snippet for {dataset_name}:")
    print('='*60)
    print("lattice:")
    print("  lognormal:")
    print("    loc:")
    # Only print lengths (a, b, c), not angles
    for val in stats["lengths_loc"]:
        print(f"      - {val}")
    print("    scale:")
    for val in stats["lengths_scale"]:
        print(f"      - {val}")
    print('='*60)


def fit_thurlemann(verbose: bool = True) -> Dict:
    """Fit lattice statistics for Thurlemann2023 dataset."""
    script_dir = Path(__file__).parent
    
    # Try standardized train file first (preferred)
    train_path = script_dir / "thurlemann23" / "preprocessed" / "standardized" / "standardized_train_molcrystal.extxyz"
    
    if not train_path.exists():
        # Try splits directory
        train_path = script_dir / "thurlemann23" / "preprocessed" / "splits" / "train_molcrystal.extxyz"
    
    if not train_path.exists():
        # Try legacy paths
        train_path = script_dir / "thurlemann23" / "splits" / "train.extxyz"
    
    if not train_path.exists():
        print(f"ERROR: Training file not found. Tried:")
        print(f"  - {script_dir / 'thurlemann23' / 'preprocessed' / 'standardized' / 'standardized_train_molcrystal.extxyz'}")
        print(f"  - {script_dir / 'thurlemann23' / 'preprocessed' / 'splits' / 'train_molcrystal.extxyz'}")
        print("Please run the preprocessing pipeline first.")
        return None
    
    print(f"\n{'='*60}")
    print("THURLEMANN2023 DATASET")
    print('='*60)
    print(f"Loading: {train_path}")
    
    structures = read(str(train_path), index=":")
    stats = compute_lattice_stats(structures, verbose=verbose)
    
    if verbose:
        print_yaml_snippet(stats, "thurlemann2023")
    
    return stats


def fit_omc25(verbose: bool = True) -> Dict:
    """Fit lattice statistics for OMC25-MCF dataset."""
    script_dir = Path(__file__).parent
    train_path = script_dir / "omc25-mcf" / "preprocessed" / "splits" / "train.extxyz"
    
    if not train_path.exists():
        # Try alternative path
        train_path = script_dir / "omc25-mcf" / "preprocessed" / "standardized" / "standardized.extxyz"
    
    if not train_path.exists():
        print(f"ERROR: Training file not found at {train_path}")
        print("Please run the preprocessing pipeline first.")
        return None
    
    print(f"\n{'='*60}")
    print("OMC25-MCF DATASET")
    print('='*60)
    print(f"Loading: {train_path}")
    
    structures = read(str(train_path), index=":")
    stats = compute_lattice_stats(structures, verbose=verbose)
    
    if verbose:
        print_yaml_snippet(stats, "omc25-mcf")
    
    return stats


def fit_custom(input_path: str, verbose: bool = True) -> Dict:
    """Fit lattice statistics for a custom dataset."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"CUSTOM DATASET: {input_path.name}")
    print('='*60)
    print(f"Loading: {input_path}")
    
    structures = read(str(input_path), index=":")
    stats = compute_lattice_stats(structures, verbose=verbose)
    
    if verbose:
        print_yaml_snippet(stats, input_path.stem)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute lognormal statistics for lattice parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit Thurlemann2023 dataset
  python lattice_fit.py --dataset thurlemann

  # Fit OMC25-MCF dataset
  python lattice_fit.py --dataset omc25

  # Fit both datasets
  python lattice_fit.py --all

  # Fit custom dataset
  python lattice_fit.py --input /path/to/train.extxyz

  # Save to YAML file
  python lattice_fit.py --dataset omc25 --output omc25_lattice.yaml
        """,
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["thurlemann", "omc25", "both"],
        default=None,
        help="Dataset to fit: thurlemann, omc25, or both",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Fit all datasets (thurlemann and omc25)",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to custom training extxyz file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output YAML file path (optional)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - only print essential output",
    )
    
    args = parser.parse_args()
    
    # Determine what to fit
    if args.input:
        stats = fit_custom(args.input, verbose=not args.quiet)
        if stats and args.output:
            save_yaml(stats, args.output, Path(args.input).stem)
    elif args.all or args.dataset == "both":
        thurlemann_stats = fit_thurlemann(verbose=not args.quiet)
        omc25_stats = fit_omc25(verbose=not args.quiet)
        
        if args.output:
            print("\nNote: --output with --all saves only the last dataset. "
                  "Use separate calls for each dataset.")
    elif args.dataset == "thurlemann":
        stats = fit_thurlemann(verbose=not args.quiet)
        if stats and args.output:
            save_yaml(stats, args.output, "thurlemann2023")
    elif args.dataset == "omc25":
        stats = fit_omc25(verbose=not args.quiet)
        if stats and args.output:
            save_yaml(stats, args.output, "omc25-mcf")
    else:
        # Default: fit both
        print("No dataset specified. Fitting all available datasets...")
        fit_thurlemann(verbose=not args.quiet)
        fit_omc25(verbose=not args.quiet)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
