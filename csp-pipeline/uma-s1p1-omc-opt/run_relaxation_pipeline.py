#!/usr/bin/env python3
"""
UMA-OMC Relaxation Pipeline for MolCrystalFlow Generated Structures.

This script orchestrates the full relaxation pipeline:
1. Combine sampled XYZ files for each molecule
2. Run rigid-body relaxation with UMA-OMC model (parallel jobs)
3. Check if all rigid-body jobs are finished
4. Combine all rigid-body optimized structures
5. Filter structures by inter-building-block distances (> 1.0 Å)
6. Run full cell optimization (parallel jobs)
7. Check if all cell-opt jobs are finished
8. Combine all cell-optimized structures

Usage:
    # Step 1: Combine input XYZ files
    python run_relaxation_pipeline.py combine_input --input_dir ./generated --output pred_combined.xyz

    # Step 2: Submit rigid-body optimization jobs
    python run_relaxation_pipeline.py submit_rigid_body --input pred_combined.xyz --chunk_size 50

    # Step 3: Check rigid-body job status
    python run_relaxation_pipeline.py check_rigid_body --expected_chunks 48

    # Step 4: Combine rigid-body results and filter
    python run_relaxation_pipeline.py combine_and_filter --min_dist 1.0

    # Step 5: Submit cell optimization jobs
    python run_relaxation_pipeline.py submit_cell_opt --input filtered_by_inter_bb.xyz --chunk_size 10

    # Step 6: Check cell-opt job status
    python run_relaxation_pipeline.py check_cell_opt --expected_chunks 240

    # Step 7: Combine cell-opt results
    python run_relaxation_pipeline.py combine_cell_opt

    # Or run all steps interactively
    python run_relaxation_pipeline.py interactive
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase.geometry import get_distances
from ase.io import read, write


# ==============================================================================
# Step 1: Combine Input XYZ Files
# ==============================================================================


def combine_input_xyz(
    input_dir: str,
    output_path: str,
    pattern: str = "crystals_z*.xyz",
    verbose: bool = True,
) -> int:
    """Combine multiple XYZ files from generation into a single file."""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    
    if not files:
        raise ValueError(f"No files found matching {input_dir / pattern}")
    
    all_atoms = []
    for f in files:
        atoms_list = read(f, index=":")
        all_atoms.extend(atoms_list)
        if verbose:
            print(f"  Loaded {len(atoms_list)} structures from {f.name}")
    
    write(output_path, all_atoms, format="extxyz")
    if verbose:
        print(f"\nCombined {len(all_atoms)} structures into {output_path}")
    
    return len(all_atoms)


# ==============================================================================
# Step 2: Submit Rigid-Body Optimization Jobs
# ==============================================================================


def generate_rigid_body_submit_script(
    input_xyz: str,
    chunk_size: int,
    total_structures: int,
    output_dir: str = "opt-results",
    log_dir: str = "log-files",
    slurm_account: str = "mingjieliu-faimm",
    time_limit: str = "6:00:00",
    partition: str = "hpg-turin",
) -> str:
    """Generate SLURM submission script for rigid-body optimization."""
    num_chunks = (total_structures + chunk_size - 1) // chunk_size
    
    script = f"""#!/bin/bash
#SBATCH --output=slurmoutputs/R-%x.%j.out
#SBATCH --error=slurmoutputs/R-%x.%j.err
#SBATCH --job-name=uma-rigidbody-opt
#SBATCH --array=0-{num_chunks - 1}
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account={slurm_account}
#SBATCH --qos={slurm_account}
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --gpus=1

module load conda
conda activate fairchem
pwd; hostname; date

START=$((0 + SLURM_ARRAY_TASK_ID * {chunk_size}))
END=$(( START + {chunk_size}))

python optimize_batch.py --start $START --end $END --input {input_xyz}
"""
    return script, num_chunks


def submit_rigid_body_jobs(
    input_xyz: str,
    chunk_size: int = 50,
    dry_run: bool = False,
    verbose: bool = True,
) -> int:
    """Submit rigid-body optimization jobs."""
    # Count structures
    atoms_list = read(input_xyz, index=":")
    total = len(atoms_list)
    del atoms_list  # Free memory
    
    script, num_chunks = generate_rigid_body_submit_script(
        input_xyz=input_xyz,
        chunk_size=chunk_size,
        total_structures=total,
    )
    
    # Create slurmoutputs directory
    os.makedirs("slurmoutputs", exist_ok=True)
    
    # Write submission script
    script_path = "submit_rigid_body_opt_generated.sh"
    with open(script_path, "w") as f:
        f.write(script)
    
    if verbose:
        print(f"Generated submission script: {script_path}")
        print(f"  Total structures: {total}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Number of jobs: {num_chunks}")
    
    if not dry_run:
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Submitted jobs: {result.stdout.strip()}")
        else:
            print(f"Error submitting jobs: {result.stderr}")
    else:
        print("Dry run - not submitting jobs")
    
    return num_chunks


# ==============================================================================
# Step 3: Check Job Status
# ==============================================================================


def check_job_completion(
    results_dir: str,
    pattern: str,
    expected_chunks: int,
    verbose: bool = True,
) -> Tuple[int, int, List[str]]:
    """Check if all optimization jobs have completed."""
    files = glob.glob(os.path.join(results_dir, pattern))
    completed = len(files)
    missing = []
    
    if verbose:
        print(f"Found {completed}/{expected_chunks} completed chunks in {results_dir}")
    
    # Parse completed ranges
    completed_ranges = set()
    for f in files:
        m = re.search(r"_(\d+)_(\d+)\.(ext)?xyz$", f)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            completed_ranges.add((start, end))
    
    return completed, expected_chunks, missing


def check_rigid_body_jobs(expected_chunks: int, verbose: bool = True) -> bool:
    """Check if all rigid-body optimization jobs are complete."""
    completed, total, missing = check_job_completion(
        results_dir="opt-results",
        pattern="pred_omc25_100step_opt_*.extxyz",
        expected_chunks=expected_chunks,
        verbose=verbose,
    )
    return completed >= expected_chunks


def check_cell_opt_jobs(expected_chunks: int, verbose: bool = True) -> bool:
    """Check if all cell optimization jobs are complete."""
    completed, total, missing = check_job_completion(
        results_dir="cell-opt-results",
        pattern="pred_omc25_*step_cell_opt_*.extxyz",
        expected_chunks=expected_chunks,
        verbose=verbose,
    )
    return completed >= expected_chunks


# ==============================================================================
# Step 4: Combine and Filter Structures
# ==============================================================================


def extract_start_end(filename: str) -> Tuple[int, int]:
    """Extract start and end indices from filename."""
    base = os.path.basename(filename)
    m = re.search(r"opt_(\d+)_(\d+)\.(ext)?xyz$", base)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (0, 0)


def combine_optimized_xyz(
    pattern: str,
    output_path: str,
    verbose: bool = True,
) -> int:
    """Combine multiple optimized XYZ files into a single file."""
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No files found for pattern: {pattern}")
    
    # Sort by start index
    files.sort(key=lambda p: extract_start_end(p))
    
    all_atoms = []
    for f in files:
        atoms_list = read(f, index=":")
        all_atoms.extend(atoms_list)
        if verbose:
            print(f"  Loaded {len(atoms_list)} structures from {os.path.basename(f)}")
    
    write(output_path, all_atoms, format="extxyz")
    if verbose:
        print(f"\nCombined {len(all_atoms)} structures into {output_path}")
    
    return len(all_atoms)


def compute_bb_min_distances(atoms, bb_indices) -> np.ndarray:
    """Compute minimum distances between all pairs of building blocks."""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    
    bb_map = defaultdict(list)
    for i, bb in enumerate(bb_indices):
        bb_map[bb].append(i)
    bb_ids = sorted(bb_map.keys())
    
    all_min_dists = []
    for ia, bb_i in enumerate(bb_ids):
        for jb in range(ia + 1, len(bb_ids)):
            bb_j = bb_ids[jb]
            inds_i = bb_map[bb_i]
            inds_j = bb_map[bb_j]
            
            pos_i = positions[inds_i]
            pos_j = positions[inds_j]
            
            _, dists = get_distances(pos_i, pos_j, cell=cell, pbc=pbc)
            all_min_dists.append(np.min(dists))
    
    return np.array(all_min_dists)


def filter_by_inter_bb_distance(
    input_xyz: str,
    output_xyz: str,
    min_distance: float = 1.0,
    bb_key: str = "bb_indices",
    save_indices: bool = True,
    verbose: bool = True,
) -> Tuple[int, int]:
    """Filter structures by minimum inter-building-block distance."""
    atoms_list = read(input_xyz, index=":")
    
    filtered_structures = []
    filtered_indices = []
    
    for idx, molcrystal in enumerate(atoms_list):
        # Try both bb_indices and bb_idx for compatibility
        if bb_key in molcrystal.arrays:
            bb_indices = molcrystal.arrays[bb_key]
        elif "bb_idx" in molcrystal.arrays:
            bb_indices = molcrystal.arrays["bb_idx"]
        else:
            if verbose:
                print(f"  Warning: Structure {idx} has no bb_indices, skipping")
            continue
        
        all_min_dists = compute_bb_min_distances(molcrystal, bb_indices)
        
        # Skip structures with only one building block
        if len(all_min_dists) == 0:
            continue
        
        if np.min(all_min_dists) > min_distance:
            filtered_structures.append(molcrystal)
            filtered_indices.append(idx)
    
    if len(filtered_structures) > 0:
        write(output_xyz, filtered_structures, format="extxyz")
    
    if save_indices:
        np.save(output_xyz.replace(".xyz", "_indices.npy"), 
                np.array(filtered_indices, dtype=int))
    
    if verbose:
        print(f"\nFiltering results:")
        print(f"  Total structures: {len(atoms_list)}")
        print(f"  Kept (min bb distance > {min_distance} Å): {len(filtered_structures)}")
        print(f"  Filtered out: {len(atoms_list) - len(filtered_structures)}")
        print(f"  Output: {output_xyz}")
    
    return len(filtered_structures), len(atoms_list)


def combine_and_filter(
    rigid_body_pattern: str = "./opt-results/pred_omc25_100step_opt_*.extxyz",
    combined_output: str = "pred_uma_rigid_body_opt.xyz",
    filtered_output: str = "filtered_by_inter_bb.xyz",
    min_distance: float = 1.0,
    verbose: bool = True,
) -> Tuple[int, int]:
    """Combine rigid-body optimized structures and filter by inter-BB distance."""
    # Step 1: Combine
    if verbose:
        print("Combining rigid-body optimized structures...")
    total = combine_optimized_xyz(rigid_body_pattern, combined_output, verbose=verbose)
    
    # Step 2: Filter
    if verbose:
        print(f"\nFiltering by inter-BB distance (min > {min_distance} Å)...")
    kept, total = filter_by_inter_bb_distance(
        combined_output, 
        filtered_output, 
        min_distance=min_distance,
        verbose=verbose,
    )
    
    return kept, total


# ==============================================================================
# Step 5: Submit Cell Optimization Jobs
# ==============================================================================


def generate_cell_opt_submit_script(
    input_xyz: str,
    chunk_size: int,
    total_structures: int,
    output_dir: str = "cell-opt-results",
    log_dir: str = "cell-opt-logs",
    slurm_account: str = "mingjieliu-faimm",
    time_limit: str = "24:00:00",
    partition: str = "hpg-turin",
    steps: int = 1000,
) -> Tuple[str, int]:
    """Generate SLURM submission script for cell optimization."""
    num_chunks = (total_structures + chunk_size - 1) // chunk_size
    
    script = f"""#!/bin/bash
#SBATCH --output=slurmoutputs/R-%x.%j.out
#SBATCH --error=slurmoutputs/R-%x.%j.err
#SBATCH --job-name=uma-cell-opt
#SBATCH --array=0-{num_chunks - 1}
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account={slurm_account}
#SBATCH --qos={slurm_account}
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --gpus=1

module load conda
conda activate fairchem
pwd; hostname; date

START=$((0 + SLURM_ARRAY_TASK_ID * {chunk_size}))
END=$(( START + {chunk_size}))

python cell_opt_optimize_batch.py --start $START --end $END --input {input_xyz}
"""
    return script, num_chunks


def submit_cell_opt_jobs(
    input_xyz: str = "filtered_by_inter_bb.xyz",
    chunk_size: int = 10,
    dry_run: bool = False,
    verbose: bool = True,
) -> int:
    """Submit cell optimization jobs."""
    # Count structures
    atoms_list = read(input_xyz, index=":")
    total = len(atoms_list)
    del atoms_list
    
    script, num_chunks = generate_cell_opt_submit_script(
        input_xyz=input_xyz,
        chunk_size=chunk_size,
        total_structures=total,
    )
    
    os.makedirs("slurmoutputs", exist_ok=True)
    
    script_path = "submit_cell_opt_generated.sh"
    with open(script_path, "w") as f:
        f.write(script)
    
    if verbose:
        print(f"Generated submission script: {script_path}")
        print(f"  Total structures: {total}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Number of jobs: {num_chunks}")
    
    if not dry_run:
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Submitted jobs: {result.stdout.strip()}")
        else:
            print(f"Error submitting jobs: {result.stderr}")
    else:
        print("Dry run - not submitting jobs")
    
    return num_chunks


# ==============================================================================
# Step 7: Combine Cell-Opt Results
# ==============================================================================


def combine_cell_opt_results(
    pattern: str = "./cell-opt-results/pred_omc25_*step_cell_opt_*.extxyz",
    output_path: str = "pred_uma_cell_opt_final.xyz",
    verbose: bool = True,
) -> int:
    """Combine cell optimization results."""
    return combine_optimized_xyz(pattern, output_path, verbose=verbose)


# ==============================================================================
# Interactive Mode
# ==============================================================================


def interactive_pipeline():
    """Run the pipeline interactively with prompts."""
    print("=" * 60)
    print("UMA-OMC Relaxation Pipeline - Interactive Mode")
    print("=" * 60)
    
    while True:
        print("\nSelect a step:")
        print("  1. Combine input XYZ files")
        print("  2. Submit rigid-body optimization jobs")
        print("  3. Check rigid-body job status")
        print("  4. Combine and filter rigid-body results")
        print("  5. Submit cell optimization jobs")
        print("  6. Check cell-opt job status")
        print("  7. Combine cell-opt results")
        print("  8. Exit")
        
        choice = input("\nEnter choice (1-8): ").strip()
        
        if choice == "1":
            input_dir = input("Input directory [./generated]: ").strip() or "./generated"
            output = input("Output file [pred_combined.xyz]: ").strip() or "pred_combined.xyz"
            combine_input_xyz(input_dir, output)
        
        elif choice == "2":
            input_xyz = input("Input XYZ [pred_combined.xyz]: ").strip() or "pred_combined.xyz"
            chunk_size = int(input("Chunk size [50]: ").strip() or "50")
            dry_run = input("Dry run? [y/N]: ").strip().lower() == "y"
            submit_rigid_body_jobs(input_xyz, chunk_size, dry_run=dry_run)
        
        elif choice == "3":
            expected = int(input("Expected number of chunks: ").strip())
            check_rigid_body_jobs(expected)
        
        elif choice == "4":
            min_dist = float(input("Minimum inter-BB distance [1.0]: ").strip() or "1.0")
            combine_and_filter(min_distance=min_dist)
        
        elif choice == "5":
            input_xyz = input("Input XYZ [filtered_by_inter_bb.xyz]: ").strip() or "filtered_by_inter_bb.xyz"
            chunk_size = int(input("Chunk size [10]: ").strip() or "10")
            dry_run = input("Dry run? [y/N]: ").strip().lower() == "y"
            submit_cell_opt_jobs(input_xyz, chunk_size, dry_run=dry_run)
        
        elif choice == "6":
            expected = int(input("Expected number of chunks: ").strip())
            check_cell_opt_jobs(expected)
        
        elif choice == "7":
            combine_cell_opt_results()
        
        elif choice == "8":
            print("Exiting.")
            break
        
        else:
            print("Invalid choice.")


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="UMA-OMC Relaxation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to run")
    
    # Combine input
    p1 = subparsers.add_parser("combine_input", help="Combine input XYZ files")
    p1.add_argument("--input_dir", type=str, default="./generated", help="Input directory")
    p1.add_argument("--output", type=str, default="pred_combined.xyz", help="Output file")
    p1.add_argument("--pattern", type=str, default="crystals_z*.xyz", help="File pattern")
    
    # Submit rigid-body
    p2 = subparsers.add_parser("submit_rigid_body", help="Submit rigid-body optimization jobs")
    p2.add_argument("--input", type=str, default="pred_combined.xyz", help="Input XYZ")
    p2.add_argument("--chunk_size", type=int, default=50, help="Chunk size")
    p2.add_argument("--dry_run", action="store_true", help="Dry run")
    
    # Check rigid-body
    p3 = subparsers.add_parser("check_rigid_body", help="Check rigid-body job status")
    p3.add_argument("--expected_chunks", type=int, required=True, help="Expected chunks")
    
    # Combine and filter
    p4 = subparsers.add_parser("combine_and_filter", help="Combine and filter rigid-body results")
    p4.add_argument("--min_dist", type=float, default=1.0, help="Min inter-BB distance")
    
    # Submit cell-opt
    p5 = subparsers.add_parser("submit_cell_opt", help="Submit cell optimization jobs")
    p5.add_argument("--input", type=str, default="filtered_by_inter_bb.xyz", help="Input XYZ")
    p5.add_argument("--chunk_size", type=int, default=10, help="Chunk size")
    p5.add_argument("--dry_run", action="store_true", help="Dry run")
    
    # Check cell-opt
    p6 = subparsers.add_parser("check_cell_opt", help="Check cell-opt job status")
    p6.add_argument("--expected_chunks", type=int, required=True, help="Expected chunks")
    
    # Combine cell-opt
    p7 = subparsers.add_parser("combine_cell_opt", help="Combine cell-opt results")
    p7.add_argument("--output", type=str, default="pred_uma_cell_opt_final.xyz", help="Output")
    
    # Interactive
    subparsers.add_parser("interactive", help="Run interactively")
    
    args = parser.parse_args()
    
    if args.command == "combine_input":
        combine_input_xyz(args.input_dir, args.output, args.pattern)
    
    elif args.command == "submit_rigid_body":
        submit_rigid_body_jobs(args.input, args.chunk_size, args.dry_run)
    
    elif args.command == "check_rigid_body":
        check_rigid_body_jobs(args.expected_chunks)
    
    elif args.command == "combine_and_filter":
        combine_and_filter(min_distance=args.min_dist)
    
    elif args.command == "submit_cell_opt":
        submit_cell_opt_jobs(args.input, args.chunk_size, args.dry_run)
    
    elif args.command == "check_cell_opt":
        check_cell_opt_jobs(args.expected_chunks)
    
    elif args.command == "combine_cell_opt":
        combine_cell_opt_results(output_path=args.output)
    
    elif args.command == "interactive":
        interactive_pipeline()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
