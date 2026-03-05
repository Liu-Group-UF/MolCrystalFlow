#!/usr/bin/env python3
"""
Pipeline Monitor Script for UMA-OMC Relaxation.

This script monitors the progress of relaxation jobs and automatically
triggers the next step when the current step completes.

Pipeline Steps:
1. Rigid-body optimization (submit_rigid_body_opt.sh)
2. Combine rigid-body results
3. Filter by inter-BB distance
4. Cell optimization (submit_cell_opt.sh)
5. Combine cell-opt results

Usage:
    # Run as a monitoring job
    python monitor_pipeline.py --mode monitor --step rigid_body

    # Or submit via SLURM
    sbatch submit_monitor.sh
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np


def log(msg: str):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def count_structures_in_xyz(filepath: str) -> int:
    """Count number of structures in an XYZ file without loading all into memory."""
    count = 0
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                n_atoms = int(line.strip())
                count += 1
                # Skip comment line and atom lines
                f.readline()  # comment
                for _ in range(n_atoms):
                    f.readline()
            except ValueError:
                continue
    return count


def check_rigid_body_completion(
    results_dir: str = "opt-results",
    pattern: str = "pred_omc25_100step_opt_*.extxyz",
    expected_total: int = None,
    chunk_size: int = 50,
) -> Tuple[bool, int, int]:
    """
    Check if rigid-body optimization is complete.
    
    Returns:
        (is_complete, completed_structures, expected_structures)
    """
    files = glob.glob(os.path.join(results_dir, pattern))
    
    if not files:
        return False, 0, expected_total or 0
    
    # Count structures in completed files
    completed = 0
    for f in files:
        try:
            completed += count_structures_in_xyz(f)
        except Exception as e:
            log(f"Warning: Could not read {f}: {e}")
    
    if expected_total is None:
        # Estimate from number of files
        expected_total = len(files) * chunk_size
    
    is_complete = completed >= expected_total or len(files) * chunk_size >= expected_total
    
    return is_complete, completed, expected_total


def check_cell_opt_completion(
    results_dir: str = "cell-opt-results",
    pattern: str = "pred_omc25_*step_cell_opt_*.extxyz",
    expected_total: int = None,
    chunk_size: int = 10,
) -> Tuple[bool, int, int]:
    """
    Check if cell optimization is complete.
    
    Returns:
        (is_complete, completed_structures, expected_structures)
    """
    files = glob.glob(os.path.join(results_dir, pattern))
    
    if not files:
        return False, 0, expected_total or 0
    
    completed = 0
    for f in files:
        try:
            completed += count_structures_in_xyz(f)
        except Exception as e:
            log(f"Warning: Could not read {f}: {e}")
    
    if expected_total is None:
        expected_total = len(files) * chunk_size
    
    is_complete = completed >= expected_total or len(files) * chunk_size >= expected_total
    
    return is_complete, completed, expected_total


def check_slurm_jobs_running(job_name_pattern: str) -> int:
    """Check how many SLURM jobs matching the pattern are still running."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-n", job_name_pattern, "-h"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split('\n') if l]
            return len(lines)
    except Exception as e:
        log(f"Warning: Could not check SLURM queue: {e}")
    return -1  # Unknown


def combine_rigid_body_results(
    pattern: str = "./opt-results/pred_omc25_100step_opt_*.extxyz",
    output: str = "pred_uma_rigid_body_opt.xyz",
) -> int:
    """Combine rigid-body optimization results."""
    from ase.io import read, write
    
    files = sorted(glob.glob(pattern), key=lambda f: extract_indices(f))
    
    if not files:
        raise ValueError(f"No files found matching {pattern}")
    
    all_atoms = []
    for f in files:
        atoms_list = read(f, index=":")
        all_atoms.extend(atoms_list)
        log(f"  Loaded {len(atoms_list)} from {os.path.basename(f)}")
    
    write(output, all_atoms, format="extxyz")
    log(f"Combined {len(all_atoms)} structures into {output}")
    
    return len(all_atoms)


def extract_indices(filename: str) -> Tuple[int, int]:
    """Extract start/end indices from filename."""
    m = re.search(r"_(\d+)_(\d+)\.(ext)?xyz$", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (0, 0)


def filter_by_inter_bb_distance(
    input_xyz: str = "pred_uma_rigid_body_opt.xyz",
    output_xyz: str = "filtered_by_inter_bb.xyz",
    min_distance: float = 1.0,
) -> Tuple[int, int]:
    """Filter structures by inter-BB distance."""
    from collections import defaultdict
    from ase.geometry import get_distances
    from ase.io import read, write
    
    atoms_list = read(input_xyz, index=":")
    
    filtered = []
    filtered_indices = []
    
    for idx, atoms in enumerate(atoms_list):
        # Get bb indices
        if 'bb_indices' in atoms.arrays:
            bb_idx = atoms.arrays['bb_indices']
        elif 'bb_idx' in atoms.arrays:
            bb_idx = atoms.arrays['bb_idx']
        else:
            log(f"Warning: Structure {idx} missing bb_indices, skipping")
            continue
        
        # Compute min distances between BBs
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        
        bb_map = defaultdict(list)
        for i, bb in enumerate(bb_idx):
            bb_map[bb].append(i)
        bb_ids = sorted(bb_map.keys())
        
        all_min_dists = []
        for ia, bb_i in enumerate(bb_ids):
            for jb in range(ia + 1, len(bb_ids)):
                bb_j = bb_ids[jb]
                pos_i = positions[bb_map[bb_i]]
                pos_j = positions[bb_map[bb_j]]
                _, dists = get_distances(pos_i, pos_j, cell=cell, pbc=pbc)
                all_min_dists.append(np.min(dists))
        
        if len(all_min_dists) == 0:
            continue
        
        if np.min(all_min_dists) > min_distance:
            filtered.append(atoms)
            filtered_indices.append(idx)
    
    if filtered:
        write(output_xyz, filtered, format="extxyz")
    
    np.save(output_xyz.replace(".xyz", "_indices.npy"), np.array(filtered_indices))
    
    log(f"Filtered: {len(filtered)}/{len(atoms_list)} kept (min dist > {min_distance} Å)")
    
    return len(filtered), len(atoms_list)


def submit_rigid_body_jobs(
    input_xyz: str,
    chunk_size: int = 50,
    script_path: str = "submit_rigid_body_auto.sh",
    slurm_account: str = "genai-mingjieliu",
    time_limit: str = "6:00:00",
    partition: str = "hpg-turin",
) -> Tuple[int, str]:
    """Generate and submit rigid-body optimization jobs."""
    # Count structures
    total = count_structures_in_xyz(input_xyz)
    num_chunks = (total + chunk_size - 1) // chunk_size
    
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
    
    with open(script_path, 'w') as f:
        f.write(script)
    
    os.makedirs("slurmoutputs", exist_ok=True)
    os.makedirs("opt-results", exist_ok=True)
    os.makedirs("log-files", exist_ok=True)
    
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip()
        log(f"Submitted rigid-body jobs: {job_id}")
        log(f"  Total structures: {total}")
        log(f"  Chunk size: {chunk_size}")
        log(f"  Number of array jobs: {num_chunks}")
        return num_chunks, job_id
    else:
        log(f"Error submitting: {result.stderr}")
        return num_chunks, ""


def submit_cell_opt_jobs(
    input_xyz: str = "filtered_by_inter_bb.xyz",
    chunk_size: int = 10,
    script_path: str = "submit_cell_opt_auto.sh",
    slurm_account: str = "genai-mingjieliu",
    time_limit: str = "24:00:00",
    partition: str = "hpg-turin",
) -> Tuple[int, str]:
    """Generate and submit cell optimization jobs."""
    # Count structures
    total = count_structures_in_xyz(input_xyz)
    num_chunks = (total + chunk_size - 1) // chunk_size
    
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
    
    with open(script_path, 'w') as f:
        f.write(script)
    
    os.makedirs("slurmoutputs", exist_ok=True)
    
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip()
        log(f"Submitted cell-opt jobs: {job_id}")
        return num_chunks, job_id
    else:
        log(f"Error submitting: {result.stderr}")
        return num_chunks, ""


def combine_cell_opt_results(
    pattern: str = "./cell-opt-results/pred_omc25_*step_cell_opt_*.extxyz",
    output: str = "pred_uma_cell_opt_final.xyz",
) -> int:
    """Combine cell optimization results."""
    from ase.io import read, write
    
    files = sorted(glob.glob(pattern), key=lambda f: extract_indices(f))
    
    if not files:
        raise ValueError(f"No files found matching {pattern}")
    
    all_atoms = []
    for f in files:
        atoms_list = read(f, index=":")
        all_atoms.extend(atoms_list)
        log(f"  Loaded {len(atoms_list)} from {os.path.basename(f)}")
    
    write(output, all_atoms, format="extxyz")
    log(f"Combined {len(all_atoms)} structures into {output}")
    
    return len(all_atoms)


def monitor_rigid_body_phase(
    input_xyz: str,
    chunk_size: int = 50,
    check_interval: int = 300,  # 5 minutes
    max_wait_hours: float = 12,
) -> bool:
    """
    Monitor rigid-body optimization phase.
    
    Returns True when complete, False if timeout.
    """
    expected_total = count_structures_in_xyz(input_xyz)
    expected_chunks = (expected_total + chunk_size - 1) // chunk_size
    
    log(f"Monitoring rigid-body optimization...")
    log(f"  Input: {input_xyz}")
    log(f"  Expected structures: {expected_total}")
    log(f"  Expected chunks: {expected_chunks}")
    
    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600
    
    while True:
        is_complete, completed, expected = check_rigid_body_completion(
            expected_total=expected_total,
            chunk_size=chunk_size,
        )
        
        running = check_slurm_jobs_running("uma-rigidbody-opt")
        
        log(f"  Progress: {completed}/{expected} structures, {running} jobs running")
        
        if is_complete:
            log("Rigid-body optimization complete!")
            return True
        
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            log(f"Timeout after {max_wait_hours} hours")
            return False
        
        # Check if no jobs running but not complete - might be a problem
        if running == 0 and completed < expected:
            log("Warning: No jobs running but not all structures processed")
            # Give it one more check in case jobs just finished
            time.sleep(60)
            is_complete, completed, _ = check_rigid_body_completion(
                expected_total=expected_total, chunk_size=chunk_size
            )
            if is_complete:
                return True
            log("Some structures may have failed. Proceeding with available results.")
            return True
        
        time.sleep(check_interval)


def monitor_cell_opt_phase(
    expected_total: int,
    chunk_size: int = 10,
    check_interval: int = 600,  # 10 minutes
    max_wait_hours: float = 48,
) -> bool:
    """
    Monitor cell optimization phase.
    
    Returns True when complete, False if timeout.
    """
    log(f"Monitoring cell optimization...")
    log(f"  Expected structures: {expected_total}")
    
    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600
    
    while True:
        is_complete, completed, expected = check_cell_opt_completion(
            expected_total=expected_total,
            chunk_size=chunk_size,
        )
        
        running = check_slurm_jobs_running("uma-cell-opt")
        
        log(f"  Progress: {completed}/{expected} structures, {running} jobs running")
        
        if is_complete:
            log("Cell optimization complete!")
            return True
        
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            log(f"Timeout after {max_wait_hours} hours")
            return False
        
        if running == 0 and completed < expected:
            log("Warning: No jobs running but not all structures processed")
            time.sleep(60)
            is_complete, completed, _ = check_cell_opt_completion(
                expected_total=expected_total, chunk_size=chunk_size
            )
            if is_complete:
                return True
            log("Some structures may have failed. Proceeding with available results.")
            return True
        
        time.sleep(check_interval)


def run_full_pipeline(
    input_xyz: str,
    rigid_body_chunk_size: int = 50,
    cell_opt_chunk_size: int = 10,
    min_inter_bb_dist: float = 1.0,
    skip_rigid_body: bool = False,
):
    """
    Run the full pipeline from rigid-body opt through cell opt.
    
    This is a ONE-SHOT job that:
    1. Submits rigid-body optimization jobs
    2. Monitors until complete
    3. Combines and filters results
    4. Submits cell optimization jobs
    5. Monitors until complete
    6. Combines final results
    """
    log("=" * 60)
    log("UMA-OMC Relaxation Pipeline - ONE-SHOT Full Run")
    log("=" * 60)
    log(f"Input: {input_xyz}")
    log(f"Rigid-body chunk size: {rigid_body_chunk_size}")
    log(f"Cell-opt chunk size: {cell_opt_chunk_size}")
    log(f"Min inter-BB distance: {min_inter_bb_dist} Å")
    log(f"Skip rigid-body: {skip_rigid_body}")
    log("=" * 60)
    
    # Check input exists
    if not os.path.exists(input_xyz):
        log(f"ERROR: Input file not found: {input_xyz}")
        return False
    
    total_input = count_structures_in_xyz(input_xyz)
    log(f"Input file contains {total_input} structures")
    
    # Phase 0: Submit rigid-body optimization jobs
    if not skip_rigid_body:
        log("\n[Phase 0] Submitting rigid-body optimization jobs...")
        try:
            num_chunks, job_id = submit_rigid_body_jobs(
                input_xyz=input_xyz,
                chunk_size=rigid_body_chunk_size,
            )
            if not job_id:
                log("ERROR: Failed to submit rigid-body jobs")
                return False
        except Exception as e:
            log(f"ERROR: Failed to submit rigid-body jobs: {e}")
            return False
        
        # Wait for jobs to start
        log("Waiting 60s for jobs to start...")
        time.sleep(60)
        
        # Phase 1: Monitor rigid-body optimization
        log("\n[Phase 1] Monitoring rigid-body optimization...")
        success = monitor_rigid_body_phase(
            input_xyz=input_xyz,
            chunk_size=rigid_body_chunk_size,
        )
        if not success:
            log("ERROR: Rigid-body phase did not complete successfully")
            return False
    else:
        log("\n[Phase 0-1] Skipping rigid-body (--skip_rigid_body)")
    
    # Phase 2: Combine rigid-body results
    log("\n[Phase 2] Combining rigid-body results...")
    try:
        total_rigid = combine_rigid_body_results()
    except Exception as e:
        log(f"ERROR: Failed to combine rigid-body results: {e}")
        return False
    
    # Phase 3: Filter by inter-BB distance
    log("\n[Phase 3] Filtering by inter-BB distance...")
    try:
        kept, total = filter_by_inter_bb_distance(min_distance=min_inter_bb_dist)
    except Exception as e:
        log(f"ERROR: Failed to filter: {e}")
        return False
    
    if kept == 0:
        log("ERROR: No structures passed filtering!")
        return False
    
    # Phase 4: Submit cell optimization jobs
    log("\n[Phase 4] Submitting cell optimization jobs...")
    try:
        num_chunks, job_id = submit_cell_opt_jobs(
            chunk_size=cell_opt_chunk_size,
        )
    except Exception as e:
        log(f"ERROR: Failed to submit cell-opt jobs: {e}")
        return False
    
    # Wait a bit for jobs to start
    log("Waiting 60s for jobs to start...")
    time.sleep(60)
    
    # Phase 5: Monitor cell optimization
    log("\n[Phase 5] Monitoring cell optimization...")
    success = monitor_cell_opt_phase(
        expected_total=kept,
        chunk_size=cell_opt_chunk_size,
    )
    if not success:
        log("ERROR: Cell-opt phase did not complete successfully")
        return False
    
    # Phase 6: Combine cell-opt results
    log("\n[Phase 6] Combining cell-opt results...")
    try:
        total_final = combine_cell_opt_results()
    except Exception as e:
        log(f"ERROR: Failed to combine cell-opt results: {e}")
        return False
    
    log("\n" + "=" * 60)
    log("Pipeline Complete!")
    log("=" * 60)
    log(f"Initial structures: {count_structures_in_xyz(input_xyz)}")
    log(f"After rigid-body: {total_rigid}")
    log(f"After filtering: {kept}")
    log(f"Final optimized: {total_final}")
    log("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Monitor and orchestrate UMA-OMC relaxation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mode",
        choices=["monitor", "full", "check_rigid", "check_cell", "combine_rigid", 
                 "filter", "submit_cell", "combine_cell"],
        default="full",
        help="Operation mode",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="pred_combined.xyz",
        help="Input XYZ file (for full pipeline)",
    )
    parser.add_argument(
        "--rigid_chunk_size",
        type=int,
        default=50,
        help="Chunk size for rigid-body optimization",
    )
    parser.add_argument(
        "--cell_chunk_size",
        type=int,
        default=10,
        help="Chunk size for cell optimization",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=1.0,
        help="Minimum inter-BB distance for filtering",
    )
    parser.add_argument(
        "--skip_rigid_body",
        action="store_true",
        help="Skip rigid-body monitoring (assume already complete)",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Expected number of structures (for check modes)",
    )
    
    args = parser.parse_args()
    
    if args.mode == "full":
        success = run_full_pipeline(
            input_xyz=args.input,
            rigid_body_chunk_size=args.rigid_chunk_size,
            cell_opt_chunk_size=args.cell_chunk_size,
            min_inter_bb_dist=args.min_dist,
            skip_rigid_body=args.skip_rigid_body,
        )
        sys.exit(0 if success else 1)
    
    elif args.mode == "check_rigid":
        is_complete, completed, expected = check_rigid_body_completion(
            expected_total=args.expected
        )
        print(f"Rigid-body: {completed}/{expected} complete={is_complete}")
    
    elif args.mode == "check_cell":
        is_complete, completed, expected = check_cell_opt_completion(
            expected_total=args.expected
        )
        print(f"Cell-opt: {completed}/{expected} complete={is_complete}")
    
    elif args.mode == "combine_rigid":
        combine_rigid_body_results()
    
    elif args.mode == "filter":
        filter_by_inter_bb_distance(min_distance=args.min_dist)
    
    elif args.mode == "submit_cell":
        submit_cell_opt_jobs(chunk_size=args.cell_chunk_size)
    
    elif args.mode == "combine_cell":
        combine_cell_opt_results()


if __name__ == "__main__":
    main()
