#!/usr/bin/env python
"""
Crystal Structure Prediction Pipeline - High-Level Orchestrator.

This script orchestrates the full CSP workflow:
  1. Structure Generation (MolCrystalFlow) → <formula>/generated/
  2. UMA-OMC uMLIP Relaxation → <formula>/results/
  3. DFT Job Preparation → <formula>/dft_jobs/
  4. DFT Result Collection & Ranking → <formula>/dft_combined_ranking.csv

Each step can be run independently or as part of the full pipeline.

Usage:
  # Full pipeline (interactive - submits jobs and waits for completion)
  python csp_pipeline.py --xyz molecule.xyz --ckpt_path /path/to/checkpoint --full

  # Step 1: Generate structures only
  python csp_pipeline.py --xyz molecule.xyz --ckpt_path /path/to/checkpoint --generate

  # Step 2: Submit relaxation (after generation)
  python csp_pipeline.py --formula C6H6 --relax

  # Step 3: Setup DFT jobs (after relaxation)
  python csp_pipeline.py --formula C6H6 --setup_dft

  # Step 4: Collect DFT results and rank (after DFT jobs complete)
  python csp_pipeline.py --formula C6H6 --collect_dft --plot

Output Structure:
  <formula>/
  ├── molecule.xyz                 # Input conformer
  ├── generated/                   # Step 1: MolCrystalFlow outputs
  │   ├── crystals_z2.xyz
  │   ├── crystals_z4.xyz
  │   └── pred_combined.xyz
  ├── results/                     # Step 2: UMA-OMC relaxation results
  │   ├── relaxation_results.csv
  │   └── top10_for_dft.csv
  ├── dft_jobs/                    # Step 3: VASP job directories
  │   ├── pbe-d3/
  │   └── pbe-mbd/
  ├── relaxed_structures/          # Step 4: Extracted structures
  │   ├── all_relaxed/
  │   └── top_4_pbe_mbd/
  ├── dft_combined_ranking.csv     # Step 4: Final ranking
  └── stability_ranking_<formula>.pdf
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add current directory to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))


# ==============================================================================
# Step 1: Structure Generation (MolCrystalFlow)
# ==============================================================================


def run_generation(
    xyz_path: str,
    ckpt_path: str,
    output_dir: str = ".",
    z_values: List[int] = [2, 4],
    num_samples: int = 1000,
    axis_flip: bool = True,
    filter_overlap: bool = True,
    overlap_threshold: float = 0.05,
    timesteps: int = 50,
    scaling: float = 9.0,
    exp_rate: float = 3.0,
    formula: Optional[str] = None,
    device: str = "cuda",
    seed: int = 42,
) -> Dict:
    """
    Run MolCrystalFlow structure generation.
    
    Returns:
        Dictionary with generation results and output paths.
    """
    from molcrystalflow_gen.packing_gen import generate_crystal_structures

    print("\n" + "=" * 70)
    print("STEP 1: Structure Generation (MolCrystalFlow)")
    print("=" * 70)

    result = generate_crystal_structures(
        xyz_path=xyz_path,
        ckpt_path=ckpt_path,
        z_values=z_values,
        num_samples=num_samples,
        has_axis_flip=axis_flip,
        num_timesteps=timesteps,
        scaling=scaling,
        exp_rate=exp_rate,
        compute_aux_features=True,
        filter_overlap=filter_overlap,
        overlap_threshold=overlap_threshold,
        output_dir=output_dir,
        formula=formula,
        device=device,
        seed=seed,
        verbose=True,
    )

    return result


# ==============================================================================
# Step 2: UMA-OMC uMLIP Relaxation
# ==============================================================================


def submit_relaxation(
    formula: str,
    base_dir: str = ".",
    xyz_file: Optional[str] = None,
    num_jobs: int = 100,
    structures_per_job: int = 50,
    conda_env: str = "uma-s1p1-mace-omc",
    partition: str = "gpu",
    time_limit: str = "24:00:00",
    wait_for_completion: bool = False,
    poll_interval: int = 60,
) -> Dict:
    """
    Submit UMA-OMC relaxation pipeline jobs.
    
    Args:
        formula: Chemical formula (determines output directory).
        base_dir: Base directory for outputs.
        xyz_file: Input XYZ file (defaults to <formula>/generated/pred_combined.xyz).
        num_jobs: Number of parallel jobs.
        structures_per_job: Number of structures per job.
        conda_env: Conda environment for relaxation.
        partition: SLURM partition.
        time_limit: Job time limit.
        wait_for_completion: Whether to wait for jobs to complete.
        poll_interval: Polling interval in seconds (if waiting).
        
    Returns:
        Dictionary with job information.
    """
    print("\n" + "=" * 70)
    print("STEP 2: UMA-OMC uMLIP Relaxation")
    print("=" * 70)

    formula_dir = Path(base_dir) / formula
    results_dir = formula_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine input file
    if xyz_file is None:
        xyz_file = formula_dir / "generated" / "pred_combined.xyz"
    else:
        xyz_file = Path(xyz_file)

    if not xyz_file.exists():
        raise FileNotFoundError(f"Input file not found: {xyz_file}")

    print(f"  Input: {xyz_file}")
    print(f"  Output: {results_dir}")
    print(f"  Jobs: {num_jobs} × {structures_per_job} structures/job")

    # Path to relaxation pipeline script
    relax_script = SCRIPT_DIR / "uma-s1p1-omc-opt" / "run_relaxation_pipeline.py"
    monitor_script = SCRIPT_DIR / "uma-s1p1-omc-opt" / "monitor_pipeline.py"

    if not relax_script.exists():
        raise FileNotFoundError(f"Relaxation script not found: {relax_script}")

    # Build submission command
    cmd = [
        "python", str(relax_script),
        "--input", str(xyz_file),
        "--output_dir", str(results_dir),
        "--num_jobs", str(num_jobs),
        "--structures_per_job", str(structures_per_job),
        "--conda_env", conda_env,
        "--partition", partition,
        "--time", time_limit,
    ]

    print(f"\n  Submitting relaxation pipeline...")
    print(f"  Command: {' '.join(cmd)}")

    # Submit jobs
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error submitting jobs: {result.stderr}")
        return {"status": "error", "error": result.stderr}

    print(result.stdout)

    job_info = {
        "status": "submitted",
        "formula": formula,
        "input_file": str(xyz_file),
        "output_dir": str(results_dir),
        "num_jobs": num_jobs,
    }

    # Wait for completion if requested
    if wait_for_completion:
        print(f"\n  Waiting for jobs to complete (polling every {poll_interval}s)...")
        job_info["status"] = _wait_for_relaxation_completion(
            results_dir, poll_interval
        )

    return job_info


def _wait_for_relaxation_completion(
    results_dir: Path, poll_interval: int = 60
) -> str:
    """Wait for relaxation jobs to complete."""
    import glob

    while True:
        # Check for completion marker
        completion_file = results_dir / "relaxation_complete.txt"
        if completion_file.exists():
            print("  Relaxation complete!")
            return "completed"

        # Check for results CSV
        results_csv = results_dir / "relaxation_results.csv"
        if results_csv.exists():
            # Check if jobs are still running
            running_jobs = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "-h"],
                capture_output=True, text=True
            )
            if "relax" not in running_jobs.stdout.lower():
                print("  Relaxation complete!")
                return "completed"

        print(f"  Jobs still running... (checking again in {poll_interval}s)")
        time.sleep(poll_interval)


def collect_relaxation_results(
    formula: str,
    base_dir: str = ".",
    top_n: int = 10,
) -> Dict:
    """
    Collect relaxation results and select top structures for DFT.
    
    Returns:
        Dictionary with collection results.
    """
    print("\n  Collecting relaxation results...")

    formula_dir = Path(base_dir) / formula
    results_dir = formula_dir / "results"

    # Path to collection script
    collect_script = SCRIPT_DIR / "uma-s1p1-omc-opt" / "collect_relaxation_results.py"

    if not collect_script.exists():
        raise FileNotFoundError(f"Collection script not found: {collect_script}")

    cmd = [
        "python", str(collect_script),
        "--results_dir", str(results_dir),
        "--top_n", str(top_n),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error collecting results: {result.stderr}")
        return {"status": "error", "error": result.stderr}

    print(result.stdout)

    return {
        "status": "completed",
        "results_dir": str(results_dir),
        "top_n": top_n,
    }


# ==============================================================================
# Step 3: DFT Job Preparation
# ==============================================================================


def setup_dft_jobs(
    formula: str,
    base_dir: str = ".",
    top_n_csv: Optional[str] = None,
    functionals: List[str] = ["pbe-d3", "pbe-mbd"],
    vasp_settings: Optional[Dict] = None,
) -> Dict:
    """
    Setup DFT jobs for top relaxed structures.
    
    Args:
        formula: Chemical formula.
        base_dir: Base directory.
        top_n_csv: Path to top-N structures CSV (defaults to <formula>/results/top10_for_dft.csv).
        functionals: List of DFT functionals to run.
        vasp_settings: Optional VASP settings overrides.
        
    Returns:
        Dictionary with DFT job information.
    """
    print("\n" + "=" * 70)
    print("STEP 3: DFT Job Preparation")
    print("=" * 70)

    formula_dir = Path(base_dir) / formula
    dft_dir = formula_dir / "dft_jobs"
    dft_dir.mkdir(parents=True, exist_ok=True)

    # Determine input file
    if top_n_csv is None:
        top_n_csv = formula_dir / "results" / "top10_for_dft.csv"
    else:
        top_n_csv = Path(top_n_csv)

    if not top_n_csv.exists():
        raise FileNotFoundError(f"Top-N CSV not found: {top_n_csv}")

    print(f"  Input: {top_n_csv}")
    print(f"  Output: {dft_dir}")
    print(f"  Functionals: {functionals}")

    # Path to DFT setup script
    dft_setup_script = SCRIPT_DIR / "dft-run" / "setup_dft_jobs.py"

    if not dft_setup_script.exists():
        raise FileNotFoundError(f"DFT setup script not found: {dft_setup_script}")

    job_dirs = {}
    for functional in functionals:
        func_dir = dft_dir / functional
        cmd = [
            "python", str(dft_setup_script),
            "--input", str(top_n_csv),
            "--output_dir", str(func_dir),
            "--functional", functional,
        ]

        print(f"\n  Setting up {functional} jobs...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            continue

        print(result.stdout)
        job_dirs[functional] = str(func_dir)

    return {
        "status": "completed",
        "formula": formula,
        "dft_dir": str(dft_dir),
        "job_dirs": job_dirs,
        "functionals": functionals,
    }


def submit_dft_jobs(
    formula: str,
    base_dir: str = ".",
    functionals: List[str] = ["pbe-d3", "pbe-mbd"],
    partition: str = "gpu",
    time_limit: str = "48:00:00",
    wait_for_completion: bool = False,
    poll_interval: int = 300,
) -> Dict:
    """
    Submit DFT jobs to the cluster.
    
    Returns:
        Dictionary with submission information.
    """
    print("\n  Submitting DFT jobs...")

    formula_dir = Path(base_dir) / formula
    dft_dir = formula_dir / "dft_jobs"

    submitted_jobs = {}
    for functional in functionals:
        func_dir = dft_dir / functional
        if not func_dir.exists():
            print(f"  Warning: {functional} directory not found, skipping")
            continue

        # Find job directories and submit
        for job_dir in sorted(func_dir.glob("structure_*")):
            submit_script = job_dir / "submit.sh"
            if submit_script.exists():
                result = subprocess.run(
                    ["sbatch", str(submit_script)],
                    capture_output=True, text=True,
                    cwd=str(job_dir)
                )
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    submitted_jobs[f"{functional}/{job_dir.name}"] = job_id
                    print(f"    Submitted {job_dir.name}: Job {job_id}")
                else:
                    print(f"    Error submitting {job_dir.name}: {result.stderr}")

    job_info = {
        "status": "submitted",
        "formula": formula,
        "dft_dir": str(dft_dir),
        "submitted_jobs": submitted_jobs,
    }

    # Wait for completion if requested
    if wait_for_completion:
        print(f"\n  Waiting for DFT jobs to complete...")
        _wait_for_dft_completion(submitted_jobs, poll_interval)
        job_info["status"] = "completed"

    return job_info


def _wait_for_dft_completion(
    submitted_jobs: Dict[str, str], poll_interval: int = 300
) -> None:
    """Wait for DFT jobs to complete."""
    job_ids = list(submitted_jobs.values())

    while job_ids:
        # Check which jobs are still running
        result = subprocess.run(
            ["squeue", "-j", ",".join(job_ids), "-h"],
            capture_output=True, text=True
        )
        running = [jid for jid in job_ids if jid in result.stdout]

        if not running:
            print("  All DFT jobs completed!")
            return

        print(f"  {len(running)}/{len(job_ids)} jobs still running...")
        time.sleep(poll_interval)


# ==============================================================================
# Step 4: DFT Result Collection & Ranking
# ==============================================================================


def collect_dft_results(
    formula: str,
    base_dir: str = ".",
    generate_plot: bool = True,
    gt_energies_file: Optional[str] = None,
    umlip_results_file: Optional[str] = None,
    top_n_structures: int = 4,
    extract_structures: bool = True,
) -> Dict:
    """
    Collect DFT results, extract structures, and generate ranking.
    
    Args:
        formula: Chemical formula.
        base_dir: Base directory.
        generate_plot: Whether to generate stability ranking plot.
        gt_energies_file: Path to ground-truth energies JSON file.
        umlip_results_file: Path to u-MLIP results CSV (for u-MLIP column in ranking).
        top_n_structures: Number of top PBE-MBD structures to extract.
        extract_structures: Whether to extract relaxed structures as XYZ.
        
    Returns:
        Dictionary with collection results.
    """
    print("\n" + "=" * 70)
    print("STEP 4: DFT Result Collection & Ranking")
    print("=" * 70)

    formula_dir = Path(base_dir) / formula

    # Path to collection script
    collect_script = SCRIPT_DIR / "dft-run" / "collect_dft_results.py"

    if not collect_script.exists():
        raise FileNotFoundError(f"DFT collection script not found: {collect_script}")

    cmd = [
        "python", str(collect_script),
        "--formula_dir", str(formula_dir),
        "--top_n", str(top_n_structures),
    ]

    if generate_plot:
        cmd.append("--plot")

    if gt_energies_file:
        cmd.extend(["--gt_energies", gt_energies_file])

    if umlip_results_file:
        cmd.extend(["--umlip_results", umlip_results_file])
    else:
        # Default to the u-MLIP results from relaxation
        default_umlip = formula_dir / "results" / "top10_for_dft.csv"
        if default_umlip.exists():
            cmd.extend(["--umlip_results", str(default_umlip)])

    if extract_structures:
        cmd.append("--extract_structures")

    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"  Warnings: {result.stderr}")

    return {
        "status": "completed" if result.returncode == 0 else "error",
        "formula": formula,
        "formula_dir": str(formula_dir),
    }


# ==============================================================================
# Full Pipeline Orchestration
# ==============================================================================


def run_full_pipeline(
    xyz_path: str,
    ckpt_path: str,
    output_dir: str = ".",
    z_values: List[int] = [2, 4],
    num_samples: int = 1000,
    axis_flip: bool = True,
    filter_overlap: bool = True,
    overlap_threshold: float = 0.05,
    formula: Optional[str] = None,
    # Relaxation settings
    num_relax_jobs: int = 100,
    structures_per_job: int = 50,
    relax_conda_env: str = "uma-s1p1-mace-omc",
    relax_partition: str = "gpu",
    relax_time: str = "24:00:00",
    top_n_for_dft: int = 10,
    # DFT settings
    dft_functionals: List[str] = ["pbe-d3", "pbe-mbd"],
    dft_partition: str = "gpu",
    dft_time: str = "48:00:00",
    # Result collection settings
    gt_energies_file: Optional[str] = None,
    top_n_structures: int = 4,
    # General settings
    device: str = "cuda",
    seed: int = 42,
    wait_for_jobs: bool = True,
    poll_interval: int = 60,
) -> Dict:
    """
    Run the full CSP pipeline from structure generation to DFT ranking.
    
    This is a convenience function that runs all steps in sequence.
    For large-scale production use, consider running steps independently
    and monitoring job status manually.
    
    Returns:
        Dictionary with results from all steps.
    """
    print("\n" + "#" * 70)
    print("# CRYSTAL STRUCTURE PREDICTION PIPELINE")
    print("#" * 70)

    results = {}

    # Step 1: Generate structures
    gen_result = run_generation(
        xyz_path=xyz_path,
        ckpt_path=ckpt_path,
        output_dir=output_dir,
        z_values=z_values,
        num_samples=num_samples,
        axis_flip=axis_flip,
        filter_overlap=filter_overlap,
        overlap_threshold=overlap_threshold,
        formula=formula,
        device=device,
        seed=seed,
    )
    results["generation"] = gen_result
    formula = gen_result["formula"]

    # Step 2: Submit relaxation
    relax_result = submit_relaxation(
        formula=formula,
        base_dir=output_dir,
        num_jobs=num_relax_jobs,
        structures_per_job=structures_per_job,
        conda_env=relax_conda_env,
        partition=relax_partition,
        time_limit=relax_time,
        wait_for_completion=wait_for_jobs,
        poll_interval=poll_interval,
    )
    results["relaxation"] = relax_result

    if relax_result["status"] != "completed" and wait_for_jobs:
        print("\n  Warning: Relaxation did not complete successfully.")
        return results

    # Collect relaxation results
    collect_result = collect_relaxation_results(
        formula=formula,
        base_dir=output_dir,
        top_n=top_n_for_dft,
    )
    results["relaxation_collection"] = collect_result

    # Step 3: Setup and submit DFT jobs
    dft_setup_result = setup_dft_jobs(
        formula=formula,
        base_dir=output_dir,
        functionals=dft_functionals,
    )
    results["dft_setup"] = dft_setup_result

    dft_submit_result = submit_dft_jobs(
        formula=formula,
        base_dir=output_dir,
        functionals=dft_functionals,
        partition=dft_partition,
        time_limit=dft_time,
        wait_for_completion=wait_for_jobs,
        poll_interval=poll_interval * 5,  # DFT jobs take longer
    )
    results["dft_submission"] = dft_submit_result

    if dft_submit_result["status"] != "completed" and wait_for_jobs:
        print("\n  Warning: DFT jobs did not complete successfully.")
        return results

    # Step 4: Collect DFT results and generate ranking
    dft_result = collect_dft_results(
        formula=formula,
        base_dir=output_dir,
        generate_plot=True,
        gt_energies_file=gt_energies_file,
        top_n_structures=top_n_structures,
        extract_structures=True,
    )
    results["dft_results"] = dft_result

    # Final summary
    print("\n" + "#" * 70)
    print("# PIPELINE COMPLETE")
    print("#" * 70)
    print(f"\n  Formula: {formula}")
    print(f"  Output directory: {output_dir}/{formula}/")
    print(f"\n  Generated: {sum(len(s) for s in gen_result['structures'].values())} structures")
    print(f"  Top structures for DFT: {top_n_for_dft}")
    print(f"  DFT functionals: {dft_functionals}")
    print(f"\n  Results:")
    print(f"    - Stability ranking: {formula}/dft_combined_ranking.csv")
    print(f"    - Ranking plot: {formula}/stability_ranking_{formula}.pdf")
    print(f"    - Top structures: {formula}/relaxed_structures/top_{top_n_structures}_pbe_mbd/")

    return results


# ==============================================================================
# CLI Interface
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Crystal Structure Prediction Pipeline - High-Level Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (requires --xyz and --ckpt_path)
  python csp_pipeline.py --xyz molecule.xyz --ckpt_path /path/to/ckpt --full

  # Step 1: Generate structures only
  python csp_pipeline.py --xyz molecule.xyz --ckpt_path /path/to/ckpt --generate

  # Step 2: Submit relaxation (after generation)
  python csp_pipeline.py --formula C6H6 --relax

  # Step 3: Setup DFT jobs (after relaxation)
  python csp_pipeline.py --formula C6H6 --setup_dft

  # Step 4: Collect DFT results (after DFT jobs complete)
  python csp_pipeline.py --formula C6H6 --collect_dft --plot

Output Structure:
  <formula>/
  ├── molecule.xyz
  ├── generated/
  ├── results/
  ├── dft_jobs/
  ├── relaxed_structures/
  ├── dft_combined_ranking.csv
  └── stability_ranking_<formula>.pdf
        """,
    )

    # Input/output
    parser.add_argument("--xyz", "-i", type=str, help="Path to molecule XYZ file")
    parser.add_argument("--ckpt_path", "-c", type=str, help="Path to MolCrystalFlow checkpoint")
    parser.add_argument("--formula", "-f", type=str, help="Chemical formula (auto-detected from XYZ if not provided)")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="Base output directory")

    # Pipeline steps (mutually exclusive in typical usage)
    step_group = parser.add_argument_group("Pipeline Steps")
    step_group.add_argument("--full", action="store_true", help="Run full pipeline")
    step_group.add_argument("--generate", action="store_true", help="Step 1: Generate structures")
    step_group.add_argument("--relax", action="store_true", help="Step 2: Submit relaxation")
    step_group.add_argument("--setup_dft", action="store_true", help="Step 3: Setup DFT jobs")
    step_group.add_argument("--submit_dft", action="store_true", help="Step 3b: Submit DFT jobs")
    step_group.add_argument("--collect_dft", action="store_true", help="Step 4: Collect DFT results")

    # Generation settings
    gen_group = parser.add_argument_group("Generation Settings")
    gen_group.add_argument("--z_values", type=int, nargs="+", default=[2, 4], help="Z values")
    gen_group.add_argument("--num_samples", "-n", type=int, default=1000, help="Samples per Z value")
    gen_group.add_argument("--axis_flip", action="store_true", default=True, help="Enable axis flip")
    gen_group.add_argument("--no_axis_flip", action="store_false", dest="axis_flip", help="Disable axis flip")
    gen_group.add_argument("--filter_overlap", action="store_true", default=True, help="Filter by overlap")
    gen_group.add_argument("--no_filter_overlap", action="store_false", dest="filter_overlap")
    gen_group.add_argument("--overlap_threshold", type=float, default=0.05, help="Overlap threshold")
    gen_group.add_argument("--timesteps", type=int, default=50, help="Integration timesteps")
    gen_group.add_argument("--scaling", type=float, default=9.0, help="Scaling for centroid fractional coords")
    gen_group.add_argument("--exp_rate", type=float, default=3.0, help="Exp rate for rotation matrices")
    gen_group.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    gen_group.add_argument("--seed", type=int, default=42, help="Random seed")

    # Relaxation settings
    relax_group = parser.add_argument_group("Relaxation Settings")
    relax_group.add_argument("--num_relax_jobs", type=int, default=100, help="Number of relaxation jobs")
    relax_group.add_argument("--structures_per_job", type=int, default=50, help="Structures per job")
    relax_group.add_argument("--relax_conda_env", type=str, default="uma-s1p1-mace-omc", help="Conda env for relaxation")
    relax_group.add_argument("--relax_partition", type=str, default="gpu", help="SLURM partition")
    relax_group.add_argument("--relax_time", type=str, default="24:00:00", help="Relaxation time limit")
    relax_group.add_argument("--top_n_for_dft", type=int, default=10, help="Top N structures for DFT")

    # DFT settings
    dft_group = parser.add_argument_group("DFT Settings")
    dft_group.add_argument("--dft_functionals", type=str, nargs="+", default=["pbe-d3", "pbe-mbd"], help="DFT functionals")
    dft_group.add_argument("--dft_partition", type=str, default="gpu", help="DFT SLURM partition")
    dft_group.add_argument("--dft_time", type=str, default="48:00:00", help="DFT time limit")

    # Result collection settings
    collect_group = parser.add_argument_group("Result Collection Settings")
    collect_group.add_argument("--plot", action="store_true", help="Generate stability ranking plot")
    collect_group.add_argument("--gt_energies", type=str, help="Ground-truth energies JSON file")
    collect_group.add_argument("--top_n_structures", type=int, default=4, help="Top N structures to extract")
    collect_group.add_argument("--extract_structures", action="store_true", default=True, help="Extract relaxed structures")

    # Job control
    job_group = parser.add_argument_group("Job Control")
    job_group.add_argument("--wait", action="store_true", help="Wait for jobs to complete")
    job_group.add_argument("--poll_interval", type=int, default=60, help="Polling interval in seconds")

    args = parser.parse_args()

    # Validate arguments
    if args.full or args.generate:
        if not args.xyz:
            parser.error("--xyz is required for generation")
        if not args.ckpt_path:
            parser.error("--ckpt_path is required for generation")

    if args.relax or args.setup_dft or args.submit_dft or args.collect_dft:
        if not args.formula and not args.xyz:
            parser.error("--formula or --xyz is required")

    # Determine formula from XYZ if not provided
    if args.xyz and not args.formula:
        from ase.io import read
        mol = read(args.xyz)
        args.formula = mol.get_chemical_formula()
        print(f"  Auto-detected formula: {args.formula}")

    # Run requested step(s)
    if args.full:
        run_full_pipeline(
            xyz_path=args.xyz,
            ckpt_path=args.ckpt_path,
            output_dir=args.output_dir,
            z_values=args.z_values,
            num_samples=args.num_samples,
            axis_flip=args.axis_flip,
            filter_overlap=args.filter_overlap,
            overlap_threshold=args.overlap_threshold,
            formula=args.formula,
            num_relax_jobs=args.num_relax_jobs,
            structures_per_job=args.structures_per_job,
            relax_conda_env=args.relax_conda_env,
            relax_partition=args.relax_partition,
            relax_time=args.relax_time,
            top_n_for_dft=args.top_n_for_dft,
            dft_functionals=args.dft_functionals,
            dft_partition=args.dft_partition,
            dft_time=args.dft_time,
            gt_energies_file=args.gt_energies,
            top_n_structures=args.top_n_structures,
            device=args.device,
            seed=args.seed,
            wait_for_jobs=args.wait,
            poll_interval=args.poll_interval,
        )

    elif args.generate:
        run_generation(
            xyz_path=args.xyz,
            ckpt_path=args.ckpt_path,
            output_dir=args.output_dir,
            z_values=args.z_values,
            num_samples=args.num_samples,
            axis_flip=args.axis_flip,
            filter_overlap=args.filter_overlap,
            overlap_threshold=args.overlap_threshold,
            formula=args.formula,
            device=args.device,
            seed=args.seed,
        )

    elif args.relax:
        submit_relaxation(
            formula=args.formula,
            base_dir=args.output_dir,
            num_jobs=args.num_relax_jobs,
            structures_per_job=args.structures_per_job,
            conda_env=args.relax_conda_env,
            partition=args.relax_partition,
            time_limit=args.relax_time,
            wait_for_completion=args.wait,
            poll_interval=args.poll_interval,
        )
        if args.wait:
            collect_relaxation_results(
                formula=args.formula,
                base_dir=args.output_dir,
                top_n=args.top_n_for_dft,
            )

    elif args.setup_dft:
        setup_dft_jobs(
            formula=args.formula,
            base_dir=args.output_dir,
            functionals=args.dft_functionals,
        )

    elif args.submit_dft:
        submit_dft_jobs(
            formula=args.formula,
            base_dir=args.output_dir,
            functionals=args.dft_functionals,
            partition=args.dft_partition,
            time_limit=args.dft_time,
            wait_for_completion=args.wait,
            poll_interval=args.poll_interval * 5,
        )

    elif args.collect_dft:
        collect_dft_results(
            formula=args.formula,
            base_dir=args.output_dir,
            generate_plot=args.plot,
            gt_energies_file=args.gt_energies,
            top_n_structures=args.top_n_structures,
            extract_structures=args.extract_structures,
        )

    else:
        parser.print_help()
        print("\n  Please specify a pipeline step: --full, --generate, --relax, --setup_dft, --submit_dft, or --collect_dft")


if __name__ == "__main__":
    main()
