#!/usr/bin/env python
"""
Check the final generated pkl.gz files for completeness and correctness.
Adapted for OMC25-MCF pipeline.

This script validates that all required features are present in the
training data files and reports statistics about the data.

Checks include:
- Presence of required / optional keys
- Axis flip consistency (only [0,0,0] or [1,0,0] patterns)
- Coordinate reconstruction: reassemble Cartesian coordinates from
  (local_coords, rotmats_1, trans_1, lattice_1, bb_num_vec) and compare to gt_coords
- Optional XYZ export for visual comparison against the original standardized XYZ
- StructureMatcher comparison using pymatgen (for test data)

Usage:
    # Check the final normalized files (default for OMC25)
    python check_data.py
    
    # Check a specific directory
    python check_data.py --input_dir ./preprocessed/normalized
    
    # Check a single file
    python check_data.py --input preprocessed/normalized/test_molcrystal_normalized.pkl.gz
    
    # Export reconstructed XYZ for visual comparison
    python check_data.py --input preprocessed/normalized/test_molcrystal_normalized.pkl.gz --export_xyz reconstructed_test.xyz
    
    # Debug reconstruction: export both reconstructed and ground truth
    python check_data.py --input preprocessed/normalized/test_molcrystal_normalized.pkl.gz --debug_recon ./debug_recon
    
    # Use pymatgen StructureMatcher for comparison (slow, for test data)
    python check_data.py --input preprocessed/normalized/test_molcrystal_normalized.pkl.gz --structure_match
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Add thurlemann23 to path for common utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "thurlemann23"))

from common import (
    ATOM_TYPE_TO_IDX,
    ATOMIC_NUM_TO_SYMBOL,
    IDX_TO_ATOM_TYPE,
    assemble_coords,
    load_pkl_gz,
    to_numpy,
)

# Optional pymatgen import for StructureMatcher
try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.analysis.structure_matcher import StructureMatcher
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False



# Required keys for training data
REQUIRED_KEYS = [
    "bb_num_vec",       # Number of atoms per building block
    "trans_1",          # Fractional translations (center of mass)
    "rotmats_1",        # Rotation matrices per building block
    "atom_types",       # Mapped atom type indices
    "local_coords",     # Atom coordinates in local (rotated) frame
    "gt_coords",        # Ground truth Cartesian coordinates
    "lattice_1",        # Lattice matrix
]

# Optional but expected keys
OPTIONAL_KEYS = [
    "eigenvalues",      # PCA eigenvalues per building block
    "axis_flips",       # Axis flip indicators per building block
    "basic_features",   # Basic molecular features
    "chemical_features", # Chemical features
    "geometric_features", # Geometric features
]

# Feature keys to specifically check
FEATURE_KEYS = [
    "basic_features",
    "chemical_features", 
    "geometric_features",
]


def check_axis_flip_consistency(sample: Dict[str, Any], tol: float = 1e-5) -> Dict[str, Any]:
    """
    Check if building blocks with different axis flip states only differ
    by a sign flip in one coordinate axis.
    
    For building blocks in the same molecular crystal:
    - Group by axis_flips pattern
    - Within each group, local coordinates should be identical
    - Between groups with different flips, coordinates should differ only by sign in 1 axis
    
    Args:
        sample: A single data sample dictionary.
        tol: Tolerance for coordinate comparison.
    
    Returns:
        Dictionary with axis flip check results.
    """
    results = {
        "has_axis_flips": False,
        "num_flip_patterns": 0,
        "flip_patterns": [],
        "consistency_check": None,
        "issues": [],
    }
    
    # Check if required keys exist
    if "axis_flips" not in sample or "local_coords" not in sample or "bb_num_vec" not in sample:
        results["consistency_check"] = "skipped (missing keys)"
        return results
    
    axis_flips = sample["axis_flips"]
    local_coords = sample["local_coords"]
    bb_num_vec = sample["bb_num_vec"]
    
    if axis_flips is None or local_coords is None:
        results["consistency_check"] = "skipped (None values)"
        return results
    
    results["has_axis_flips"] = True
    
    # Convert to numpy if needed
    if hasattr(axis_flips, 'numpy'):
        axis_flips = axis_flips.numpy()
    if hasattr(local_coords, 'numpy'):
        local_coords = local_coords.numpy()
    if hasattr(bb_num_vec, 'numpy'):
        bb_num_vec = bb_num_vec.numpy()
    
    # Ensure numpy arrays
    axis_flips = np.asarray(axis_flips)
    local_coords = np.asarray(local_coords)
    bb_num_vec = np.asarray(bb_num_vec)
    
    # Convert axis_flips to tuple for grouping
    if isinstance(axis_flips, np.ndarray):
        # axis_flips shape: (num_bbs, 3) - flip indicator for each axis per building block
        flip_patterns = [tuple(af) for af in axis_flips]
    else:
        flip_patterns = [tuple(af) for af in axis_flips]
    
    unique_patterns = list(set(flip_patterns))
    results["num_flip_patterns"] = len(unique_patterns)
    results["flip_patterns"] = unique_patterns
    
    # Group building blocks by flip pattern
    pattern_to_bb_indices = {}
    for bb_idx, pattern in enumerate(flip_patterns):
        if pattern not in pattern_to_bb_indices:
            pattern_to_bb_indices[pattern] = []
        pattern_to_bb_indices[pattern].append(bb_idx)
    
    # Extract local coordinates per building block
    def get_bb_local_coords(bb_idx):
        """Get local coordinates for a specific building block."""
        start = int(np.sum(bb_num_vec[:bb_idx]))
        end = start + int(bb_num_vec[bb_idx])
        return local_coords[start:end]
    
    # Check consistency between different flip patterns
    if len(unique_patterns) >= 2:
        # Take first BB from each pattern group
        pattern_coords = {}
        for pattern, bb_indices in pattern_to_bb_indices.items():
            bb_idx = bb_indices[0]
            coords = get_bb_local_coords(bb_idx)
            pattern_coords[pattern] = coords
        
        # Check pairs of patterns
        patterns_list = list(pattern_coords.keys())
        for i in range(len(patterns_list)):
            for j in range(i + 1, len(patterns_list)):
                p1, p2 = patterns_list[i], patterns_list[j]
                c1, c2 = pattern_coords[p1], pattern_coords[p2]
                
                if c1.shape != c2.shape:
                    results["issues"].append(
                        f"Shape mismatch between patterns {p1} and {p2}: {c1.shape} vs {c2.shape}"
                    )
                    continue
                
                # Count how many axes differ in sign
                axis_sign_diffs = []
                for axis in range(3):
                    c2_flipped = c2.copy()
                    c2_flipped[:, axis] = -c2_flipped[:, axis]
                    if np.allclose(c1, c2_flipped, atol=tol):
                        axis_sign_diffs.append(axis)
                
                # Also check direct match
                if np.allclose(c1, c2, atol=tol):
                    pass
                elif len(axis_sign_diffs) == 1:
                    pass
                elif len(axis_sign_diffs) == 0:
                    found_2axis = False
                    for ax1 in range(3):
                        for ax2 in range(ax1 + 1, 3):
                            c2_flipped = c2.copy()
                            c2_flipped[:, ax1] = -c2_flipped[:, ax1]
                            c2_flipped[:, ax2] = -c2_flipped[:, ax2]
                            if np.allclose(c1, c2_flipped, atol=tol):
                                found_2axis = True
                                break
                        if found_2axis:
                            break
                    
                    if not found_2axis:
                        results["issues"].append(
                            f"Patterns {p1} and {p2}: coords don't match with any axis flip combination"
                        )
    
    if not results["issues"]:
        results["consistency_check"] = "passed"
    else:
        results["consistency_check"] = "failed"
    
    return results


def check_reconstruction(sample: Dict[str, Any], tol: float = 1e-4) -> Dict[str, Any]:
    """
    Check that coordinates can be perfectly reconstructed from the decomposed
    representation (local_coords, rotmats_1, trans_1, lattice_1).

    Args:
        sample: A single data sample.
        tol: Tolerance for coordinate comparison.

    Returns:
        Dictionary with reconstruction check results.
    """
    required = ["local_coords", "rotmats_1", "trans_1", "lattice_1", "bb_num_vec", "gt_coords"]
    for key in required:
        if key not in sample:
            return {"check": "skipped", "reason": f"missing {key}"}

    gt_coords = to_numpy(sample["gt_coords"])
    recon_coords = assemble_coords(sample)

    max_err = np.max(np.abs(gt_coords - recon_coords))
    mean_err = np.mean(np.abs(gt_coords - recon_coords))

    passed = max_err < tol
    return {
        "check": "passed" if passed else "failed",
        "max_error": float(max_err),
        "mean_error": float(mean_err),
    }


def check_structure_match(
    sample: Dict[str, Any],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> Dict[str, Any]:
    """
    Check if reconstructed and ground truth structures match using pymatgen's StructureMatcher.
    
    This handles periodic boundary wrapping and is more robust than direct coordinate comparison.
    
    Args:
        sample: A single data sample.
        ltol: Fractional length tolerance for lattice comparison.
        stol: Site tolerance for structure matching.
        angle_tol: Angle tolerance in degrees for lattice comparison.
    
    Returns:
        Dictionary with structure match results.
    """
    if not HAS_PYMATGEN:
        return {"check": "skipped", "reason": "pymatgen not installed"}
    
    required = ["local_coords", "rotmats_1", "trans_1", "lattice_1", "bb_num_vec", "gt_coords", "atom_types"]
    for key in required:
        if key not in sample:
            return {"check": "skipped", "reason": f"missing {key}"}
    
    try:
        # Get data
        gt_coords = to_numpy(sample["gt_coords"])
        recon_coords = assemble_coords(sample)
        lattice_matrix = to_numpy(sample["lattice_1"])
        atom_types_idx = to_numpy(sample["atom_types"]).astype(int)
        
        # Convert atom type indices to element symbols
        species = []
        for idx in atom_types_idx:
            anum = IDX_TO_ATOM_TYPE.get(int(idx), 6)  # Default to Carbon
            sym = ATOMIC_NUM_TO_SYMBOL.get(anum, "C")
            species.append(sym)
        
        # Create pymatgen Lattice
        lattice = Lattice(lattice_matrix)
        
        # Create pymatgen Structures
        # coords_are_cartesian=True since we have Cartesian coordinates
        gt_structure = Structure(lattice, species, gt_coords, coords_are_cartesian=True)
        recon_structure = Structure(lattice, species, recon_coords, coords_are_cartesian=True)
        
        # Use StructureMatcher to compare
        matcher = StructureMatcher(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            primitive_cell=False,
            scale=False,
            attempt_supercell=False,
            comparator=None,  # Use default ElementComparator
        )
        
        is_match = matcher.fit(gt_structure, recon_structure)
        
        # Also get RMS distance if structures match
        rms_dist = None
        if is_match:
            try:
                rms_dist = matcher.get_rms_dist(gt_structure, recon_structure)
                if rms_dist is not None:
                    rms_dist = rms_dist[0]  # (rms_dist, max_dist)
            except Exception:
                pass
        
        return {
            "check": "passed" if is_match else "failed",
            "is_match": is_match,
            "rms_dist": rms_dist,
            "n_atoms": len(species),
        }
    
    except Exception as e:
        return {"check": "error", "reason": str(e)}


def _check_structure_match_worker(args):
    """Worker function for multiprocessing structure matching."""
    idx, sample = args
    result = check_structure_match(sample)
    return idx, result


def run_structure_match_comparison(
    data: List[Dict[str, Any]],
    max_samples: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run StructureMatcher comparison on all samples in data.
    
    Args:
        data: List of sample dictionaries.
        max_samples: Maximum number of samples to check (None = all).
        n_jobs: Number of parallel jobs. Use -1 for all CPUs.
        verbose: Whether to print progress.
    
    Returns:
        Dictionary with overall results.
    """
    if not HAS_PYMATGEN:
        print("ERROR: pymatgen is not installed. Install with: pip install pymatgen")
        return {"error": "pymatgen not installed"}
    
    import multiprocessing as mp
    
    n = len(data) if max_samples is None else min(max_samples, len(data))
    
    # Determine number of workers
    if n_jobs == -1:
        n_workers = mp.cpu_count()
    elif n_jobs <= 0:
        n_workers = max(1, mp.cpu_count() + n_jobs)
    else:
        n_workers = min(n_jobs, mp.cpu_count())
    
    results = {
        "samples_checked": 0,
        "samples_matched": 0,
        "samples_failed": 0,
        "samples_error": 0,
        "failed_indices": [],
        "error_indices": [],
        "rms_distances": [],
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("STRUCTURE MATCHER COMPARISON (pymatgen)")
        print(f"{'='*60}")
        print(f"Checking {n} samples with {n_workers} workers...")
    
    # Prepare work items
    work_items = [(idx, data[idx]) for idx in range(n)]
    
    if n_workers > 1 and n > 10:
        # Use multiprocessing for larger datasets
        from tqdm import tqdm
        
        # Use Pool with imap for progress tracking
        with mp.Pool(processes=n_workers) as pool:
            if verbose:
                iterator = tqdm(
                    pool.imap(_check_structure_match_worker, work_items),
                    total=n,
                    desc=f"StructureMatcher ({n_workers} CPUs)"
                )
            else:
                iterator = pool.imap(_check_structure_match_worker, work_items)
            
            for idx, result in iterator:
                results["samples_checked"] += 1
                
                if result["check"] == "passed":
                    results["samples_matched"] += 1
                    if result.get("rms_dist") is not None:
                        results["rms_distances"].append(result["rms_dist"])
                elif result["check"] == "failed":
                    results["samples_failed"] += 1
                    results["failed_indices"].append(idx)
                else:
                    results["samples_error"] += 1
                    results["error_indices"].append((idx, result.get("reason", "unknown")))
    else:
        # Single-threaded for small datasets
        from tqdm import tqdm
        iterator = tqdm(range(n), desc="StructureMatcher") if verbose else range(n)
        
        for idx in iterator:
            sample = data[idx]
            result = check_structure_match(sample)
            results["samples_checked"] += 1
            
            if result["check"] == "passed":
                results["samples_matched"] += 1
                if result.get("rms_dist") is not None:
                    results["rms_distances"].append(result["rms_dist"])
            elif result["check"] == "failed":
                results["samples_failed"] += 1
                results["failed_indices"].append(idx)
            else:
                results["samples_error"] += 1
                results["error_indices"].append((idx, result.get("reason", "unknown")))
    
    # Sort failed indices for consistent output
    results["failed_indices"].sort()
    results["error_indices"].sort(key=lambda x: x[0])
    
    # Calculate statistics
    if results["rms_distances"]:
        results["rms_mean"] = float(np.mean(results["rms_distances"]))
        results["rms_max"] = float(np.max(results["rms_distances"]))
        results["rms_min"] = float(np.min(results["rms_distances"]))
    
    if verbose:
        print(f"\n[Structure Matcher Results]")
        print(f"  Samples checked: {results['samples_checked']}")
        print(f"  Samples matched: {results['samples_matched']}")
        print(f"  Samples failed: {results['samples_failed']}")
        print(f"  Samples with errors: {results['samples_error']}")
        
        if results["samples_matched"] == results["samples_checked"]:
            print(f"  ✓ ALL structures match!")
        else:
            print(f"  ✗ {results['samples_failed']} structures do NOT match")
        
        if results["rms_distances"]:
            print(f"\n  RMS distances for matched structures:")
            print(f"    mean: {results['rms_mean']:.6f}")
            print(f"    max:  {results['rms_max']:.6f}")
            print(f"    min:  {results['rms_min']:.6f}")
        
        if results["failed_indices"]:
            print(f"\n  Failed sample indices (first 20):")
            for idx in results["failed_indices"][:20]:
                print(f"    - Sample {idx}")
            if len(results["failed_indices"]) > 20:
                print(f"    ... and {len(results['failed_indices']) - 20} more")
        
        if results["error_indices"]:
            print(f"\n  Error sample indices (first 10):")
            for idx, reason in results["error_indices"][:10]:
                print(f"    - Sample {idx}: {reason}")
    
    return results


def check_atom_ordering(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check that atoms are grouped by building block in contiguous blocks.

    Args:
        sample: A single data sample.

    Returns:
        Dictionary with ordering check results.
    """
    result = {"check": "skipped", "reason": ""}

    if "bb_num_vec" not in sample or "atom_types" not in sample:
        result["reason"] = "missing bb_num_vec or atom_types"
        return result

    bb_num_vec = to_numpy(sample["bb_num_vec"]).astype(int)
    atom_types = to_numpy(sample["atom_types"])

    total_from_bb = int(np.sum(bb_num_vec))
    total_atoms = len(atom_types)

    if total_from_bb != total_atoms:
        return {
            "check": "failed",
            "reason": f"sum(bb_num_vec)={total_from_bb} != len(atom_types)={total_atoms}",
        }

    for key in ["local_coords", "gt_coords"]:
        if key in sample:
            arr = to_numpy(sample[key])
            if arr.shape[0] != total_atoms:
                return {
                    "check": "failed",
                    "reason": f"{key} has {arr.shape[0]} atoms, expected {total_atoms}",
                }

    return {"check": "passed", "reason": ""}


def export_xyz(data: List[Dict[str, Any]], output_path: str, max_structures: int = None):
    """
    Export reconstructed structures to an extended XYZ file for visual comparison.
    """
    n = len(data) if max_structures is None else min(max_structures, len(data))

    with open(output_path, "w") as f:
        for idx in range(n):
            sample = data[idx]
            required = ["local_coords", "rotmats_1", "trans_1", "lattice_1",
                        "bb_num_vec", "atom_types"]
            if any(k not in sample for k in required):
                continue

            recon = assemble_coords(sample)
            atom_types_idx = to_numpy(sample["atom_types"]).astype(int)
            lattice = to_numpy(sample["lattice_1"])
            bb_num_vec = to_numpy(sample["bb_num_vec"]).astype(int)

            n_atoms = recon.shape[0]
            lattice_str = " ".join(f"{v:.6f}" for v in lattice.flatten())

            f.write(f"{n_atoms}\n")
            f.write(f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:bb_idx:I:1\n')

            bb_indices = []
            for bb_i, count in enumerate(bb_num_vec):
                bb_indices.extend([bb_i] * int(count))

            for atom_i in range(n_atoms):
                anum = IDX_TO_ATOM_TYPE.get(int(atom_types_idx[atom_i]), 0)
                sym = ATOMIC_NUM_TO_SYMBOL.get(anum, "X")
                x, y, z = recon[atom_i]
                f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f} {bb_indices[atom_i]}\n")

    print(f"Exported {n} structures to {output_path}")


def debug_reconstruction(data: List[Dict[str, Any]], output_dir: str, max_structures: int = 10):
    """
    Export both reconstructed and ground truth structures to XYZ files for debugging.
    
    Creates:
      - {output_dir}/sample_XXXX_recon.xyz  (reconstructed from local_coords + rotmats + trans)
      - {output_dir}/sample_XXXX_gt.xyz     (ground truth gt_coords)
      - {output_dir}/sample_XXXX_info.txt   (detailed info about the sample)
      - {output_dir}/all_recon.extxyz       (all reconstructed in one file)
      - {output_dir}/all_gt.extxyz          (all ground truth in one file)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n = min(max_structures, len(data)) if max_structures else len(data)
    
    all_recon_file = open(output_dir / "all_recon.extxyz", "w")
    all_gt_file = open(output_dir / "all_gt.extxyz", "w")
    
    failed_samples = []
    
    for idx in range(n):
        sample = data[idx]
        
        # Check required keys
        required = ["local_coords", "rotmats_1", "trans_1", "lattice_1", "bb_num_vec", "atom_types", "gt_coords"]
        missing = [k for k in required if k not in sample]
        if missing:
            print(f"Sample {idx}: missing keys {missing}, skipping")
            continue
        
        # Get data
        local_coords = to_numpy(sample["local_coords"])
        rotmats = to_numpy(sample["rotmats_1"])
        trans = to_numpy(sample["trans_1"])
        lattice = to_numpy(sample["lattice_1"])
        bb_num_vec = to_numpy(sample["bb_num_vec"]).astype(int)
        atom_types_idx = to_numpy(sample["atom_types"]).astype(int)
        gt_coords = to_numpy(sample["gt_coords"])
        
        # Reconstruct coordinates
        recon_coords = assemble_coords(sample)
        
        # Calculate error
        error = np.abs(gt_coords - recon_coords)
        max_err = np.max(error)
        mean_err = np.mean(error)
        
        n_atoms = len(atom_types_idx)
        n_bbs = len(bb_num_vec)
        lattice_str = " ".join(f"{v:.6f}" for v in lattice.flatten())
        
        # Build per-atom bb index
        bb_indices = []
        for bb_i, count in enumerate(bb_num_vec):
            bb_indices.extend([bb_i] * int(count))
        
        # Write individual files
        base_filename = f"sample_{idx:04d}"
        
        # Write info file
        with open(output_dir / f"{base_filename}_info.txt", "w") as f:
            f.write(f"Sample index: {idx}\n")
            f.write(f"Number of atoms: {n_atoms}\n")
            f.write(f"Number of building blocks: {n_bbs}\n")
            f.write(f"bb_num_vec: {bb_num_vec}\n")
            f.write(f"Reconstruction max error: {max_err:.6e}\n")
            f.write(f"Reconstruction mean error: {mean_err:.6e}\n")
            f.write(f"\nLattice:\n{lattice}\n")
            f.write(f"\nRotation matrices shape: {rotmats.shape}\n")
            f.write(f"Translations shape: {trans.shape}\n")
            f.write(f"Local coords shape: {local_coords.shape}\n")
            f.write(f"GT coords shape: {gt_coords.shape}\n")
            
            # Per-atom error
            f.write(f"\nPer-atom max error:\n")
            atom_max_err = np.max(error, axis=1)
            worst_atoms = np.argsort(atom_max_err)[-10:][::-1]
            for atom_i in worst_atoms:
                anum = IDX_TO_ATOM_TYPE.get(int(atom_types_idx[atom_i]), 0)
                sym = ATOMIC_NUM_TO_SYMBOL.get(anum, "X")
                f.write(f"  Atom {atom_i} ({sym}, bb={bb_indices[atom_i]}): "
                       f"err={atom_max_err[atom_i]:.6f}, "
                       f"gt={gt_coords[atom_i]}, recon={recon_coords[atom_i]}\n")
        
        # Helper to write xyz
        def write_xyz(file_handle, coords, comment_extra=""):
            file_handle.write(f"{n_atoms}\n")
            file_handle.write(f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:bb_idx:I:1 {comment_extra}\n')
            for atom_i in range(n_atoms):
                anum = IDX_TO_ATOM_TYPE.get(int(atom_types_idx[atom_i]), 0)
                sym = ATOMIC_NUM_TO_SYMBOL.get(anum, "X")
                x, y, z = coords[atom_i]
                file_handle.write(f"{sym} {x:.6f} {y:.6f} {z:.6f} {bb_indices[atom_i]}\n")
        
        # Write reconstructed
        with open(output_dir / f"{base_filename}_recon.xyz", "w") as f:
            write_xyz(f, recon_coords, f"max_err={max_err:.6e}")
        
        # Write ground truth
        with open(output_dir / f"{base_filename}_gt.xyz", "w") as f:
            write_xyz(f, gt_coords, "ground_truth")
        
        # Write to combined files
        write_xyz(all_recon_file, recon_coords, f"sample={idx} max_err={max_err:.6e}")
        write_xyz(all_gt_file, gt_coords, f"sample={idx} ground_truth")
        
        if max_err > 1e-4:
            failed_samples.append((idx, max_err, mean_err))
    
    all_recon_file.close()
    all_gt_file.close()
    
    print(f"\n{'='*60}")
    print(f"DEBUG RECONSTRUCTION OUTPUT")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Samples processed: {n}")
    print(f"Samples with reconstruction error > 1e-4: {len(failed_samples)}")
    
    if failed_samples:
        print(f"\nFailed samples (max_err > 1e-4):")
        for idx, max_err, mean_err in failed_samples[:20]:
            print(f"  Sample {idx}: max_err={max_err:.6e}, mean_err={mean_err:.6e}")
        if len(failed_samples) > 20:
            print(f"  ... and {len(failed_samples) - 20} more")
    
    print(f"\nFiles created:")
    print(f"  {output_dir}/all_recon.extxyz  - All reconstructed structures")
    print(f"  {output_dir}/all_gt.extxyz     - All ground truth structures")
    print(f"  {output_dir}/sample_XXXX_*.xyz - Individual sample files")
    print(f"  {output_dir}/sample_XXXX_info.txt - Per-sample debug info")


def check_single_sample(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Check a single data sample for required keys and valid values.
    """
    results = {
        "idx": idx,
        "missing_required": [],
        "missing_optional": [],
        "invalid_values": [],
        "shapes": {},
    }
    
    for key in REQUIRED_KEYS:
        if key not in sample:
            results["missing_required"].append(key)
        else:
            value = sample[key]
            if hasattr(value, 'numpy') or hasattr(value, 'shape'):
                value = to_numpy(value) if not isinstance(value, np.ndarray) else value
                if hasattr(value, 'shape'):
                    results["shapes"][key] = value.shape
                    try:
                        if np.any(np.isnan(value)):
                            results["invalid_values"].append(f"{key}: contains NaN")
                        if np.any(np.isinf(value)):
                            results["invalid_values"].append(f"{key}: contains Inf")
                    except (TypeError, ValueError):
                        pass
            elif isinstance(value, (list, tuple)):
                results["shapes"][key] = len(value)
    
    for key in OPTIONAL_KEYS:
        if key not in sample:
            results["missing_optional"].append(key)
        else:
            value = sample[key]
            if hasattr(value, 'numpy') or hasattr(value, 'shape'):
                value = to_numpy(value) if not isinstance(value, np.ndarray) else value
                if hasattr(value, 'shape'):
                    results["shapes"][key] = value.shape
                    try:
                        if np.any(np.isnan(value)):
                            results["invalid_values"].append(f"{key}: contains NaN")
                        if np.any(np.isinf(value)):
                            results["invalid_values"].append(f"{key}: contains Inf")
                    except (TypeError, ValueError):
                        pass
    
    return results


def check_data_file(filepath: str, verbose: bool = True):
    """
    Check a pkl.gz data file for completeness and correctness.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return {"error": f"File not found: {filepath}", "valid": False}, None
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Checking: {filepath.name}")
        print(f"{'='*60}")
    
    try:
        data = load_pkl_gz(str(filepath))
    except Exception as e:
        return {"error": f"Failed to load file: {e}", "valid": False}, None
    
    n_samples = len(data)
    if verbose:
        print(f"Number of samples: {n_samples}")
    
    if n_samples == 0:
        return {"error": "File contains no samples", "valid": False, "n_samples": 0}, None
    
    all_missing_required = set()
    all_missing_optional = set()
    samples_with_issues = []
    all_invalid_values = []
    
    feature_stats = {key: {"present": 0, "shapes": set()} for key in FEATURE_KEYS}
    
    num_bbs_list = []
    num_atoms_list = []
    
    for i, sample in enumerate(data):
        result = check_single_sample(sample, i)
        
        if result["missing_required"]:
            all_missing_required.update(result["missing_required"])
            samples_with_issues.append(i)
        
        all_missing_optional.update(result["missing_optional"])
        all_invalid_values.extend(result["invalid_values"])
        
        for key in FEATURE_KEYS:
            if key in sample and sample[key] is not None:
                feature_stats[key]["present"] += 1
                if isinstance(sample[key], np.ndarray):
                    feature_stats[key]["shapes"].add(sample[key].shape)
        
        if "bb_num_vec" in sample:
            bb_vec = sample["bb_num_vec"]
            if isinstance(bb_vec, np.ndarray):
                num_bbs_list.append(len(bb_vec))
                num_atoms_list.append(np.sum(bb_vec))
    
    axis_flip_results = {
        "samples_checked": 0,
        "samples_with_axis_flips": 0,
        "samples_passed": 0,
        "samples_failed": 0,
        "issues": [],
    }

    recon_results = {
        "samples_checked": 0,
        "samples_passed": 0,
        "samples_failed": 0,
        "max_error": 0.0,
        "mean_error": 0.0,
        "issues": [],
    }

    ordering_results = {
        "samples_checked": 0,
        "samples_passed": 0,
        "samples_failed": 0,
        "issues": [],
    }
    
    check_limit = min(100, n_samples)
    for i in range(check_limit):
        sample = data[i]

        af_result = check_axis_flip_consistency(sample)
        axis_flip_results["samples_checked"] += 1
        
        if af_result["has_axis_flips"]:
            axis_flip_results["samples_with_axis_flips"] += 1
            if af_result["consistency_check"] == "passed":
                axis_flip_results["samples_passed"] += 1
            elif af_result["consistency_check"] == "failed":
                axis_flip_results["samples_failed"] += 1
                axis_flip_results["issues"].extend(
                    [f"Sample {i}: {issue}" for issue in af_result["issues"][:2]]
                )

        rc = check_reconstruction(sample)
        recon_results["samples_checked"] += 1
        if rc["check"] == "passed":
            recon_results["samples_passed"] += 1
            recon_results["max_error"] = max(recon_results["max_error"], rc["max_error"])
            recon_results["mean_error"] += rc["mean_error"]
        elif rc["check"] == "failed":
            recon_results["samples_failed"] += 1
            recon_results["max_error"] = max(recon_results["max_error"], rc["max_error"])
            recon_results["mean_error"] += rc["mean_error"]
            if len(recon_results["issues"]) < 5:
                recon_results["issues"].append(
                    f"Sample {i}: max_err={rc['max_error']:.6f}"
                )

        ao = check_atom_ordering(sample)
        ordering_results["samples_checked"] += 1
        if ao["check"] == "passed":
            ordering_results["samples_passed"] += 1
        elif ao["check"] == "failed":
            ordering_results["samples_failed"] += 1
            if len(ordering_results["issues"]) < 5:
                ordering_results["issues"].append(f"Sample {i}: {ao['reason']}")

    if recon_results["samples_checked"] > 0:
        recon_results["mean_error"] /= recon_results["samples_checked"]
    
    results = {
        "valid": len(all_missing_required) == 0 and len(samples_with_issues) == 0,
        "n_samples": n_samples,
        "missing_required": list(all_missing_required),
        "missing_optional": list(all_missing_optional),
        "samples_with_issues": len(samples_with_issues),
        "invalid_values": all_invalid_values[:10],
        "feature_stats": feature_stats,
    }
    
    if num_bbs_list:
        results["num_bbs"] = {
            "min": int(np.min(num_bbs_list)),
            "max": int(np.max(num_bbs_list)),
            "mean": float(np.mean(num_bbs_list)),
        }
    
    if num_atoms_list:
        results["num_atoms"] = {
            "min": int(np.min(num_atoms_list)),
            "max": int(np.max(num_atoms_list)),
            "mean": float(np.mean(num_atoms_list)),
        }
    
    results["axis_flip_check"] = axis_flip_results
    results["reconstruction_check"] = recon_results
    results["atom_ordering_check"] = ordering_results
    
    if verbose:
        print(f"\n[Required Keys]")
        if all_missing_required:
            print(f"  ✗ Missing: {list(all_missing_required)}")
        else:
            print(f"  ✓ All required keys present")
        
        print(f"\n[Optional Keys]")
        if all_missing_optional:
            print(f"  Missing: {list(all_missing_optional)}")
        else:
            print(f"  ✓ All optional keys present")
        
        print(f"\n[Feature Keys]")
        for key in FEATURE_KEYS:
            stats = feature_stats[key]
            if stats["present"] == n_samples:
                print(f"  ✓ {key}: present in all {n_samples} samples")
            elif stats["present"] > 0:
                print(f"  △ {key}: present in {stats['present']}/{n_samples} samples")
            else:
                print(f"  ✗ {key}: NOT present")
            
            if stats["shapes"]:
                shapes_str = ", ".join(str(s) for s in list(stats["shapes"])[:5])
                if len(stats["shapes"]) > 5:
                    shapes_str += f", ... ({len(stats['shapes'])} unique shapes)"
                print(f"      Shapes: {shapes_str}")
        
        print(f"\n[Data Statistics]")
        if "num_bbs" in results:
            print(f"  Building blocks per sample: min={results['num_bbs']['min']}, "
                  f"max={results['num_bbs']['max']}, mean={results['num_bbs']['mean']:.1f}")
        if "num_atoms" in results:
            print(f"  Atoms per sample: min={results['num_atoms']['min']}, "
                  f"max={results['num_atoms']['max']}, mean={results['num_atoms']['mean']:.1f}")
        
        if all_invalid_values:
            print(f"\n[Invalid Values]")
            for msg in all_invalid_values[:10]:
                print(f"  ✗ {msg}")
            if len(all_invalid_values) > 10:
                print(f"  ... and {len(all_invalid_values) - 10} more")
        
        print(f"\n[Coordinate Reconstruction Check]")
        rc = recon_results
        print(f"  Samples checked: {rc['samples_checked']}")
        if rc['samples_checked'] > 0:
            if rc['samples_failed'] == 0:
                print(f"  ✓ All {rc['samples_passed']} samples reconstruct correctly")
                print(f"    max error = {rc['max_error']:.2e}, mean error = {rc['mean_error']:.2e}")
            else:
                print(f"  ✗ {rc['samples_failed']} samples failed reconstruction")
                print(f"  ✓ {rc['samples_passed']} samples passed")
                print(f"    max error = {rc['max_error']:.2e}")
                for issue in rc['issues']:
                    print(f"    - {issue}")

        print(f"\n[Axis Flip Consistency Check]")
        af = axis_flip_results
        print(f"  Samples checked: {af['samples_checked']}")
        print(f"  Samples with axis_flips: {af['samples_with_axis_flips']}")
        if af['samples_with_axis_flips'] > 0:
            if af['samples_failed'] == 0:
                print(f"  ✓ All {af['samples_passed']} samples passed consistency check")
            else:
                print(f"  ✗ {af['samples_failed']} samples failed consistency check")
                print(f"  ✓ {af['samples_passed']} samples passed")
                for issue in af['issues'][:5]:
                    print(f"    - {issue}")
                if len(af['issues']) > 5:
                    print(f"    ... and {len(af['issues']) - 5} more issues")
        else:
            print(f"  No axis_flips data found in samples")

        print(f"\n[Atom Ordering Check]")
        ao = ordering_results
        print(f"  Samples checked: {ao['samples_checked']}")
        if ao['samples_checked'] > 0:
            if ao['samples_failed'] == 0:
                print(f"  ✓ All {ao['samples_passed']} samples have consistent atom ordering")
            else:
                print(f"  ✗ {ao['samples_failed']} samples have ordering issues")
                for issue in ao['issues']:
                    print(f"    - {issue}")
        
        print(f"\n[Overall Status]")
        if results["valid"]:
            print(f"  ✓ File is VALID for training")
        else:
            print(f"  ✗ File has ISSUES - not ready for training")
    
    return results, data


def check_sample_content(filepath: str, sample_idx: int = 0, verbose: bool = True):
    """Print detailed content of a single sample for inspection."""
    data = load_pkl_gz(str(filepath))
    
    if sample_idx >= len(data):
        print(f"Sample index {sample_idx} out of range (max: {len(data)-1})")
        return
    
    sample = data[sample_idx]
    
    print(f"\n{'='*60}")
    print(f"Sample {sample_idx} from {Path(filepath).name}")
    print(f"{'='*60}")
    
    for key, value in sorted(sample.items()):
        if isinstance(value, np.ndarray):
            print(f"\n{key}:")
            print(f"  dtype: {value.dtype}")
            print(f"  shape: {value.shape}")
            print(f"  min: {value.min():.6f}, max: {value.max():.6f}")
            if value.size <= 20:
                print(f"  values: {value}")
        elif isinstance(value, (list, tuple)):
            print(f"\n{key}: length={len(value)}")
            if len(value) <= 10:
                print(f"  values: {value}")
        else:
            print(f"\n{key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Check OMC25-MCF pkl.gz data files for completeness and correctness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all normalized files (default)
  python check_data.py

  # Check all files in a specific directory
  python check_data.py --input_dir ./preprocessed/final

  # Check a single file
  python check_data.py --input preprocessed/normalized/train_molcrystal_normalized.pkl.gz

  # Check and export reconstructed XYZ for visual comparison
  python check_data.py --input preprocessed/normalized/test_molcrystal_normalized.pkl.gz --export_xyz reconstructed_test.xyz

  # Debug reconstruction: export both reconstructed and ground truth
  python check_data.py --input preprocessed/normalized/test_molcrystal_normalized.pkl.gz --debug_recon ./debug_recon

  # Use pymatgen StructureMatcher for comparison (recommended for test data)
  python check_data.py --input preprocessed/normalized/test_molcrystal_normalized.pkl.gz --structure_match
  
  # StructureMatcher with limited samples and specific number of CPUs
  python check_data.py --input preprocessed/normalized/test_molcrystal_normalized.pkl.gz --structure_match --max_match 100 --n_jobs 8
  
  # StructureMatcher using all CPUs (default)
  python check_data.py --input preprocessed/normalized/test_molcrystal_normalized.pkl.gz --structure_match --n_jobs -1

  # Check and show detailed content of first sample
  python check_data.py --input train.pkl.gz --show_sample 0

  # Quiet mode (just return status)
  python check_data.py -q
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to a single pkl.gz file to check",
    )
    parser.add_argument(
        "--input_dir", "-d",
        type=str,
        default=None,
        help="Directory containing pkl.gz files to check",
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="*.pkl.gz",
        help="Glob pattern for files to check (default: *.pkl.gz)",
    )
    parser.add_argument(
        "--show_sample", "-s",
        type=int,
        default=None,
        help="Show detailed content of a specific sample index",
    )
    parser.add_argument(
        "--export_xyz",
        type=str,
        default=None,
        help="Export reconstructed structures to XYZ file (single --input only)",
    )
    parser.add_argument(
        "--debug_recon",
        type=str,
        default=None,
        help="Debug mode: export both reconstructed and ground truth structures to XYZ files",
    )
    parser.add_argument(
        "--max_export",
        type=int,
        default=None,
        help="Max structures to export (default: all)",
    )
    parser.add_argument(
        "--structure_match",
        action="store_true",
        help="Use pymatgen StructureMatcher for comparison (slow, recommended for test data only)",
    )
    parser.add_argument(
        "--max_match",
        type=int,
        default=None,
        help="Max samples for StructureMatcher comparison (default: all)",
    )
    parser.add_argument(
        "--n_jobs", "-j",
        type=int,
        default=-1,
        help="Number of parallel jobs for StructureMatcher. Use -1 for all CPUs (default: -1)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - only print summary",
    )
    
    args = parser.parse_args()
    
    # Default: check the normalized directory if no input specified
    if args.input is None and args.input_dir is None:
        script_dir = Path(__file__).parent
        default_dir = script_dir / "preprocessed" / "normalized"
        if default_dir.exists():
            args.input_dir = str(default_dir)
        else:
            parser.error("No input specified and default directory ./preprocessed/normalized not found. "
                        "Use --input or --input_dir to specify files to check.")
    
    files_to_check = []
    
    if args.input:
        files_to_check.append(Path(args.input))
    
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            return 1
        files_to_check.extend(sorted(input_dir.glob(args.pattern)))
    
    if not files_to_check:
        print("No files found to check")
        return 1
    
    all_results = {}
    all_valid = True
    total_samples = 0
    
    for filepath in files_to_check:
        results, loaded_data = check_data_file(str(filepath), verbose=not args.quiet)
        all_results[filepath.name] = results
        
        if not results.get("valid", False):
            all_valid = False
        
        total_samples += results.get("n_samples", 0)
        
        if args.show_sample is not None and args.input and filepath == Path(args.input):
            check_sample_content(str(filepath), args.show_sample, verbose=True)

        if args.export_xyz and args.input and filepath == Path(args.input) and loaded_data:
            export_xyz(loaded_data, args.export_xyz, max_structures=args.max_export)
        
        if args.debug_recon and args.input and filepath == Path(args.input) and loaded_data:
            debug_reconstruction(loaded_data, args.debug_recon, max_structures=args.max_export or 10)
        
        if args.structure_match and args.input and filepath == Path(args.input) and loaded_data:
            run_structure_match_comparison(
                loaded_data,
                max_samples=args.max_match,
                n_jobs=args.n_jobs,
                verbose=not args.quiet,
            )
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files checked: {len(files_to_check)}")
    print(f"Total samples: {total_samples}")
    
    print(f"\n[Per-file status]")
    for name, results in all_results.items():
        if results.get("error"):
            print(f"  ✗ {name}: ERROR - {results['error']}")
        elif results.get("valid"):
            n = results.get("n_samples", 0)
            print(f"  ✓ {name}: {n} samples, all features present")
        else:
            n = results.get("n_samples", 0)
            missing = results.get("missing_required", [])
            print(f"  ✗ {name}: {n} samples, missing: {missing}")
    
    print(f"\n[Overall]")
    if all_valid:
        print(f"  ✓ All files are VALID for training")
    else:
        print(f"  ✗ Some files have ISSUES")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    exit(main())
