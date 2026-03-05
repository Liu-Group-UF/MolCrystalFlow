#!/usr/bin/env python
"""
Check the final generated pkl.gz files for completeness and correctness.

This script validates that all required features are present in the
training data files and reports statistics about the data.

Checks include:
- Presence of required / optional keys
- Axis flip consistency (only [0,0,0] or [1,0,0] patterns)
- Coordinate reconstruction: reassemble Cartesian coordinates from
  (local_coords, rotmats_1, trans_1, lattice_1, bb_num_vec) and compare to gt_coords
- Optional XYZ export for visual comparison against the original standardized XYZ

Usage:
    python check_data.py --input_dir ./final
    python check_data.py --input train_molcrystal.pkl.gz
    python check_data.py --input test_molcrystal.pkl.gz --export_xyz reconstructed_test.xyz
"""
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from common import (
    ATOM_TYPE_TO_IDX,
    ATOMIC_NUM_TO_SYMBOL,
    IDX_TO_ATOM_TYPE,
    assemble_coords,
    load_pkl_gz,
    to_numpy,
)


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
    
    # Check consistency within same flip pattern (should be identical up to permutation)
    # This is complex because atom ordering might differ, so we skip this for now
    
    # Check consistency between different flip patterns
    # For 2 patterns differing by a 2-axis flip, coords should differ by sign in 1 axis
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
                # For each axis, check if flipping sign makes coords match
                axis_sign_diffs = []
                for axis in range(3):
                    # Check if coords match when we flip sign of this axis
                    c2_flipped = c2.copy()
                    c2_flipped[:, axis] = -c2_flipped[:, axis]
                    
                    # Check if this makes them approximately equal
                    if np.allclose(c1, c2_flipped, atol=tol):
                        axis_sign_diffs.append(axis)
                
                # Also check direct match
                if np.allclose(c1, c2, atol=tol):
                    # Same coordinates despite different flip patterns - this is OK
                    pass
                elif len(axis_sign_diffs) == 1:
                    # Exactly one axis sign flip - expected behavior
                    pass
                elif len(axis_sign_diffs) == 0:
                    # Check if 2-axis flip works (flip 2 axes simultaneously)
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


def check_atom_ordering(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check that atoms are grouped by building block in contiguous blocks.

    The bb_num_vec should describe contiguous groups: atoms 0..bb_num_vec[0]-1
    belong to bb 0, etc. This verifies that the atom count totals match and that
    the atom_types count equals sum(bb_num_vec).

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

    # Verify that local_coords and gt_coords have the same atom count
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

    Each structure is written in extended XYZ format with a Lattice= comment,
    matching the format of the standardized input XYZ files.

    Args:
        data: List of sample dictionaries.
        output_path: Path to write the XYZ file.
        max_structures: Maximum number of structures to export (None = all).
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

            # Build per-atom bb index
            bb_indices = []
            for bb_i, count in enumerate(bb_num_vec):
                bb_indices.extend([bb_i] * int(count))

            for atom_i in range(n_atoms):
                anum = IDX_TO_ATOM_TYPE.get(int(atom_types_idx[atom_i]), 0)
                sym = ATOMIC_NUM_TO_SYMBOL.get(anum, "X")
                x, y, z = recon[atom_i]
                f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f} {bb_indices[atom_i]}\n")

    print(f"Exported {n} structures to {output_path}")


def check_single_sample(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Check a single data sample for required keys and valid values.
    
    Returns:
        Dictionary with check results.
    """
    results = {
        "idx": idx,
        "missing_required": [],
        "missing_optional": [],
        "invalid_values": [],
        "shapes": {},
    }
    
    # Check required keys
    for key in REQUIRED_KEYS:
        if key not in sample:
            results["missing_required"].append(key)
        else:
            value = sample[key]
            # Convert tensors to numpy
            if hasattr(value, 'numpy') or hasattr(value, 'shape'):
                value = to_numpy(value) if not isinstance(value, np.ndarray) else value
                if hasattr(value, 'shape'):
                    results["shapes"][key] = value.shape
                    # Check for NaN/Inf
                    try:
                        if np.any(np.isnan(value)):
                            results["invalid_values"].append(f"{key}: contains NaN")
                        if np.any(np.isinf(value)):
                            results["invalid_values"].append(f"{key}: contains Inf")
                    except (TypeError, ValueError):
                        pass  # Skip NaN/Inf check for non-float types
            elif isinstance(value, (list, tuple)):
                results["shapes"][key] = len(value)
    
    # Check optional keys
    for key in OPTIONAL_KEYS:
        if key not in sample:
            results["missing_optional"].append(key)
        else:
            value = sample[key]
            # Convert tensors to numpy
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
                        pass  # Skip NaN/Inf check for non-float types
    
    return results


def check_data_file(filepath: str, verbose: bool = True):
    """
    Check a pkl.gz data file for completeness and correctness.
    
    Args:
        filepath: Path to the pkl.gz file.
        verbose: Whether to print detailed information.
    
    Returns:
        Tuple of (results_dict, loaded_data_list_or_None).
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return {"error": f"File not found: {filepath}", "valid": False}, None
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Checking: {filepath.name}")
        print(f"{'='*60}")
    
    # Load data
    try:
        data = load_pkl_gz(str(filepath))
    except Exception as e:
        return {"error": f"Failed to load file: {e}", "valid": False}, None
    
    n_samples = len(data)
    if verbose:
        print(f"Number of samples: {n_samples}")
    
    if n_samples == 0:
        return {"error": "File contains no samples", "valid": False, "n_samples": 0}, None
    
    # Check all samples
    all_missing_required = set()
    all_missing_optional = set()
    samples_with_issues = []
    all_invalid_values = []
    
    # Track feature statistics
    feature_stats = {key: {"present": 0, "shapes": set()} for key in FEATURE_KEYS}
    
    # Track general statistics
    num_bbs_list = []
    num_atoms_list = []
    
    for i, sample in enumerate(data):
        result = check_single_sample(sample, i)
        
        if result["missing_required"]:
            all_missing_required.update(result["missing_required"])
            samples_with_issues.append(i)
        
        all_missing_optional.update(result["missing_optional"])
        all_invalid_values.extend(result["invalid_values"])
        
        # Track feature presence
        for key in FEATURE_KEYS:
            if key in sample and sample[key] is not None:
                feature_stats[key]["present"] += 1
                if isinstance(sample[key], np.ndarray):
                    feature_stats[key]["shapes"].add(sample[key].shape)
        
        # Track building block and atom counts
        if "bb_num_vec" in sample:
            bb_vec = sample["bb_num_vec"]
            if isinstance(bb_vec, np.ndarray):
                num_bbs_list.append(len(bb_vec))
                num_atoms_list.append(np.sum(bb_vec))
    
    # Check axis flip consistency for a subset of samples
    axis_flip_results = {
        "samples_checked": 0,
        "samples_with_axis_flips": 0,
        "samples_passed": 0,
        "samples_failed": 0,
        "issues": [],
    }

    # Check coordinate reconstruction for a subset of samples
    recon_results = {
        "samples_checked": 0,
        "samples_passed": 0,
        "samples_failed": 0,
        "max_error": 0.0,
        "mean_error": 0.0,
        "issues": [],
    }

    # Check atom ordering consistency for a subset of samples
    ordering_results = {
        "samples_checked": 0,
        "samples_passed": 0,
        "samples_failed": 0,
        "issues": [],
    }
    
    # Check up to 100 samples for axis flip consistency & reconstruction
    check_limit = min(100, n_samples)
    for i in range(check_limit):
        sample = data[i]

        # Axis flip check
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

        # Reconstruction check
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

        # Atom ordering check
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
    
    # Summary
    results = {
        "valid": len(all_missing_required) == 0 and len(samples_with_issues) == 0,
        "n_samples": n_samples,
        "missing_required": list(all_missing_required),
        "missing_optional": list(all_missing_optional),
        "samples_with_issues": len(samples_with_issues),
        "invalid_values": all_invalid_values[:10],  # Limit to first 10
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
    
    # Add axis flip results
    results["axis_flip_check"] = axis_flip_results

    # Add reconstruction results
    results["reconstruction_check"] = recon_results

    # Add atom ordering results
    results["atom_ordering_check"] = ordering_results
    
    # Print results
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
        
        # Print coordinate reconstruction results
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

        # Print axis flip consistency results
        print(f"\n[Axis Flip Consistency Check]")
        af = axis_flip_results
        print(f"  Samples checked: {af['samples_checked']}")
        print(f"  Samples with axis_flips: {af['samples_with_axis_flips']}")
        if af['samples_with_axis_flips'] > 0:
            if af['samples_failed'] == 0:
                print(f"  ✓ All {af['samples_passed']} samples passed consistency check")
                print(f"    (Building blocks with different flip states differ by sign in 1-2 axes)")
            else:
                print(f"  ✗ {af['samples_failed']} samples failed consistency check")
                print(f"  ✓ {af['samples_passed']} samples passed")
                for issue in af['issues'][:5]:
                    print(f"    - {issue}")
                if len(af['issues']) > 5:
                    print(f"    ... and {len(af['issues']) - 5} more issues")
        else:
            print(f"  No axis_flips data found in samples")

        # Print atom ordering results
        print(f"\n[Atom Ordering Check]")
        ao = ordering_results
        print(f"  Samples checked: {ao['samples_checked']}")
        if ao['samples_checked'] > 0:
            if ao['samples_failed'] == 0:
                print(f"  ✓ All {ao['samples_passed']} samples have consistent atom ordering")
                print(f"    (sum(bb_num_vec) == len(atom_types) == len(local_coords) == len(gt_coords))")
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
        description="Check pkl.gz data files for completeness and correctness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all files in a directory
  python check_data.py --input_dir ./final

  # Check a single file
  python check_data.py --input train_molcrystal.pkl.gz

  # Check and export reconstructed XYZ for visual comparison
  python check_data.py --input test_molcrystal.pkl.gz --export_xyz reconstructed_test.xyz

  # Check and show detailed content of first sample
  python check_data.py --input train.pkl.gz --show_sample 0

  # Quiet mode (just return status)
  python check_data.py --input_dir ./final -q
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
        "--max_export",
        type=int,
        default=None,
        help="Max structures to export (default: all)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - only print summary",
    )
    
    args = parser.parse_args()
    
    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input_dir must be specified")
    
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
    
    # Check each file
    all_results = {}
    all_valid = True
    total_samples = 0
    
    for filepath in files_to_check:
        results, loaded_data = check_data_file(str(filepath), verbose=not args.quiet)
        all_results[filepath.name] = results
        
        if not results.get("valid", False):
            all_valid = False
        
        total_samples += results.get("n_samples", 0)
        
        # Show sample content if requested
        if args.show_sample is not None and args.input and filepath == Path(args.input):
            check_sample_content(str(filepath), args.show_sample, verbose=True)

        # Export reconstructed XYZ if requested
        if args.export_xyz and args.input and filepath == Path(args.input) and loaded_data:
            export_xyz(loaded_data, args.export_xyz, max_structures=args.max_export)
    
    # Print summary
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
