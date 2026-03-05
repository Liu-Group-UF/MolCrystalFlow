"""Utility functions for XYZ structure matching and RMSD computation."""

import json
import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ase.io import read
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

np.set_printoptions(precision=16, suppress=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_extxyz(file_path):
    """
    Load an extxyz file and extract atomic positions, atom types, and lattice parameters.
    Args:
        file_path (str): Path to the extxyz file.
    Returns:
        structures (list): List of dictionaries with keys 'cart_coords', 'atom_types', 'lattice'.
    """
    atoms_list = read(file_path, index=":", format='extxyz')
    structures = []
    for atoms in atoms_list:
        lattice = atoms.get_cell_lengths_and_angles()  # Lattice parameters (a, b, c, alpha, beta, gamma)
        cell = atoms.get_cell()  # Lattice cell (3x3 matrix)
        cart_coords = atoms.positions  # Cartesian coordinates (Nx3)
        atom_types = atoms.get_chemical_symbols()  # Atom types (list of strings)
        bb_idx = atoms.arrays['bb_idx'] if 'bb_idx' in atoms.arrays else None  # Optional building block indices
        structures.append({
            'cart_coords': cart_coords,
            'atom_types': atom_types,
            'lattice': lattice,
            'cell': cell,
            'bb_idx': bb_idx
        })
    return structures


def save_structures_to_xyz(structures, filename):
    """
    Save structures to XYZ format file with lattice cell information.
    Args:
        structures (list): List of structure dictionaries with 'cart_coords', 'atom_types', 'lattice'
        filename (str): Output filename
    """
    with open(filename, 'w') as f:
        for i, struct in enumerate(structures):
            cart_coords = struct['cart_coords']
            atom_types = struct['atom_types']
            cell = struct['cell']
            bb_idx = struct.get('bb_idx')
            num_atoms = len(atom_types)
            f.write(f"{num_atoms}\n")

            # Handle both cell matrix (3x3) and parameter (6-element) formats
            if cell.shape == (3, 3):
                # Already a 3x3 matrix
                cell_matrix = cell
                # Extract parameters for backward compatibility in comment
                lattice_obj = Lattice(cell)
                a, b, c, alpha, beta, gamma = lattice_obj.parameters
            else:
                # Assume 6-parameter format (a, b, c, alpha, beta, gamma)
                a, b, c, alpha, beta, gamma = cell
                lattice_obj = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
                cell_matrix = lattice_obj.matrix

            # Write comprehensive lattice information
            f.write(f"Structure {i}: Lattice=\"{cell_matrix[0,0]:.6f} {cell_matrix[0,1]:.6f} {cell_matrix[0,2]:.6f} "
                   f"{cell_matrix[1,0]:.6f} {cell_matrix[1,1]:.6f} {cell_matrix[1,2]:.6f} "
                   f"{cell_matrix[2,0]:.6f} {cell_matrix[2,1]:.6f} {cell_matrix[2,2]:.6f}\" "
                   f"Properties=species:S:1:pos:R:3:bb_idx:I:1 a={a:.6f} b={b:.6f} c={c:.6f} "
                   f"alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f}\n")

            # Write atomic coordinates
            for atom_type, coord, per_bb in zip(atom_types, cart_coords, bb_idx):
                f.write(f"{atom_type} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {per_bb}\n")

def convert_to_fractional(structures):
    """
    Convert atomic positions to fractional coordinates for a list of structures.
    Args:
        structures (list): List of dictionaries with keys 'cart_coords', 'atom_types', 'lattice'.
    Returns:
        structures (list): Updated list with fractional coordinates.
    """
    for structure in structures:
        # Handle both lattice matrix (3x3) and parameter (6-element) formats
        if isinstance(structure['lattice'], np.ndarray) and structure['lattice'].shape == (3, 3):
            # Already a 3x3 matrix
            lattice = Lattice(structure['lattice'])
        else:
            # Assume 6-parameter format (a, b, c, alpha, beta, gamma)
            lattice = Lattice.from_parameters(*structure['lattice'])

        frac_coords = lattice.get_fractional_coords(structure['cart_coords'])
        structure['cart_coords'] = frac_coords  # Replace Cartesian with fractional
    return structures


def process_one(gt, pred, matcher):
    """
    Perform structure matching between two structures with error handling.
    Args:
        gt (dict): Ground truth structure with keys 'cart_coords', 'atom_types', 'lattice'.
        pred (dict): Predicted structure with keys 'cart_coords', 'atom_types', 'lattice'.
        matcher (StructureMatcher): Pymatgen StructureMatcher object.
    Returns:
        dict: Dictionary containing RMSD and error information.
    """
    try:
        # Validate input data
        for key in ['cart_coords', 'atom_types', 'lattice']:
            if key not in gt or key not in pred:
                return {
                    'rmsd': None,
                    'error': f"Missing key: {key}",
                    'success': False
                }

        # Validate lattice parameters/matrices
        def validate_lattice(lattice_data):
            if isinstance(lattice_data, np.ndarray) and lattice_data.shape == (3, 3):
                return all(np.isfinite(x) for x in lattice_data.flatten())
            else:
                return all(np.isfinite(x) for x in lattice_data)

        if not validate_lattice(gt['lattice']) or not validate_lattice(pred['lattice']):
            return {
                'rmsd': None,
                'error': "Invalid lattice parameters/matrix (NaN or Inf)",
                'success': False
            }

        # Validate coordinates
        if not np.all(np.isfinite(gt['cart_coords'])) or not np.all(np.isfinite(pred['cart_coords'])):
            return {
                'rmsd': None,
                'error': "Invalid coordinates (NaN or Inf)",
                'success': False
            }

        # Create structures with error handling
        def create_lattice_object(lattice_data):
            if isinstance(lattice_data, np.ndarray) and lattice_data.shape == (3, 3):
                return Lattice(lattice_data)
            else:
                return Lattice.from_parameters(*lattice_data)

        try:
            gt_structure = Structure(
                lattice=create_lattice_object(gt['lattice']),
                species=gt['atom_types'],
                coords=gt['cart_coords'],
                coords_are_cartesian=True
            )
        except Exception as e:
            return {
                'rmsd': None,
                'error': f"Failed to create ground truth structure: {str(e)}",
                'success': False
            }

        try:
            pred_structure = Structure(
                lattice=create_lattice_object(pred['lattice']),
                species=pred['atom_types'],
                coords=pred['cart_coords'],
                coords_are_cartesian=True
            )
        except Exception as e:
            return {
                'rmsd': None,
                'error': f"Failed to create predicted structure: {str(e)}",
                'success': False
            }

        # # Perform matching with error handling
        try:
            rmsd = matcher.get_rms_dist(gt_structure, pred_structure)
            rmsd = rmsd if rmsd is None else rmsd[0]
            return {
                'rmsd': rmsd,
                'error': None,
                'success': True
            }
        except Exception as e:
            return {
                'rmsd': None,
                'error': f"Matching failed: {str(e)}",
                'success': False
            }

    except Exception as e:
        return {
            'rmsd': None,
            'error': f"Unexpected error: {str(e)}",
            'success': False
        }

def match_structures(gt_file, pred_file, num_cpus=8,
                     stol=0.5, ltol=0.3, angle_tol=10,
                     output_json='rmsd_summary.json'):
    """
    Perform structure matching for two extxyz files with comprehensive error handling.
    Args:
        gt_file (str): Path to ground truth extxyz file.
        pred_file (str): Path to predicted extxyz file.
        num_cpus (int): Number of CPUs for parallel processing.
        stol (float): Site tolerance for structure matching.
        ltol (float): Lattice tolerance for structure matching.
        angle_tol (float): Angle tolerance for structure matching.
    Returns:
        tuple: (DataFrame with results, match rate, error summary)
    """
    try:
        # Load structures from extxyz files with error handling
        try:
            gt_structures = load_extxyz(gt_file)
            pred_structures = load_extxyz(pred_file)
        except Exception as e:
            raise ValueError(f"Failed to load structures: {str(e)}")

        # Ensure one-to-one mapping
        if len(gt_structures) != len(pred_structures):
            raise ValueError(f"Number of structures mismatch: GT={len(gt_structures)}, Pred={len(pred_structures)}")

        # Perform structure matching
        matcher = StructureMatcher(stol=stol, ltol=ltol, angle_tol=angle_tol, scale=False)
        results = p_map(
            partial(process_one, matcher=matcher),
            gt_structures,
            pred_structures,
            num_cpus=num_cpus
        )

        # Process results and collect errors
        rmsd_values = []
        error_log = []
        for idx, result in enumerate(results):
            if result['success']:
                rmsd_values.append(result['rmsd'])
            else:
                rmsd_values.append(None)
                error_log.append({
                    'structure_idx': idx,
                    'error': result['error']
                })

        # Create results DataFrame
        rmsd_df = pd.DataFrame({
            'RMSD': rmsd_values,
        })

        # Compute statistics on valid results only
        valid_rmsd = [r for r in rmsd_values if r is not None]
        total_structures = len(rmsd_values)
        valid_structures = len(valid_rmsd)

        if valid_structures > 0:
            mean_rmsd = np.mean(valid_rmsd)
            std_rmsd = np.std(valid_rmsd) if len(valid_rmsd) > 1 else 0
            mr = (valid_structures / total_structures) * 100
        else:
            mean_rmsd = None
            std_rmsd = None
            mr = 0

        # Prepare detailed summary
        summary = {
            'match_rate_percent': mr,
            'mean_rmsd': mean_rmsd,
            'std_rmsd': std_rmsd,
            'total_structures': total_structures,
            'valid_matches': valid_structures,
            'failed_matches': total_structures - valid_structures,
            'error_summary': {
                'total_errors': len(error_log),
                'error_details': error_log
            }
        }

        # Save summary to JSON
        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"Summary saved to {output_json}")
        print(f"\nResults Summary:")
        print(f"Total structures: {total_structures}")
        print(f"Valid matches: {valid_structures}")
        print(f"Failed matches: {total_structures - valid_structures}")
        print(f"Match rate: {mr:.2f}%")
        if mean_rmsd is not None:
            print(f"Mean RMSD: {mean_rmsd:.4f}")
            print(f"Std RMSD: {std_rmsd:.4f}")
        if error_log:
            print(f"\nFound {len(error_log)} errors. Check {output_json} for details.")

        return rmsd_df, mr, summary

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return None, 0, {'error': str(e)}


def match_structures_multi_samples(gt_file, pred_file, num_samples, num_cpus=8,
                                   stol=0.5, ltol=0.3, angle_tol=10,
                                   output_csv='rmsd_results.csv',
                                   output_json='summary.json',
                                   verbose=True, show_sample_progress=True):
    """
    Perform structure matching similar to evaluate.py approach with multiple samples.
    For each GT structure, compare against multiple predicted samples and compute statistics.

    Args:
        gt_file (str): Path to ground truth extxyz file (e.g., 750 structures).
        pred_file (str): Path to predicted extxyz file (e.g., 3750 structures = 750 * 5 samples).
        num_samples (int): Number of predicted samples per ground truth structure.
        num_cpus (int): Number of CPUs for parallel processing.
        stol (float): Site tolerance for structure matching.
        ltol (float): Lattice tolerance for structure matching.
        angle_tol (float): Angle tolerance for structure matching.
        output_csv (str): Path to save CSV with per-sample and minimum RMSD values.
        output_json (str): Path to save JSON summary.
        verbose (bool): Whether to print detailed progress and results.
        show_sample_progress (bool): Whether to show per-sample processing progress and match rates.

    Returns:
        tuple: (DataFrame with results, overall match rate, summary dict)
    """
    try:
        # Load structures
        if verbose:
            print("Loading structures...")
        gt_structures = load_extxyz(gt_file)
        pred_structures = load_extxyz(pred_file)

        num_gt = len(gt_structures)
        expected_pred = num_gt * num_samples

        if len(pred_structures) != expected_pred:
            raise ValueError(f"Expected {expected_pred} predicted structures "
                           f"({num_gt} GT × {num_samples} samples), got {len(pred_structures)}")

        if verbose:
            print(f"Loaded {num_gt} GT structures and {len(pred_structures)} predicted structures")
            print(f"Processing {num_samples} samples per GT structure...")

        # Group predicted structures by samples
        pred_samples = []
        for sample_idx in range(num_samples):
            sample_structures = []
            for gt_idx in range(num_gt):
                pred_idx = gt_idx  + sample_idx * num_gt
                sample_structures.append(pred_structures[pred_idx])
            pred_samples.append(sample_structures)

        # Create matcher
        matcher = StructureMatcher(stol=stol, ltol=ltol, angle_tol=angle_tol, scale=False)

        # Compute RMSD for each sample
        if verbose:
            print("Computing RMSD for each sample...")

        # Collect all RMSD data first to avoid DataFrame fragmentation
        all_rmsd_data = {}

        for sample_idx in range(num_samples):
            if show_sample_progress:
                print(f"Processing sample {sample_idx + 1}/{num_samples}...")

            # Compute RMSD for this sample
            if show_sample_progress:
                # Use p_map with progress bar
                results = p_map(
                    partial(process_one, matcher=matcher),
                    gt_structures,
                    pred_samples[sample_idx],
                    num_cpus=num_cpus
                )
            else:
                # Use regular multiprocessing without progress bar
                from multiprocessing import Pool
                with Pool(num_cpus) as pool:
                    results = pool.starmap(
                        process_one,
                        [(gt, pred, matcher) for gt, pred in zip(gt_structures, pred_samples[sample_idx])]
                    )

            # Extract RMSD values, handling errors
            rmsd_values = []
            for result in results:
                if result['success'] and result['rmsd'] is not None:
                    rmsd_values.append(result['rmsd'])
                else:
                    rmsd_values.append(None)

            all_rmsd_data[f'sample_{sample_idx}'] = rmsd_values

        # Create DataFrame from all collected data at once (avoids fragmentation)
        rmsd_df = pd.DataFrame(all_rmsd_data)

        # Compute minimum RMSD across samples (like evaluate.py)
        rmsd_df['min'] = rmsd_df.min(axis=1)

        # Compute match rates for each sample and overall
        sample_match_rates = {}
        for sample_idx in range(num_samples):
            col_name = f'sample_{sample_idx}'
            valid_count = rmsd_df[col_name].notna().sum()
            match_rate = (valid_count / num_gt) * 100
            sample_match_rates[col_name] = match_rate
            if show_sample_progress:
                print(f"Sample {sample_idx} match rate: {match_rate:.2f}%")

        # Overall match rate based on minimum RMSD
        overall_valid = rmsd_df['min'].notna().sum()
        overall_match_rate = (overall_valid / num_gt) * 100
        if verbose:
            print(f"Overall match rate (best per structure): {overall_match_rate:.2f}%")

        # Compute summary statistics
        min_rmsd_valid = rmsd_df['min'].dropna()
        if len(min_rmsd_valid) > 0:
            mean_min_rmsd = min_rmsd_valid.mean()
            std_min_rmsd = min_rmsd_valid.std()
        else:
            mean_min_rmsd = None
            std_min_rmsd = None

        # Prepare summary
        summary = {
            'num_gt_structures': num_gt,
            'num_samples': num_samples,
            'overall_match_rate_percent': overall_match_rate,
            'mean_min_rmsd': mean_min_rmsd,
            'std_min_rmsd': std_min_rmsd,
            'valid_matches': int(overall_valid),
            'failed_matches': num_gt - int(overall_valid),
            'sample_match_rates': sample_match_rates
        }

        # Save results
        rmsd_df.to_csv(output_csv, index=False)
        if verbose:
            print(f"RMSD results saved to {output_csv}")

        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=4)
        if verbose:
            print(f"Summary saved to {output_json}")

        # Save best matched structures to XYZ files
        best_pred_structures = []
        best_gt_structures = []

        for gt_idx in range(num_gt):
            # Find the sample with minimum RMSD for this GT structure
            sample_rmsd_values = []
            for sample_idx in range(num_samples):
                rmsd_val = rmsd_df.iloc[gt_idx][f'sample_{sample_idx}']
                sample_rmsd_values.append(rmsd_val)

            # Find the best sample (lowest RMSD) that's not None/NaN
            valid_indices = [i for i, rmsd in enumerate(sample_rmsd_values) if pd.notna(rmsd)]

            if valid_indices:
                best_sample_idx = min(valid_indices, key=lambda i: sample_rmsd_values[i])

                # Get the corresponding predicted structure
                best_pred_structure = pred_samples[best_sample_idx][gt_idx]
                best_pred_structures.append(best_pred_structure)

                # Get the corresponding ground truth structure
                best_gt_structures.append(gt_structures[gt_idx])

        # Save to XYZ files
        if best_pred_structures:
            input_dir = os.path.dirname(gt_file)
            gt_base = os.path.splitext(os.path.basename(gt_file))[0]
            pred_base = os.path.splitext(os.path.basename(pred_file))[0]

            best_pred_xyz = os.path.join(input_dir, f"{pred_base}_best_predictions.xyz")
            best_gt_xyz = os.path.join(input_dir, f"{gt_base}_matched_ground_truth.xyz")

            save_structures_to_xyz(best_pred_structures, best_pred_xyz)
            save_structures_to_xyz(best_gt_structures, best_gt_xyz)

            if verbose:
                print(f"Best predicted structures saved to: {best_pred_xyz}")
                print(f"Corresponding ground truth structures saved to: {best_gt_xyz}")
                print(f"Saved {len(best_pred_structures)} structure pairs with valid matches")

        # Print final summary
        print(f"\nFinal Summary:")
        print(f"Overall match rate: {overall_match_rate:.2f}%")
        if mean_min_rmsd is not None:
            print(f"Mean minimum RMSD: {mean_min_rmsd:.4f}")
            print(f"Std minimum RMSD: {std_min_rmsd:.4f}")

        return rmsd_df, overall_match_rate, summary

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return None, 0, {'error': str(e)}
