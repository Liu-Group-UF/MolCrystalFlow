#!/usr/bin/env python3
"""
Collect UMA-OMC Cell Optimization Results.

This script:
1. Combines all available cell-opt result files (handling missing jobs)
2. Parses optimization logs for convergence information
3. Computes lattice energy per molecule and density
4. Tracks full index chain: original → filtered → cell-opt
5. Saves detailed CSV with all structure information
6. Extracts top N lowest lattice energy structures for DFT evaluation
7. Organizes all outputs in a folder named after the molecule's chemical formula

Index Chain:
    data/xyz/pred_combined.xyz (original generated)
        ↓ filtering by inter-BB distance
    data/xyz/filtered_by_inter_bb.xyz (filtered_idx → orig_idx via data/npy/filtered_by_inter_bb_indices.npy)
        ↓ cell optimization
    cell-opt-results/*.extxyz (cell_opt_idx → filtered_idx)
        ↓ collection
    <formula>/results/cell_opt_all_results.csv (full index tracking)

Usage:
    python collect_relaxation_results.py
    python collect_relaxation_results.py --top_n 10
    python collect_relaxation_results.py --formula C9H9N3O5 --top_n 10
"""

import argparse
import glob
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import atomic_masses
from ase.io import read, write

# Optional: plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# Plotting Functions
# ==============================================================================


def plot_energy_landscape(
    df: pd.DataFrame,
    formula: str,
    output_path: str,
    energy_cutoff_kj: float = 20.0,
    highlight_top_n: int = 10,
):
    """
    Plot energy landscape: lattice energy vs density.
    
    Only shows data points within `energy_cutoff_kj` kJ/mol of the minimum.
    Highlights top N lowest energy structures.
    
    Args:
        df: DataFrame with E_per_mol_eV and density_g_cm3 columns.
        formula: Chemical formula for title.
        output_path: Path to save the plot.
        energy_cutoff_kj: Energy cutoff in kJ/mol from minimum.
        highlight_top_n: Number of top structures to highlight.
    """
    if not HAS_MATPLOTLIB:
        print("  Warning: matplotlib not available, skipping energy landscape plot")
        return
    
    # Filter valid data
    df_valid = df[df["E_per_mol_eV"].notna() & df["density_g_cm3"].notna()].copy()
    if len(df_valid) == 0:
        print("  Warning: No valid data for energy landscape plot")
        return
    
    # Convert to kJ/mol relative to minimum
    eV_to_kJ = 96.485  # 1 eV = 96.485 kJ/mol
    E_min = df_valid["E_per_mol_eV"].min()
    df_valid["E_rel_kJ"] = (df_valid["E_per_mol_eV"] - E_min) * eV_to_kJ
    
    # Filter by energy cutoff
    df_filtered = df_valid[df_valid["E_rel_kJ"] <= energy_cutoff_kj].copy()
    
    if len(df_filtered) == 0:
        print(f"  Warning: No structures within {energy_cutoff_kj} kJ/mol cutoff")
        return
    
    # Get top N
    df_top = df_filtered.nsmallest(highlight_top_n, "E_rel_kJ")
    df_rest = df_filtered[~df_filtered.index.isin(df_top.index)]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot all points within cutoff (gray)
    ax.scatter(
        df_rest["density_g_cm3"],
        df_rest["E_rel_kJ"],
        c="gray",
        alpha=0.5,
        s=30,
        label=f"All structures (n={len(df_filtered)})",
        edgecolors="none",
    )
    
    # Highlight top N (colored by rank)
    scatter = ax.scatter(
        df_top["density_g_cm3"],
        df_top["E_rel_kJ"],
        c=range(1, len(df_top) + 1),
        cmap="coolwarm_r",
        s=100,
        edgecolors="black",
        linewidths=1,
        label=f"Top {len(df_top)} structures",
        zorder=5,
    )
    
    # Add rank labels for top structures
    for i, (_, row) in enumerate(df_top.iterrows(), 1):
        ax.annotate(
            str(i),
            (row["density_g_cm3"], row["E_rel_kJ"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Rank", fontsize=12)
    
    # Labels and title
    ax.set_xlabel("Density (g/cm³)", fontsize=12)
    ax.set_ylabel("Relative Lattice Energy (kJ/mol)", fontsize=12)
    ax.set_title(
        f"UMA-OMC Energy Landscape: {formula}\n"
        f"(Showing {len(df_filtered)} structures within {energy_cutoff_kj} kJ/mol of minimum)",
        fontsize=14,
    )
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=10)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved energy landscape plot: {output_path}")
    print(f"    - Showing {len(df_filtered)} structures within {energy_cutoff_kj} kJ/mol cutoff")
    print(f"    - Highlighted top {len(df_top)} lowest energy structures")


# ==============================================================================
# Helper Functions
# ==============================================================================


def parse_last_bfgs_line(log_path: str):
    """Extract last BFGS line (steps, potential energy, max force) from ASE log."""
    pattern = re.compile(r"BFGS:\s+(\d+)\s+\S+\s+([-\d.]+)\s+([-\d.]+)")
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        matches = [pattern.search(line) for line in lines if "BFGS:" in line]
        matches = [m for m in matches if m]
        if not matches:
            return None, None, None
        last = matches[-1]
        return int(last.group(1)), float(last.group(2)), float(last.group(3))
    except Exception as e:
        print(f"Warning: Could not parse log {log_path}: {e}")
        return None, None, None


def get_molecular_formula_from_symbols(symbols):
    """Convert list of chemical symbols to molecular formula string."""
    counter = Counter(symbols)
    formula = ""
    element_order = ['C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    remaining_elements = set(counter.keys()) - set(element_order)
    ordered_elements = [e for e in element_order if e in counter] + sorted(remaining_elements)
    
    for element in ordered_elements:
        count = counter[element]
        if count == 1:
            formula += element
        else:
            formula += f"{element}{count}"
    
    return formula


def get_building_block_formulas(atoms, bb_key="bb_idx"):
    """Extract molecular formula for each building block in the structure."""
    bb_idx = atoms.arrays.get(bb_key, atoms.arrays.get("bb_indices", np.zeros(len(atoms), dtype=int)))
    unique_bb = np.unique(bb_idx)
    
    formulas = []
    for bb in unique_bb:
        bb_atoms = atoms[bb_idx == bb]
        symbols = bb_atoms.get_chemical_symbols()
        formula = get_molecular_formula_from_symbols(symbols)
        formulas.append(formula)
    
    return formulas


def compute_density(atoms, bb_key="bb_idx"):
    """Compute density (g/cm³) from ASE Atoms object."""
    cell_vol = atoms.get_volume()  # Å^3
    if cell_vol <= 0:
        return np.nan

    bb_idx = atoms.arrays.get(bb_key, atoms.arrays.get("bb_indices", np.zeros(len(atoms), dtype=int)))
    unique_bb = np.unique(bb_idx)
    n_bb = len(unique_bb)

    # Compute mean molecular weight over all building blocks
    mw_list = []
    for bb in unique_bb:
        bb_atoms = atoms[bb_idx == bb]
        mw = np.sum([atomic_masses[a.number] for a in bb_atoms])
        mw_list.append(mw)
    mean_mw = np.mean(mw_list)

    # Convert Å^3 to cm^3 (1 Å^3 = 1e-24 cm^3)
    vol_per_mol = cell_vol / n_bb
    density = 1.66054 * mean_mw / vol_per_mol
    return density


def get_lattice_parameters(atoms):
    """Extract lattice parameters (a, b, c, alpha, beta, gamma) from atoms."""
    cell = atoms.get_cell()
    lengths = cell.lengths()
    angles = cell.angles()
    return {
        "a": lengths[0],
        "b": lengths[1],
        "c": lengths[2],
        "alpha": angles[0],
        "beta": angles[1],
        "gamma": angles[2],
        "volume": atoms.get_volume(),
    }


def extract_start_end(filename: str):
    """Extract start and end indices from filename."""
    base = os.path.basename(filename)
    m = re.search(r"cell_opt_(\d+)_(\d+)\.(ext)?xyz$", base)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (0, 0)


def load_filter_indices(filter_indices_path: str = "filtered_by_inter_bb_indices.npy"):
    """
    Load the indices mapping from filtered structures back to original.
    
    Returns:
        np.ndarray where filtered_to_orig[filtered_idx] = original_idx
    """
    if os.path.exists(filter_indices_path):
        return np.load(filter_indices_path)
    else:
        print(f"Warning: Filter indices file not found: {filter_indices_path}")
        return None


# ==============================================================================
# Main Functions
# ==============================================================================


def combine_cell_opt_results(
    results_dir: str = "cell-opt-results",
    pattern: str = "pred_omc25_*step_cell_opt_*.extxyz",
    output_xyz: str = None,
    verbose: bool = True,
):
    """
    Combine all cell-opt result files into one list.
    Handles missing jobs by tracking original indices.
    
    Returns:
        all_atoms: list of ASE Atoms objects
        index_mapping: list of dicts with index information
    """
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        raise ValueError(f"No files found for pattern: {results_dir}/{pattern}")
    
    # Sort by start index
    files.sort(key=lambda p: extract_start_end(p))
    
    all_atoms = []
    index_mapping = []
    
    for f in files:
        start, end = extract_start_end(f)
        try:
            atoms_list = read(f, index=":")
            for local_idx, atoms in enumerate(atoms_list):
                all_atoms.append(atoms)
                # Index in the filtered file (input to cell-opt)
                filtered_idx = start + local_idx
                index_mapping.append({
                    "cell_opt_idx": len(all_atoms) - 1,
                    "filtered_idx": filtered_idx,
                    "chunk_file": os.path.basename(f),
                    "chunk_start": start,
                    "chunk_end": end,
                    "local_idx_in_chunk": local_idx,
                })
            if verbose:
                print(f"  Loaded {len(atoms_list)} structures from {os.path.basename(f)} (filtered indices {start}-{start+len(atoms_list)-1})")
        except Exception as e:
            print(f"  Warning: Could not read {f}: {e}")
    
    if all_atoms and output_xyz:
        write(output_xyz, all_atoms, format="extxyz")
        if verbose:
            print(f"\nCombined {len(all_atoms)} structures into {output_xyz}")
    
    return all_atoms, index_mapping


def collect_relaxation_data(
    all_atoms,
    index_mapping,
    filtered_to_orig: np.ndarray = None,
    log_dir: str = "cell-opt-logs",
    bb_key: str = "bb_idx",
    verbose: bool = True,
):
    """
    Collect detailed relaxation data for all structures with full index tracking.
    
    Args:
        all_atoms: list of ASE Atoms from cell-opt results
        index_mapping: list of dicts with cell_opt_idx, filtered_idx, etc.
        filtered_to_orig: array mapping filtered_idx → original_idx in pred_combined.xyz
        log_dir: directory containing optimization log files
        bb_key: key for building block indices in atoms.arrays
    
    Returns:
        pandas DataFrame with all structure information and full index chain
    """
    results = []
    
    for i, atoms in enumerate(all_atoms):
        mapping = index_mapping[i]
        filtered_idx = mapping["filtered_idx"]
        
        # Map back to original index in pred_combined.xyz
        if filtered_to_orig is not None and filtered_idx < len(filtered_to_orig):
            orig_idx = int(filtered_to_orig[filtered_idx])
        else:
            orig_idx = None
        
        # Try to find log file (uses filtered_idx for naming)
        log_name = f"continue_opt_{filtered_idx:05d}.log"
        log_path = os.path.join(log_dir, log_name)
        
        steps, energy, max_force = None, None, None
        if os.path.exists(log_path):
            steps, energy, max_force = parse_last_bfgs_line(log_path)
        
        # If no log found, try to get energy from atoms.info
        if energy is None and "energy" in atoms.info:
            energy = atoms.info["energy"]
        
        # Compute properties
        formulas = get_building_block_formulas(atoms, bb_key=bb_key)
        bb_idx_arr = atoms.arrays.get(bb_key, atoms.arrays.get("bb_indices", np.zeros(len(atoms), dtype=int)))
        n_bb = len(np.unique(bb_idx_arr))
        density = compute_density(atoms, bb_key=bb_key)
        lattice_params = get_lattice_parameters(atoms)
        
        # Compute energy per molecule
        E_per_mol = energy / n_bb if energy is not None and n_bb > 0 else None
        
        # Dominant formula
        unique_formulas = list(dict.fromkeys(formulas))
        dominant_formula = unique_formulas[0] if len(unique_formulas) == 1 else "mixed"
        
        results.append({
            # Index chain (most important for traceability)
            "cell_opt_idx": mapping["cell_opt_idx"],
            "filtered_idx": filtered_idx,
            "orig_idx": orig_idx,
            # Source info
            "chunk_file": mapping["chunk_file"],
            # Structure info
            "n_atoms": len(atoms),
            "n_bb": n_bb,
            "formulas": str(formulas),
            "dominant_formula": dominant_formula,
            # Optimization results
            "opt_steps": steps,
            "total_energy_eV": energy,
            "E_per_mol_eV": E_per_mol,
            "max_force_eV_A": max_force,
            # Physical properties
            "density_g_cm3": density,
            **lattice_params,
        })
    
    df = pd.DataFrame(results)
    
    if verbose:
        print(f"\nCollected data for {len(df)} structures")
        print(f"  Formulas found: {df['dominant_formula'].value_counts().to_dict()}")
        if df['E_per_mol_eV'].notna().any():
            print(f"  Energy range: {df['E_per_mol_eV'].min():.4f} to {df['E_per_mol_eV'].max():.4f} eV/mol")
    
    return df


def extract_top_n_structures(
    df: pd.DataFrame,
    all_atoms: list,
    top_n: int = 10,
    output_dir: str = "results",
    verbose: bool = True,
):
    """
    Extract top N lowest lattice energy structures for DFT evaluation.
    
    Saves:
        - top{N}_for_dft.xyz: structures with full metadata in atoms.info
        - top{N}_for_dft.csv: detailed metadata table
    """
    # Filter out structures without energy
    df_valid = df[df["E_per_mol_eV"].notna()].copy()
    
    # Sort by energy per molecule (lowest first)
    df_sorted = df_valid.sort_values("E_per_mol_eV", ascending=True)
    
    # Take top N
    df_top = df_sorted.head(top_n).copy()
    df_top["rank"] = range(1, len(df_top) + 1)
    
    # Extract structures with full metadata
    top_structures = []
    for rank, (_, row) in enumerate(df_top.iterrows(), 1):
        idx = int(row["cell_opt_idx"])
        atoms = all_atoms[idx].copy()
        
        # Add full index chain and metadata to atoms.info
        atoms.info["rank"] = rank
        atoms.info["cell_opt_idx"] = int(row["cell_opt_idx"])
        atoms.info["filtered_idx"] = int(row["filtered_idx"])
        atoms.info["orig_idx"] = int(row["orig_idx"]) if pd.notna(row["orig_idx"]) else None
        atoms.info["E_per_mol_eV"] = float(row["E_per_mol_eV"])
        atoms.info["density_g_cm3"] = float(row["density_g_cm3"])
        atoms.info["dominant_formula"] = row["dominant_formula"]
        atoms.info["opt_steps"] = int(row["opt_steps"]) if pd.notna(row["opt_steps"]) else None
        
        top_structures.append(atoms)
    
    # Save outputs
    output_xyz = os.path.join(output_dir, f"top{top_n}_for_dft.xyz")
    output_csv = os.path.join(output_dir, f"top{top_n}_for_dft.csv")
    
    write(output_xyz, top_structures, format="extxyz")
    df_top.to_csv(output_csv, index=False)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Top {top_n} Lowest Lattice Energy Structures")
        print(f"{'='*70}")
        print(f"{'Rank':<5} {'orig_idx':<10} {'filtered_idx':<12} {'E (eV/mol)':<14} {'ρ (g/cm³)':<12} {'Formula'}")
        print("-" * 70)
        for _, row in df_top.iterrows():
            orig_str = str(int(row['orig_idx'])) if pd.notna(row['orig_idx']) else "N/A"
            print(f"{int(row['rank']):<5} {orig_str:<10} {int(row['filtered_idx']):<12} "
                  f"{row['E_per_mol_eV']:<14.4f} {row['density_g_cm3']:<12.3f} {row['dominant_formula']}")
        print(f"\nSaved: {output_xyz}")
        print(f"       {output_csv}")
    
    return df_top, top_structures


def main():
    parser = argparse.ArgumentParser(
        description="Collect UMA-OMC Cell Optimization Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Index Chain:
    orig_idx     : index in pred_combined.xyz (original generated structures)
    filtered_idx : index in filtered_by_inter_bb.xyz (after inter-BB filtering)
    cell_opt_idx : index in combined cell-opt results

Example:
    python collect_relaxation_results.py --top_n 10
    python collect_relaxation_results.py --formula C9H9N3O5 --top_n 10
        """
    )
    parser.add_argument("--results_dir", type=str, default="cell-opt-results",
                        help="Directory containing cell-opt result files")
    parser.add_argument("--log_dir", type=str, default="cell-opt-logs",
                        help="Directory containing optimization log files")
    parser.add_argument("--filter_indices", type=str, default="data/npy/filtered_by_inter_bb_indices.npy",
                        help="NPY file mapping filtered indices to original indices")
    parser.add_argument("--formula", type=str, default=None,
                        help="Chemical formula for organizing outputs (auto-detected if not provided)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <formula>/results)")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top structures to extract for DFT")
    parser.add_argument("--bb_key", type=str, default="bb_idx",
                        help="Key for building block indices in atoms.arrays")
    parser.add_argument("--save_combined_xyz", action="store_true",
                        help="Also save combined XYZ of all cell-opt results")
    parser.add_argument("--plot", action="store_true",
                        help="Generate energy landscape plot (lattice energy vs density)")
    parser.add_argument("--energy_cutoff", type=float, default=20.0,
                        help="Energy cutoff in kJ/mol from minimum for plot (default: 20.0)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("UMA-OMC Cell Optimization Results Collection")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 0: Load filter indices for full traceability
    print("\n[Step 0] Loading index mapping...")
    filtered_to_orig = load_filter_indices(args.filter_indices)
    if filtered_to_orig is not None:
        print(f"  Loaded {len(filtered_to_orig)} filtered → original index mappings")
    
    # Step 1: Combine all cell-opt results
    print("\n[Step 1] Combining cell-opt result files...")
    all_atoms, index_mapping = combine_cell_opt_results(
        results_dir=args.results_dir,
        output_xyz=None,  # Will save later after determining formula
    )
    
    # Determine the dominant formula for folder naming
    if args.formula:
        formula = args.formula
    else:
        # Auto-detect from first structure
        if all_atoms:
            formulas = get_building_block_formulas(all_atoms[0], bb_key=args.bb_key)
            unique_formulas = list(dict.fromkeys(formulas))
            formula = unique_formulas[0] if len(unique_formulas) == 1 else "mixed"
        else:
            formula = "unknown"
    
    print(f"\n  Molecule formula: {formula}")
    
    # Set up output directory based on formula
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(formula, "results")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}")
    
    # Save combined XYZ if requested
    if args.save_combined_xyz:
        combined_xyz = os.path.join(output_dir, "cell_opt_combined.xyz")
        write(combined_xyz, all_atoms, format="extxyz")
        print(f"  Saved combined XYZ: {combined_xyz}")
    
    # Report missing chunks
    if filtered_to_orig is not None:
        n_filtered = len(filtered_to_orig)
        expected_chunks = set(range(0, n_filtered, 10))
        found_chunks = set(m["chunk_start"] for m in index_mapping)
        missing_chunks = sorted(expected_chunks - found_chunks)
        if missing_chunks:
            print(f"\n  ⚠️  Missing chunks (filtered indices): {missing_chunks}")
            missing_orig = [int(filtered_to_orig[c]) for c in missing_chunks if c < len(filtered_to_orig)]
            print(f"      Corresponding original indices: {missing_orig[:10]}...")
    
    # Step 2: Collect detailed relaxation data
    print("\n[Step 2] Collecting relaxation data...")
    df = collect_relaxation_data(
        all_atoms, index_mapping,
        filtered_to_orig=filtered_to_orig,
        log_dir=args.log_dir,
        bb_key=args.bb_key,
    )
    
    # Save full CSV
    full_csv = os.path.join(output_dir, "cell_opt_all_results.csv")
    df.to_csv(full_csv, index=False)
    print(f"\n  Saved full results to: {full_csv}")
    
    # Step 3: Extract top N structures for DFT
    print(f"\n[Step 3] Extracting top {args.top_n} lowest energy structures...")
    df_top, top_structures = extract_top_n_structures(
        df, all_atoms,
        top_n=args.top_n,
        output_dir=output_dir,
    )
    
    # Step 4: Summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")
    print(f"  Molecule: {formula}")
    print(f"  Total structures processed: {len(df)}")
    print(f"  Structures with valid energy: {df['E_per_mol_eV'].notna().sum()}")
    
    if filtered_to_orig is not None:
        n_missing = len(filtered_to_orig) - len(df)
        print(f"  Missing structures (failed/skipped jobs): {n_missing}")
    
    if df['E_per_mol_eV'].notna().any():
        print(f"\n  Energy per molecule (eV):")
        print(f"    Min:  {df['E_per_mol_eV'].min():.4f}")
        print(f"    Max:  {df['E_per_mol_eV'].max():.4f}")
        print(f"    Mean: {df['E_per_mol_eV'].mean():.4f}")
        print(f"    Std:  {df['E_per_mol_eV'].std():.4f}")
    
    print(f"\n  Density (g/cm³):")
    print(f"    Min:  {df['density_g_cm3'].min():.3f}")
    print(f"    Max:  {df['density_g_cm3'].max():.3f}")
    print(f"    Mean: {df['density_g_cm3'].mean():.3f}")
    
    # Step 5: Generate energy landscape plot if requested
    if args.plot:
        print(f"\n[Step 5] Generating energy landscape plot...")
        plot_path = os.path.join(output_dir, f"energy_landscape_{formula}.pdf")
        plot_energy_landscape(
            df,
            formula=formula,
            output_path=plot_path,
            energy_cutoff_kj=args.energy_cutoff,
            highlight_top_n=args.top_n,
        )
    
    print(f"\n{'='*70}")
    print(f"Output files in: {output_dir}/")
    print(f"  - cell_opt_all_results.csv  (all {len(df)} structures)")
    print(f"  - top{args.top_n}_for_dft.xyz        (top {args.top_n} for DFT)")
    print(f"  - top{args.top_n}_for_dft.csv        (metadata for top {args.top_n})")
    if args.save_combined_xyz:
        print(f"  - cell_opt_combined.xyz     (combined XYZ)")
    if args.plot:
        print(f"  - energy_landscape_{formula}.pdf")
    print(f"{'='*70}")
    print("Done!")


if __name__ == "__main__":
    main()
