#!/usr/bin/env python3
"""
Collect and analyze VASP DFT results for crystal structure ranking.

This script parses VASP output files (OUTCAR, OSZICAR, CONTCAR) from completed
DFT jobs and compiles results into a summary CSV for comparison and ranking.

Features:
    - Collect energies, densities, lattice parameters from VASP outputs
    - Extract relaxed structures (CONTCAR) as XYZ files with unwrapped molecules
    - Generate stability ranking plots comparing u-MLIP vs DFT
    - Save top-N structures for further analysis
    - Include ground-truth/experimental energies in CSVs and plots

Usage:
    # Collect results and generate plots
    python collect_dft_results.py --input_dir ./C9H9N3O5 --plot

    # With ground truth energies for comparison
    python collect_dft_results.py --input_dir ./C9H9N3O5 --plot --gt_energies ../gt_energies.json

    # Extract top-4 MBD structures
    python collect_dft_results.py --input_dir ./C9H9N3O5 --top_n 4

Output:
    <formula>/
    ├── dft_results_pbe_d3.csv
    ├── dft_results_pbe_mbd.csv
    ├── dft_combined_ranking.csv
    ├── relaxed_structures/
    │   ├── pbe_d3_all.xyz         (unwrapped molecules)
    │   ├── pbe_mbd_all.xyz        (unwrapped molecules)
    │   └── top4_pbe_mbd.xyz       (unwrapped molecules)
    └── stability_ranking_<formula>.pdf
"""

import argparse
import json
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ase import Atoms, io
from ase.io import read, write
from ase.neighborlist import NeighborList


# ==============================================================================
# Covalent radii cutoffs for neighbor / molecule detection (Å)
# ==============================================================================

ELEMENT_CUTOFFS: Dict[str, float] = {
    "H": 0.29,
    "C": 0.70,
    "N": 0.48,
    "O": 0.46,
    "F": 0.68,
    "S": 1.10,
    "Cl": 1.10,
    "P": 1.20,
    "Br": 1.20,
    "I": 1.40,
    "B": 0.85,
    "Si": 1.20,
}


# ==============================================================================
# Molecule Unwrapping
# ==============================================================================

def find_all_neighbors(
    atoms: Atoms, nl: NeighborList, start_node: int
) -> Dict[int, np.ndarray]:
    """Find all connected atoms using BFS (breadth-first search).

    Args:
        atoms: ASE Atoms object.
        nl: Pre-computed ASE NeighborList.
        start_node: Index of the starting atom.

    Returns:
        Dictionary mapping atom index → cumulative displacement vector.
    """
    visited: set = set()
    queue = deque([(start_node, np.zeros(3))])
    neighbors_group: Dict[int, np.ndarray] = {start_node: np.zeros(3)}

    while queue:
        node, offset = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        indices, _offsets = nl.get_neighbors(node)
        for neighbor in indices:
            if neighbor in neighbors_group:
                continue
            vec = atoms.get_distances(node, [neighbor], mic=True, vector=True)[0]
            new_offset = offset + vec
            neighbors_group[neighbor] = new_offset
            queue.append((neighbor, new_offset))

    return neighbors_group


def get_molecule_groups(
    atoms: Atoms,
) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Identify molecular groups (connected components) in the crystal structure.

    Args:
        atoms: ASE Atoms object.

    Returns:
        molecule_groups: List of lists containing atom indices for each molecule.
        offset_groups: List of arrays containing position offsets for each molecule.
    """
    symbols = atoms.get_chemical_symbols()

    # Check for unknown elements and use a default cutoff
    cutoffs = []
    for s in symbols:
        if s in ELEMENT_CUTOFFS:
            cutoffs.append(ELEMENT_CUTOFFS[s])
        else:
            print(f"  Warning: Unknown element '{s}', using default cutoff 1.0 Å")
            cutoffs.append(1.0)

    nl = NeighborList(
        cutoffs=np.array(cutoffs), self_interaction=False, bothways=True
    )
    nl.update(atoms)

    molecule_groups: List[List[int]] = []
    offset_groups: List[np.ndarray] = []
    chosen: set = set()

    for i in range(len(atoms)):
        if i in chosen:
            continue
        neighbors_map = find_all_neighbors(atoms, nl, i)
        chosen.update(neighbors_map.keys())
        molecule_groups.append(list(neighbors_map.keys()))
        offset_groups.append(np.array(list(neighbors_map.values())))

    return molecule_groups, offset_groups


def unwrap_molecules(atoms: Atoms) -> Atoms:
    """
    Unwrap molecules across periodic boundaries so each is fully connected.
    
    This ensures that all atoms belonging to the same molecule are placed
    in the same periodic image, making the molecule visually connected
    in Cartesian coordinates.

    Args:
        atoms: ASE Atoms object with periodic boundary conditions.

    Returns:
        unwrapped_atoms: ASE Atoms object with updated positions.
    """
    try:
        molecule_groups, offset_groups = get_molecule_groups(atoms)
    except Exception as e:
        print(f"  Warning: Could not unwrap molecules: {e}")
        return atoms
    
    unwrapped_positions = np.zeros_like(atoms.positions)

    for mol, offsets in zip(molecule_groups, offset_groups):
        ref_pos = atoms.positions[mol[0]]
        # Offsets are relative vectors that place all atoms in the same image
        new_positions = ref_pos + offsets
        unwrapped_positions[mol] = new_positions

    # Create a copy so we don't modify the original object in-place
    unwrapped_atoms = atoms.copy()
    unwrapped_atoms.positions = unwrapped_positions
    unwrapped_atoms.center()

    return unwrapped_atoms


# ==============================================================================
# VASP Output Parsing
# ==============================================================================

def parse_outcar(outcar_path: Path) -> Dict:
    """Parse VASP OUTCAR file for key results."""
    results = {
        "converged": False,
        "total_energy": None,
        "n_ionic_steps": 0,
        "max_force": None,
        "volume": None,
        "pressure": None,
    }
    
    if not outcar_path.exists():
        return results
    
    try:
        with open(outcar_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Warning: Could not read {outcar_path}: {e}")
        return results
    
    # Parse from end of file (most recent values)
    for line in reversed(lines):
        # Final energy
        if "free  energy   TOTEN" in line and results["total_energy"] is None:
            match = re.search(r"=\s*([-\d.]+)", line)
            if match:
                results["total_energy"] = float(match.group(1))
        
        # Volume
        if "volume of cell" in line and results["volume"] is None:
            match = re.search(r":\s*([\d.]+)", line)
            if match:
                results["volume"] = float(match.group(1))
        
        # Pressure
        if "external pressure" in line and results["pressure"] is None:
            match = re.search(r"=\s*([-\d.]+)\s*kB", line)
            if match:
                results["pressure"] = float(match.group(1))
    
    # Count ionic steps
    for line in lines:
        if "F=" in line or "E0=" in line:
            results["n_ionic_steps"] += 1
    
    # Check convergence (reached required accuracy)
    for line in reversed(lines[-100:]):
        if "reached required accuracy" in line:
            results["converged"] = True
            break
    
    # Get max force from CONTCAR or final ionic step
    for line in reversed(lines):
        if "TOTAL-FORCE" in line:
            # Parse force block
            try:
                idx = lines.index(line)
                force_lines = []
                for fl in lines[idx+2:]:
                    if "---" in fl or not fl.strip():
                        break
                    parts = fl.split()
                    if len(parts) >= 6:
                        fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                        force_lines.append(np.sqrt(fx**2 + fy**2 + fz**2))
                if force_lines:
                    results["max_force"] = max(force_lines)
            except Exception:
                pass
            break
    
    return results


def parse_oszicar(oszicar_path: Path) -> Dict:
    """Parse VASP OSZICAR file for electronic convergence info."""
    results = {
        "n_electronic_steps": 0,
        "final_e0": None,
    }
    
    if not oszicar_path.exists():
        return results
    
    try:
        with open(oszicar_path, "r") as f:
            lines = f.readlines()
    except Exception:
        return results
    
    for line in reversed(lines):
        if "E0=" in line:
            match = re.search(r"E0=\s*([-\d.E+]+)", line)
            if match:
                results["final_e0"] = float(match.group(1))
            break
    
    results["n_electronic_steps"] = sum(1 for line in lines if "DAV:" in line or "RMM:" in line)
    
    return results


def get_lattice_params(contcar_path: Path) -> Dict:
    """Get lattice parameters from CONTCAR."""
    results = {
        "a": None, "b": None, "c": None,
        "alpha": None, "beta": None, "gamma": None,
    }
    
    if not contcar_path.exists():
        return results
    
    try:
        atoms = read(str(contcar_path), format="vasp")
        cell = atoms.get_cell()
        lengths = cell.lengths()
        angles = cell.angles()
        
        results["a"] = lengths[0]
        results["b"] = lengths[1]
        results["c"] = lengths[2]
        results["alpha"] = angles[0]
        results["beta"] = angles[1]
        results["gamma"] = angles[2]
        results["n_atoms"] = len(atoms)
    except Exception as e:
        print(f"  Warning: Could not parse {contcar_path}: {e}")
    
    return results


def compute_density(n_atoms: int, volume: float, formula: str) -> float:
    """Compute density in g/cm³ from volume and formula."""
    # Estimate molecular weight from formula
    from ase.data import atomic_masses, atomic_numbers
    
    mass = 0.0
    # Simple parsing: C9H9N3O5
    element = ""
    count_str = ""
    
    for char in formula + "X":  # X to flush last element
        if char.isupper():
            if element:
                count = int(count_str) if count_str else 1
                try:
                    mass += atomic_masses[atomic_numbers[element]] * count
                except KeyError:
                    pass
            element = char
            count_str = ""
        elif char.islower():
            element += char
        elif char.isdigit():
            count_str += char
    
    # volume in Å³, mass in g/mol
    # density = mass / (volume * N_A) in g/Å³, convert to g/cm³
    avogadro = 6.022e23
    volume_cm3 = volume * 1e-24  # Å³ to cm³
    
    # Assuming n_atoms corresponds to some number of formula units
    # This is approximate - ideally parse from structure
    n_formula_units = 2  # Typically Z=2 for these crystals
    total_mass_g = (mass * n_formula_units) / avogadro
    
    return total_mass_g / volume_cm3 if volume_cm3 > 0 else 0.0


# ==============================================================================
# Results Collection
# ==============================================================================

def collect_relaxed_structure(contcar_path: Path, energy: float = None, 
                               job_idx: int = None, theory: str = None) -> Optional[Atoms]:
    """Read relaxed structure from CONTCAR and add metadata."""
    if not contcar_path.exists():
        return None
    
    try:
        atoms = read(str(contcar_path), format="vasp")
        # Add metadata to atoms info
        if energy is not None:
            atoms.info["energy"] = energy
            atoms.info["E_per_mol_eV"] = energy / 2  # Assuming Z=2
        if job_idx is not None:
            atoms.info["job_idx"] = job_idx
        if theory is not None:
            atoms.info["theory"] = theory
        return atoms
    except Exception as e:
        print(f"  Warning: Could not read {contcar_path}: {e}")
        return None


def collect_theory_results(
    theory_dir: Path,
    theory: str,
    formula: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Collect results from all job directories for one theory."""
    if not theory_dir.exists():
        if verbose:
            print(f"  Directory not found: {theory_dir}")
        return pd.DataFrame()
    
    job_dirs = sorted([d for d in theory_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if verbose:
        print(f"  Found {len(job_dirs)} job directories in {theory_dir}")
    
    results = []
    
    for job_dir in job_dirs:
        job_idx = int(job_dir.name)
        
        outcar = job_dir / "OUTCAR"
        oszicar = job_dir / "OSZICAR"
        contcar = job_dir / "CONTCAR"
        
        # Check if job has run
        if not outcar.exists():
            if verbose:
                print(f"    Job {job_idx:03d}: Not started")
            continue
        
        # Parse results
        outcar_data = parse_outcar(outcar)
        oszicar_data = parse_oszicar(oszicar)
        lattice_data = get_lattice_params(contcar)
        
        # Compute derived quantities
        n_atoms = lattice_data.get("n_atoms", 0)
        volume = outcar_data.get("volume", 0)
        energy = outcar_data.get("total_energy")
        
        # Energy per molecule (assuming Z=2 typically)
        n_molecules = 2  # Adjust based on your structures
        e_per_mol = energy / n_molecules if energy is not None else None
        
        # Density
        density = compute_density(n_atoms, volume, formula) if volume and n_atoms else None
        
        row = {
            "job_idx": job_idx,
            "theory": f"pbe-{theory}",
            "converged": outcar_data["converged"],
            "n_ionic_steps": outcar_data["n_ionic_steps"],
            "total_energy_eV": energy,
            "E_per_mol_eV": e_per_mol,
            "max_force_eV_A": outcar_data["max_force"],
            "volume_A3": volume,
            "pressure_kB": outcar_data["pressure"],
            "density_g_cm3": density,
            **{k: v for k, v in lattice_data.items() if k != "n_atoms"},
            "n_atoms": n_atoms,
        }
        
        results.append(row)
        
        status = "✅" if outcar_data["converged"] else "⏳"
        if verbose and job_idx < 5:  # Only show first few
            e_str = f"{energy:.4f}" if energy else "N/A"
            print(f"    Job {job_idx:03d}: {status} E={e_str} eV, {outcar_data['n_ionic_steps']} steps")
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        # Sort by energy
        df = df.sort_values("total_energy_eV", ascending=True, na_position="last")
        df["rank"] = range(1, len(df) + 1)
    
    return df


def collect_all_results(
    input_dir: str,
    theories: List[str] = ["d3", "mbd"],
    output_dir: Optional[str] = None,
    formula: Optional[str] = None,
    gt_energies: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Collect results from all DFT calculations.
    
    Args:
        input_dir: Directory containing DFT job folders.
        theories: List of DFT theories to collect (d3, mbd).
        output_dir: Output directory for CSV files.
        formula: Chemical formula (auto-detected from directory name if not provided).
        gt_energies: Ground truth energies dict from JSON file.
        verbose: Print progress.
        
    Returns:
        Dictionary of DataFrames with results for each theory.
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Auto-detect formula from directory name
    if formula is None:
        formula = input_dir.name
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"{'='*60}")
        print(f"Collecting DFT Results for {formula}")
        print(f"{'='*60}")
    
    all_results = {}
    
    for theory in theories:
        theory_dir = input_dir / f"pbe-{theory}"
        
        if verbose:
            print(f"\nProcessing PBE-{theory.upper()}...")
        
        df = collect_theory_results(theory_dir, theory, formula, verbose=verbose)
        
        if len(df) > 0:
            # Save individual results
            csv_path = output_dir / f"dft_results_pbe_{theory}.csv"
            df.to_csv(csv_path, index=False)
            if verbose:
                print(f"  Saved: {csv_path}")
            
            all_results[theory] = df
    
    # Create combined ranking if both theories available
    if "d3" in all_results and "mbd" in all_results:
        df_d3 = all_results["d3"].copy()
        df_mbd = all_results["mbd"].copy()
        
        # Merge on job_idx
        df_combined = pd.merge(
            df_d3[["job_idx", "total_energy_eV", "E_per_mol_eV", "converged"]].rename(
                columns={
                    "total_energy_eV": "E_d3_eV",
                    "E_per_mol_eV": "E_per_mol_d3_eV",
                    "converged": "converged_d3",
                }
            ),
            df_mbd[["job_idx", "total_energy_eV", "E_per_mol_eV", "converged", 
                    "density_g_cm3", "a", "b", "c", "alpha", "beta", "gamma"]].rename(
                columns={
                    "total_energy_eV": "E_mbd_eV",
                    "E_per_mol_eV": "E_per_mol_mbd_eV",
                    "converged": "converged_mbd",
                }
            ),
            on="job_idx",
            how="outer",
        )
        
        # Sort by MBD energy (usually more accurate for molecular crystals)
        df_combined = df_combined.sort_values("E_mbd_eV", ascending=True, na_position="last")
        df_combined["rank_mbd"] = range(1, len(df_combined) + 1)
        
        # Also rank by D3
        df_combined = df_combined.sort_values("E_d3_eV", ascending=True, na_position="last")
        df_combined["rank_d3"] = range(1, len(df_combined) + 1)
        
        # Final sort by MBD
        df_combined = df_combined.sort_values("E_mbd_eV", ascending=True, na_position="last")
        
        # Add ground-truth experimental energies if available
        if gt_energies and formula in gt_energies:
            gt = gt_energies[formula]
            # Add GT energies as columns
            df_combined["GT_E_per_mol_d3_eV"] = gt.get("PBE-D3")
            df_combined["GT_E_per_mol_mbd_eV"] = gt.get("PBE-MBD")
            df_combined["GT_E_per_mol_umlip_eV"] = gt.get("u-MLIP")
            
            # Compute relative energies to GT
            if gt.get("PBE-MBD") is not None:
                df_combined["rel_E_mbd_to_GT_eV"] = df_combined["E_per_mol_mbd_eV"] - gt["PBE-MBD"]
            if gt.get("PBE-D3") is not None:
                df_combined["rel_E_d3_to_GT_eV"] = df_combined["E_per_mol_d3_eV"] - gt["PBE-D3"]
            
            if verbose:
                print(f"\n  Ground-truth energies for {formula}:")
                print(f"    PBE-D3: {gt.get('PBE-D3', 'N/A')} eV/mol")
                print(f"    PBE-MBD: {gt.get('PBE-MBD', 'N/A')} eV/mol")
                print(f"    u-MLIP: {gt.get('u-MLIP', 'N/A')} eV/mol")
        
        csv_path = output_dir / "dft_combined_ranking.csv"
        df_combined.to_csv(csv_path, index=False)
        if verbose:
            print(f"\nSaved combined ranking: {csv_path}")
        
        all_results["combined"] = df_combined
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        
        for theory in theories:
            if theory in all_results:
                df = all_results[theory]
                n_converged = df["converged"].sum()
                n_total = len(df)
                if n_total > 0:
                    best_e = df["E_per_mol_eV"].min()
                    print(f"  PBE-{theory.upper()}: {n_converged}/{n_total} converged, best E={best_e:.4f} eV/mol")
        
        if "combined" in all_results:
            df = all_results["combined"]
            print(f"\n  Top 3 by PBE-MBD:")
            for _, row in df.head(3).iterrows():
                print(f"    #{int(row['rank_mbd'])}: job_{int(row['job_idx']):03d} "
                      f"E_mbd={row['E_mbd_eV']:.4f} eV, "
                      f"E_d3={row['E_d3_eV']:.4f} eV")
            
            # Print GT comparison if available
            if "GT_E_per_mol_mbd_eV" in df.columns and df["GT_E_per_mol_mbd_eV"].notna().any():
                gt_mbd = df["GT_E_per_mol_mbd_eV"].iloc[0]
                best_mbd = df["E_per_mol_mbd_eV"].min()
                print(f"\n  Ground-truth PBE-MBD: {gt_mbd:.4f} eV/mol")
                print(f"  Best predicted: {best_mbd:.4f} eV/mol (Δ = {best_mbd - gt_mbd:.4f} eV)")
        
        print(f"{'='*60}")
    
    return all_results


# ==============================================================================
# Structure Extraction
# ==============================================================================

def extract_relaxed_structures(
    input_dir: Path,
    df_results: pd.DataFrame,
    theory: str,
    output_dir: Path,
    unwrap: bool = True,
    verbose: bool = True,
) -> List[Atoms]:
    """Extract relaxed structures from CONTCAR files.
    
    Args:
        input_dir: Directory containing DFT job folders.
        df_results: DataFrame with DFT results.
        theory: DFT theory level (d3 or mbd).
        output_dir: Directory to save XYZ files.
        unwrap: If True, unwrap molecules across periodic boundaries.
        verbose: Print progress.
        
    Returns:
        List of ASE Atoms objects.
    """
    theory_dir = input_dir / f"pbe-{theory}"
    structures = []
    
    for _, row in df_results.iterrows():
        job_idx = int(row["job_idx"])
        contcar_path = theory_dir / f"{job_idx:03d}" / "CONTCAR"
        
        atoms = collect_relaxed_structure(
            contcar_path,
            energy=row.get("total_energy_eV"),
            job_idx=job_idx,
            theory=f"pbe-{theory}",
        )
        
        if atoms is not None:
            # Unwrap molecules so they are fully connected in Cartesian coordinates
            if unwrap:
                atoms = unwrap_molecules(atoms)
            
            # Add more metadata
            atoms.info["rank"] = int(row.get("rank", 0))
            atoms.info["density_g_cm3"] = row.get("density_g_cm3", 0)
            atoms.info["converged"] = row.get("converged", False)
            structures.append(atoms)
    
    if structures and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        xyz_path = output_dir / f"pbe_{theory}_all.xyz"
        write(str(xyz_path), structures, format="extxyz")
        if verbose:
            print(f"  Saved {len(structures)} unwrapped relaxed structures to {xyz_path}")
    
    return structures


def extract_top_n_structures(
    input_dir: Path,
    df_combined: pd.DataFrame,
    theory: str,
    n: int,
    output_dir: Path,
    unwrap: bool = True,
    verbose: bool = True,
) -> List[Atoms]:
    """Extract top-N structures by energy for a given theory.
    
    Args:
        input_dir: Directory containing DFT job folders.
        df_combined: Combined DataFrame with DFT results.
        theory: DFT theory level (d3 or mbd).
        n: Number of top structures to extract.
        output_dir: Directory to save XYZ files.
        unwrap: If True, unwrap molecules across periodic boundaries.
        verbose: Print progress.
        
    Returns:
        List of ASE Atoms objects.
    """
    theory_dir = input_dir / f"pbe-{theory}"
    
    # Get top-N by energy
    energy_col = f"E_{theory}_eV"
    if energy_col not in df_combined.columns:
        energy_col = "E_per_mol_eV" if "E_per_mol_eV" in df_combined.columns else None
    
    if energy_col:
        df_sorted = df_combined.sort_values(energy_col, ascending=True).head(n)
    else:
        df_sorted = df_combined.head(n)
    
    structures = []
    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        job_idx = int(row["job_idx"])
        contcar_path = theory_dir / f"{job_idx:03d}" / "CONTCAR"
        
        energy = row.get(f"E_{theory}_eV") or row.get("total_energy_eV")
        atoms = collect_relaxed_structure(
            contcar_path,
            energy=energy,
            job_idx=job_idx,
            theory=f"pbe-{theory}",
        )
        
        if atoms is not None:
            # Unwrap molecules so they are fully connected in Cartesian coordinates
            if unwrap:
                atoms = unwrap_molecules(atoms)
            
            atoms.info["rank"] = rank
            atoms.info["density_g_cm3"] = row.get("density_g_cm3", 0)
            structures.append(atoms)
    
    if structures and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        xyz_path = output_dir / f"top{n}_pbe_{theory}.xyz"
        write(str(xyz_path), structures, format="extxyz")
        if verbose:
            print(f"  Saved top-{n} unwrapped structures to {xyz_path}")
    
    return structures


# ==============================================================================
# Stability Ranking Plot
# ==============================================================================

def plot_stability_ranking(
    df_results: pd.DataFrame,
    formula: str,
    output_dir: Path,
    gt_energies: Optional[Dict] = None,
    umlip_results_path: Optional[str] = None,
    top_n: int = 4,
    verbose: bool = True,
) -> None:
    """
    Generate stability ranking plot comparing u-MLIP, PBE-D3, and PBE-MBD.
    
    Args:
        df_results: DataFrame with combined DFT results
        formula: Chemical formula
        output_dir: Output directory for the plot
        gt_energies: Ground truth energies dict (optional)
        umlip_results_path: Path to u-MLIP relaxation results CSV (optional)
        top_n: Number of top structures to plot
        verbose: Print progress
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Warning: matplotlib not available, skipping plot generation")
        return
    
    # Load u-MLIP results if provided
    # The u-MLIP CSV should be the top10_for_dft.csv which has 'rank' column
    # DFT job_idx 0 corresponds to rank 1, job_idx 1 to rank 2, etc.
    umlip_energies = {}
    if umlip_results_path:
        try:
            df_umlip = pd.read_csv(umlip_results_path)
            # Check if this is the top10_for_dft.csv (has 'rank' column)
            if "rank" in df_umlip.columns:
                # Map rank to E_per_mol_eV (rank 1 -> job_idx 0, rank 2 -> job_idx 1, ...)
                for _, row in df_umlip.iterrows():
                    job_idx = int(row["rank"]) - 1  # Convert rank to 0-indexed job_idx
                    umlip_energies[job_idx] = row["E_per_mol_eV"]
                if verbose:
                    print(f"  Loaded u-MLIP energies from {umlip_results_path} (using rank mapping)")
            else:
                # Fallback: assume cell_opt_idx directly maps to job_idx
                for _, row in df_umlip.iterrows():
                    umlip_energies[int(row["cell_opt_idx"])] = row["E_per_mol_eV"]
                if verbose:
                    print(f"  Loaded u-MLIP energies from {umlip_results_path} (using cell_opt_idx)")
        except Exception as e:
            print(f"  Warning: Could not load u-MLIP results: {e}")
    
    # Plot settings
    plt.rcParams.update({
        "font.size": 14,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    methods = ["u-MLIP", "PBE-D3", "PBE-MBD"]
    x = np.arange(len(methods))
    x_mid = methods.index("PBE-D3")
    
    # Nature-style colors
    colors = ["#e64b35", "#4dbbd5", "#00a087", "#3c5488", "#f39b7f"]
    poly_labels = {0: r"$\alpha$", 1: r"$\beta$", 2: r"$\gamma$", 3: r"$\delta$", "GT": "GT"}
    
    segment_width = 0.12
    EV_TO_KJMOL = 96.485
    
    # Prepare data
    # Get top-N structures sorted by MBD energy
    df_sorted = df_results.sort_values("E_mbd_eV", ascending=True).head(top_n).copy()
    df_sorted = df_sorted.reset_index(drop=True)
    
    # Build energy table: rows = polymorphs, cols = methods
    energies = {}
    for idx, row in df_sorted.iterrows():
        job_idx = int(row["job_idx"])
        # Get u-MLIP energy from the relaxation results (job_idx matches cell_opt_idx)
        umlip_e = umlip_energies.get(job_idx)
        energies[idx] = {
            "u-MLIP": umlip_e,
            "PBE-D3": row.get("E_per_mol_d3_eV"),
            "PBE-MBD": row.get("E_per_mol_mbd_eV"),
        }
    
    # Add ground truth if available
    if gt_energies and formula in gt_energies:
        gt = gt_energies[formula]
        energies["GT"] = {
            "u-MLIP": gt.get("u-MLIP"),
            "PBE-D3": gt.get("PBE-D3"),
            "PBE-MBD": gt.get("PBE-MBD"),
        }
    
    # Convert to relative energies (kJ/mol)
    # Find minimum for each method
    min_energies = {}
    for method in methods:
        vals = [e[method] for e in energies.values() if e.get(method) is not None]
        min_energies[method] = min(vals) if vals else 0
    
    # Compute relative energies
    rel_energies = {}
    for poly, e_dict in energies.items():
        rel_energies[poly] = {}
        for method in methods:
            if e_dict.get(method) is not None:
                rel_energies[poly][method] = (e_dict[method] - min_energies[method]) * EV_TO_KJMOL
            else:
                rel_energies[poly][method] = np.nan
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    
    for i, (poly, rel) in enumerate(rel_energies.items()):
        y = np.array([rel.get(m, np.nan) for m in methods])
        
        if np.all(np.isnan(y)):
            continue
        
        color = colors[i % len(colors)]
        label = poly_labels.get(poly, f"#{poly}")
        
        # Connect bar endpoints
        for j in range(len(x) - 1):
            if not (np.isnan(y[j]) or np.isnan(y[j + 1])):
                ax.plot(
                    [x[j] + segment_width, x[j + 1] - segment_width],
                    [y[j], y[j + 1]],
                    linestyle=":",
                    color=color,
                    linewidth=2,
                )
        
        # Thick horizontal bar segments
        for xi, yi in zip(x, y):
            if not np.isnan(yi):
                ax.plot(
                    [xi - segment_width, xi + segment_width],
                    [yi, yi],
                    color=color,
                    linewidth=6,
                    solid_capstyle="round",
                )
        
        # Label on PBE-D3 bar
        y_mid = y[x_mid]
        if not np.isnan(y_mid):
            ax.text(
                x[x_mid],
                y_mid + 0.3,
                label,
                color=color,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_xlabel("CSP Ranking Methods")
    ax.set_ylabel(r"Relative energy (kJ/mol)")
    ax.set_title(f"Stability Ranking: {formula}")
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"stability_ranking_{formula}.pdf"
    png_path = output_dir / f"stability_ranking_{formula}.png"
    
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()
    
    if verbose:
        print(f"  Saved stability ranking plot: {pdf_path}")


# ==============================================================================
# Main Collection with All Features
# ==============================================================================

def collect_and_analyze(
    input_dir: str,
    theories: List[str] = ["d3", "mbd"],
    output_dir: Optional[str] = None,
    formula: Optional[str] = None,
    top_n: int = 4,
    extract_structures: bool = True,
    generate_plot: bool = False,
    gt_energies_path: Optional[str] = None,
    umlip_results_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Collect DFT results, extract structures, and generate plots.
    
    This is the main entry point combining all functionality.
    
    Args:
        input_dir: Directory containing pbe-d3/ and pbe-mbd/ subdirectories.
        theories: List of DFT theories to collect.
        output_dir: Output directory (default: same as input_dir).
        formula: Chemical formula (auto-detected from directory name).
        top_n: Number of top structures to extract.
        extract_structures: Whether to extract relaxed structures as XYZ.
        generate_plot: Whether to generate stability ranking plot.
        gt_energies_path: Path to ground-truth energies JSON file.
        umlip_results_path: Path to u-MLIP relaxation results CSV.
        verbose: Print progress.
        
    Returns:
        Dictionary of DataFrames with results.
    """
    input_dir = Path(input_dir)
    
    # Auto-detect formula
    if formula is None:
        formula = input_dir.name
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
    
    # Load ground truth energies early (used for both CSV and plot)
    gt_energies = None
    if gt_energies_path:
        try:
            with open(gt_energies_path, "r") as f:
                gt_energies = json.load(f)
            if verbose:
                print(f"Loaded ground truth energies from {gt_energies_path}")
        except Exception as e:
            print(f"Warning: Could not load ground truth: {e}")
    
    # Collect results (pass gt_energies to include in CSVs)
    all_results = collect_all_results(
        input_dir=str(input_dir),
        theories=theories,
        output_dir=str(output_dir),
        formula=formula,
        gt_energies=gt_energies,
        verbose=verbose,
    )
    
    # Extract relaxed structures (with unwrapping)
    if extract_structures:
        struct_dir = output_dir / "relaxed_structures"
        
        if verbose:
            print(f"\nExtracting relaxed structures (with molecule unwrapping)...")
        
        for theory in theories:
            if theory in all_results:
                extract_relaxed_structures(
                    input_dir=input_dir,
                    df_results=all_results[theory],
                    theory=theory,
                    output_dir=struct_dir,
                    unwrap=True,
                    verbose=verbose,
                )
        
        # Extract top-N by MBD
        if "combined" in all_results and "mbd" in theories:
            extract_top_n_structures(
                input_dir=input_dir,
                df_combined=all_results["combined"],
                theory="mbd",
                n=top_n,
                output_dir=struct_dir,
                unwrap=True,
                verbose=verbose,
            )
    
    # Generate stability ranking plot
    if generate_plot and "combined" in all_results:
        if verbose:
            print(f"\nGenerating stability ranking plot...")
        
        plot_stability_ranking(
            df_results=all_results["combined"],
            formula=formula,
            output_dir=output_dir,
            gt_energies=gt_energies,
            umlip_results_path=umlip_results_path,
            top_n=top_n,
            verbose=verbose,
        )
    
    return all_results


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect and analyze VASP DFT results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect all results
  python collect_dft_results.py --input_dir ./C9H9N3O5

  # Collect results and generate stability plot
  python collect_dft_results.py --input_dir ./C9H9N3O5 --plot

  # With ground truth comparison
  python collect_dft_results.py --input_dir ./C9H9N3O5 --plot --gt_energies ../gt_energies.json

  # Extract top-4 structures only
  python collect_dft_results.py --input_dir ./C9H9N3O5 --top_n 4

Output:
  <formula>/
  ├── dft_results_pbe_d3.csv
  ├── dft_results_pbe_mbd.csv
  ├── dft_combined_ranking.csv
  ├── relaxed_structures/
  │   ├── pbe_d3_all.xyz
  │   ├── pbe_mbd_all.xyz
  │   └── top4_pbe_mbd.xyz
  └── stability_ranking_<formula>.pdf
        """,
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Directory containing pbe-d3/ and pbe-mbd/ subdirectories",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Output directory for CSV files (default: same as input)",
    )
    parser.add_argument(
        "--formula", "-f",
        type=str,
        default=None,
        help="Chemical formula (auto-detected from directory name)",
    )
    parser.add_argument(
        "--theories", "-t",
        type=str,
        nargs="+",
        default=["d3", "mbd"],
        choices=["d3", "mbd"],
        help="DFT theories to collect (default: d3 mbd)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=4,
        help="Number of top structures to extract (default: 4)",
    )
    parser.add_argument(
        "--no_structures",
        action="store_true",
        help="Skip extraction of relaxed structures",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate stability ranking plot",
    )
    parser.add_argument(
        "--gt_energies",
        type=str,
        default=None,
        help="Path to ground truth energies JSON file",
    )
    parser.add_argument(
        "--umlip_results",
        type=str,
        default=None,
        help="Path to u-MLIP relaxation results CSV (for plotting)",
    )
    
    args = parser.parse_args()
    
    collect_and_analyze(
        input_dir=args.input_dir,
        theories=args.theories,
        output_dir=args.output_dir,
        formula=args.formula,
        top_n=args.top_n,
        extract_structures=not args.no_structures,
        generate_plot=args.plot,
        gt_energies_path=args.gt_energies,
        umlip_results_path=args.umlip_results,
        verbose=True,
    )


if __name__ == "__main__":
    main()
