#!/usr/bin/env python3
"""
Setup VASP DFT jobs for top-N UMA-OMC relaxed structures.

This script reads top-N structures from the relaxation pipeline output and creates
VASP job directories for both PBE-D3(BJ) and PBE-MBD levels of theory.

Usage:
    # From the dft-run folder:
    python setup_dft_jobs.py --input ../uma-s1p1-omc-opt/C9H9N3O5/results/top10_for_dft.xyz

    # Or specify formula explicitly:
    python setup_dft_jobs.py --input top10.xyz --formula C9H9N3O5

    # Custom output directory:
    python setup_dft_jobs.py --input top10.xyz --output_dir ./my_dft_jobs

Output Structure:
    <formula>/
    ├── pbe-d3/
    │   ├── 000/  (POSCAR, INCAR, KPOINTS, POTCAR, run_vasp.slurm)
    │   ├── 001/
    │   └── submit_all.sh
    └── pbe-mbd/
        ├── 000/
        ├── 001/
        └── submit_all.sh
"""

import argparse
import os
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms, io
from ase.io.vasp import write_vasp

try:
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp.inputs import Kpoints
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    print("Warning: pymatgen not available, using fallback k-point generation")


# ==============================================================================
# Configuration
# ==============================================================================

# Default paths (HiPerGator)
DEFAULT_POTCAR_ROOT = "/orange/mingjieliu/zhiyu/potpaw_PBE"

# VASP settings
DEFAULT_ENCUT = 520
DEFAULT_KPPA = 1000  # k-points per atom for pymatgen

# SLURM settings
SLURM_ACCOUNT = "mingjieliu"
SLURM_QOS = "mingjieliu"
SLURM_TIME_D3 = "2-00:00:00"
SLURM_TIME_MBD = "3-00:00:00"  # MBD can be slower
SLURM_NODES = 1
SLURM_NTASKS = 32
SLURM_MEM_PER_CPU = "2gb"


# ==============================================================================
# K-point Generation
# ==============================================================================

def get_kpoints_fallback(atoms: Atoms, spacing: float = 0.2, min_k: int = 1) -> Tuple[int, int, int]:
    """Fallback k-point generation based on reciprocal cell lengths."""
    try:
        rec = atoms.cell.reciprocal()
        rec_lengths = [np.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in rec]
        nk = [max(min_k, int(np.ceil(l / (2 * np.pi * spacing)))) for l in rec_lengths]
        return tuple(nk)
    except Exception:
        return (2, 2, 2)  # Safe default


def get_kpoints_pymatgen(atoms: Atoms, kppa: int = 1000) -> Tuple[int, int, int]:
    """Use pymatgen's automatic k-point generation."""
    if not HAS_PYMATGEN:
        return get_kpoints_fallback(atoms)
    
    try:
        adaptor = AseAtomsAdaptor()
        structure = adaptor.get_structure(atoms)
        kpoints = Kpoints.automatic_density(structure, kppa=kppa)
        
        if hasattr(kpoints, 'kpts') and len(kpoints.kpts) > 0:
            return tuple(kpoints.kpts[0])
        else:
            return get_kpoints_fallback(atoms)
    except Exception as e:
        print(f"Warning: pymatgen k-point generation failed: {e}")
        return get_kpoints_fallback(atoms)


# ==============================================================================
# INCAR Templates
# ==============================================================================

def get_incar_pbe_d3(system_name: str, encut: int = DEFAULT_ENCUT) -> str:
    """Generate INCAR for PBE-D3(BJ) calculation."""
    return f"""SYSTEM = {system_name}

# XC functional and dispersion
GGA    = PE
IVDW   = 12       # D3(BJ) dispersion correction

# Precision
PREC   = Accurate
ENCUT  = {encut}
LASPH  = .TRUE.   # Non-spherical contributions

# Electronic convergence
ALGO   = Normal
EDIFF  = 1E-7
ISMEAR = 0
SIGMA  = 0.05

# Ionic relaxation
IBRION = 2        # Conjugate gradient
ISIF   = 3        # Relax ions, cell shape, and volume
NSW    = 1500
EDIFFG = -0.01    # Force convergence (eV/Å)

# Performance
LREAL  = Auto
LWAVE  = .FALSE.
"""


def get_incar_pbe_mbd(system_name: str, encut: int = DEFAULT_ENCUT) -> str:
    """Generate INCAR for PBE-MBD calculation."""
    return f"""SYSTEM = {system_name}

# XC functional and dispersion
GGA    = PE
IVDW   = 263      # MBD@rsSCS dispersion correction

# Precision
PREC   = Accurate
ENCUT  = {encut}
LASPH  = .TRUE.   # Non-spherical contributions

# Electronic convergence
ALGO   = Normal
EDIFF  = 1E-7
ISMEAR = 0
SIGMA  = 0.05

# Ionic relaxation
IBRION = 2        # Conjugate gradient
ISIF   = 3        # Relax ions, cell shape, and volume
NSW    = 1500
EDIFFG = -0.01    # Force convergence (eV/Å)

# Performance
LREAL  = Auto
LWAVE  = .FALSE.
"""


# ==============================================================================
# SLURM Script
# ==============================================================================

def get_slurm_script(
    job_name: str,
    time_limit: str = "2-00:00:00",
    account: str = SLURM_ACCOUNT,
    qos: str = SLURM_QOS,
    nodes: int = SLURM_NODES,
    ntasks: int = SLURM_NTASKS,
    mem_per_cpu: str = SLURM_MEM_PER_CPU,
) -> str:
    """Generate SLURM submission script for VASP."""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time_limit}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --distribution=cyclic:cyclic
#SBATCH --account={account}
#SBATCH --qos={qos}

module purge
module use /home/mingjieliu/modules
module load vasp/6.4.3.cpu.base

export OMP_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=1
export SLURM_MPI_TYPE=pmix_v5
unset SLURM_MEM_PER_GPU
unset SLURM_MEM_PER_NODE

srun vasp_std > vasp.log
"""


# ==============================================================================
# Structure Processing
# ==============================================================================

def reorder_atoms_for_vasp(atoms: Atoms) -> Tuple[Atoms, List[Tuple[str, int]]]:
    """Reorder atoms by element for VASP compatibility."""
    symbols = atoms.get_chemical_symbols()
    
    # Get unique elements in order of first appearance
    elements = []
    for el in symbols:
        if el not in elements:
            elements.append(el)
    
    counts = Counter(symbols)
    
    # Reorder atoms
    new_symbols = []
    new_positions = []
    for elem in elements:
        for i, sym in enumerate(symbols):
            if sym == elem:
                new_symbols.append(sym)
                new_positions.append(atoms.positions[i])
    
    ordered_atoms = Atoms(
        symbols=new_symbols,
        positions=new_positions,
        cell=atoms.get_cell(),
        pbc=atoms.pbc
    )
    
    symbol_count = [(e, counts[e]) for e in elements]
    return ordered_atoms, symbol_count


def setup_vasp_job(
    atoms: Atoms,
    job_dir: Path,
    job_name: str,
    theory: str,  # "d3" or "mbd"
    encut: int = DEFAULT_ENCUT,
    kppa: int = DEFAULT_KPPA,
    potcar_root: str = DEFAULT_POTCAR_ROOT,
) -> bool:
    """Set up a single VASP job directory."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Reorder atoms
    ordered_atoms, symbol_count = reorder_atoms_for_vasp(atoms)
    elements = [e for e, _ in symbol_count]
    
    # POSCAR
    poscar_path = job_dir / "POSCAR"
    write_vasp(str(poscar_path), ordered_atoms, direct=True, vasp5=True, symbol_count=symbol_count)
    
    # INCAR
    incar_path = job_dir / "INCAR"
    if theory == "d3":
        incar_text = get_incar_pbe_d3(job_name, encut)
    else:
        incar_text = get_incar_pbe_mbd(job_name, encut)
    incar_path.write_text(incar_text)
    
    # KPOINTS
    kpoints_path = job_dir / "KPOINTS"
    nx, ny, nz = get_kpoints_pymatgen(atoms, kppa=kppa)
    kpoints_path.write_text(f"Automatic mesh\n0\nGamma\n{nx} {ny} {nz}\n0 0 0\n")
    
    # POTCAR
    potcar_path = job_dir / "POTCAR"
    potcar_root = Path(potcar_root)
    with open(potcar_path, "wb") as outf:
        for elem in elements:
            found = False
            for cand in [elem, f"{elem}_sv", f"{elem}_pv"]:
                p = potcar_root / cand / "POTCAR"
                if p.is_file():
                    outf.write(p.read_bytes())
                    found = True
                    break
            if not found:
                print(f"  ⚠️  Missing POTCAR for {elem}")
                return False
    
    # SLURM script
    slurm_path = job_dir / "run_vasp.slurm"
    time_limit = SLURM_TIME_D3 if theory == "d3" else SLURM_TIME_MBD
    slurm_path.write_text(get_slurm_script(job_name, time_limit=time_limit))
    
    return True


def create_submit_all_script(parent_dir: Path) -> None:
    """Create a script to submit all jobs in subdirectories."""
    script_path = parent_dir / "submit_all.sh"
    script_path.write_text("""#!/bin/bash
# Submit all VASP jobs in subdirectories
for d in */; do
    if [ -f "${d}run_vasp.slurm" ]; then
        echo "Submitting job in $d"
        (cd "$d" && sbatch run_vasp.slurm)
    fi
done
echo "All jobs submitted!"
""")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# ==============================================================================
# Main Setup Function
# ==============================================================================

def setup_dft_jobs(
    input_xyz: str,
    output_dir: Optional[str] = None,
    formula: Optional[str] = None,
    encut: int = DEFAULT_ENCUT,
    kppa: int = DEFAULT_KPPA,
    potcar_root: str = DEFAULT_POTCAR_ROOT,
    theories: List[str] = ["d3", "mbd"],
    verbose: bool = True,
) -> dict:
    """
    Set up DFT jobs for all structures in an XYZ file.
    
    Args:
        input_xyz: Path to input XYZ file with structures
        output_dir: Base output directory (default: ./<formula>/)
        formula: Chemical formula (auto-detected if not provided)
        encut: VASP ENCUT parameter
        kppa: k-points per atom for pymatgen
        potcar_root: Path to POTCAR library
        theories: List of theories to set up ("d3", "mbd")
        verbose: Print progress
        
    Returns:
        Dictionary with setup statistics
    """
    input_xyz = Path(input_xyz)
    
    if not input_xyz.exists():
        raise FileNotFoundError(f"Input file not found: {input_xyz}")
    
    # Load structures
    structures = io.read(str(input_xyz), index=":")
    n_structures = len(structures)
    
    if verbose:
        print(f"Loaded {n_structures} structures from {input_xyz}")
    
    # Auto-detect formula from first structure
    if formula is None:
        first_atoms = structures[0]
        formula = first_atoms.get_chemical_formula()
        if verbose:
            print(f"Auto-detected formula: {formula}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(".") / formula
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "formula": formula,
        "n_structures": n_structures,
        "theories": {},
    }
    
    # Set up jobs for each theory
    for theory in theories:
        theory_dir = output_dir / f"pbe-{theory}"
        theory_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"\nSetting up PBE-{theory.upper()} jobs in {theory_dir}/")
        
        success_count = 0
        for idx, atoms in enumerate(structures):
            job_name = f"{formula}_pbe_{theory}_{idx:03d}"
            job_dir = theory_dir / f"{idx:03d}"
            
            success = setup_vasp_job(
                atoms=atoms,
                job_dir=job_dir,
                job_name=job_name,
                theory=theory,
                encut=encut,
                kppa=kppa,
                potcar_root=potcar_root,
            )
            
            if success:
                success_count += 1
            
            if verbose and (idx + 1) % 10 == 0:
                print(f"  Set up {idx + 1}/{n_structures} jobs")
        
        # Create submit script
        create_submit_all_script(theory_dir)
        
        stats["theories"][theory] = {
            "dir": str(theory_dir),
            "n_jobs": success_count,
        }
        
        if verbose:
            print(f"  ✅ Created {success_count} jobs in {theory_dir}/")
    
    # Create master submit script
    master_script = output_dir / "submit_all_dft.sh"
    master_script.write_text(f"""#!/bin/bash
# Submit all DFT jobs for {formula}

echo "Submitting PBE-D3 jobs..."
(cd pbe-d3 && bash submit_all.sh)

echo "Submitting PBE-MBD jobs..."
(cd pbe-mbd && bash submit_all.sh)

echo "All DFT jobs submitted for {formula}!"
""")
    master_script.chmod(master_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    
    if verbose:
        print(f"\n{'='*60}")
        print("DFT Job Setup Complete")
        print(f"{'='*60}")
        print(f"  Formula: {formula}")
        print(f"  Structures: {n_structures}")
        print(f"  Output: {output_dir}/")
        for theory, info in stats["theories"].items():
            print(f"    - pbe-{theory}/: {info['n_jobs']} jobs")
        print(f"\n  To submit all jobs:")
        print(f"    cd {output_dir} && bash submit_all_dft.sh")
        print(f"{'='*60}")
    
    return stats


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Setup VASP DFT jobs for UMA-OMC relaxed structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup jobs from relaxation results
  python setup_dft_jobs.py --input ../uma-s1p1-omc-opt/C9H9N3O5/results/top10_for_dft.xyz

  # Specify output directory
  python setup_dft_jobs.py --input top10.xyz --output_dir ./C9H9N3O5_dft

  # Only PBE-D3 (no MBD)
  python setup_dft_jobs.py --input top10.xyz --theories d3

Output Structure:
  <formula>/
  ├── pbe-d3/
  │   ├── 000/, 001/, ...
  │   └── submit_all.sh
  ├── pbe-mbd/
  │   ├── 000/, 001/, ...
  │   └── submit_all.sh
  └── submit_all_dft.sh
        """,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input XYZ file with top-N structures",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: ./<formula>/)",
    )
    parser.add_argument(
        "--formula", "-f",
        type=str,
        default=None,
        help="Chemical formula (auto-detected if not provided)",
    )
    parser.add_argument(
        "--theories", "-t",
        type=str,
        nargs="+",
        default=["d3", "mbd"],
        choices=["d3", "mbd"],
        help="DFT theories to set up (default: d3 mbd)",
    )
    parser.add_argument(
        "--encut",
        type=int,
        default=DEFAULT_ENCUT,
        help=f"VASP ENCUT parameter (default: {DEFAULT_ENCUT})",
    )
    parser.add_argument(
        "--kppa",
        type=int,
        default=DEFAULT_KPPA,
        help=f"k-points per atom (default: {DEFAULT_KPPA})",
    )
    parser.add_argument(
        "--potcar_root",
        type=str,
        default=DEFAULT_POTCAR_ROOT,
        help="Path to POTCAR library",
    )
    
    args = parser.parse_args()
    
    setup_dft_jobs(
        input_xyz=args.input,
        output_dir=args.output_dir,
        formula=args.formula,
        encut=args.encut,
        kppa=args.kppa,
        potcar_root=args.potcar_root,
        theories=args.theories,
        verbose=True,
    )


if __name__ == "__main__":
    main()
