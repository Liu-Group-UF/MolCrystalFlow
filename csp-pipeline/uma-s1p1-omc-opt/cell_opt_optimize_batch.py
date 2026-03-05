#!/usr/bin/env python3
"""
Cell optimization batch script using UMA-OMC model.

Performs full atomic + cell relaxation (no rigid body constraints).

Usage:
    python cell_opt_optimize_batch.py --start 0 --end 10 --input filtered_by_inter_bb.xyz
"""
import warnings
warnings.filterwarnings("ignore")
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import argparse
from ase.io import read, write
from ase.optimize import BFGS
from utils.rigid_constraint import add_rigid_body_constraints
from ase.filters import FrechetCellFilter
import os

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omc")

def sanitize_atoms_for_fairchem(atoms):
    # Charge
    if "charge" not in atoms.info:
        atoms.info["charge"] = 0
    atoms.info["charge"] = int(atoms.info["charge"])

    # FAIR-Chem sometimes expects "spin" instead of "spin_multiplicity"
    if "spin" in atoms.info:
        atoms.info["spin"] = int(atoms.info["spin"])
    elif "spin_multiplicity" in atoms.info:
        atoms.info["spin"] = int(atoms.info["spin_multiplicity"])
    else:
        atoms.info["spin"] = 1  # default singlet

    return atoms

def main(start, end, input_xyz):
    # Load only the requested slice of structures
    atoms_list = read(input_xyz, format="extxyz", index=f"{start}:{end}")

    optimized_structures = []
    steps = 1000
    for i, atoms in enumerate(atoms_list, start=start):
        # Clear constraints
        atoms = sanitize_atoms_for_fairchem(atoms)
        atoms.set_constraint([])

        # Add rigid body constraints (your function)
        # constraints = add_rigid_body_constraints(atoms)
        # atoms.set_constraint(constraints)

        # Attach calculator
        atoms.calc = calc
        fcf = FrechetCellFilter(atoms)
        os.makedirs("cell-opt-logs", exist_ok=True)
        opt=BFGS(fcf, logfile=f"cell-opt-logs/continue_opt_{i:05d}.log")
        # opt=BFGS(fcf)
        opt.run(fmax=0.01, steps=steps)
        # Run optimization (100 steps)
        
        atoms.set_constraint([])
        optimizer = BFGS(atoms)
        optimizer.run(steps=steps, fmax=0.01)

        # Save the final optimized structure
        atoms.set_constraint([])
        optimized_structures.append(atoms.copy())

    # Write all optimized structures for this chunk
    os.makedirs("cell-opt-results", exist_ok=True)
    outfile = f"./cell-opt-results/pred_omc25_{steps}step_cell_opt_{start:05d}_{end:05d}.extxyz"
    write(outfile, optimized_structures, format="extxyz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="Start index of slice")
    parser.add_argument("--end", type=int, required=True, help="End index of slice")
    parser.add_argument("--input", type=str, default="./filtered_by_inter_bb.xyz", help="Input XYZ file")
    args = parser.parse_args()

    main(args.start, args.end, args.input)

